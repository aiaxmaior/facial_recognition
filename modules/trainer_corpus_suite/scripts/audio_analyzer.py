#!/usr/bin/env python3
"""
Unified Audio Analyzer
======================

Single audio processing pipeline that extracts all features and embeddings,
then formats output for either professional (CoT reasoning) or hobby 
(diffusion conditioning) contexts.

Features:
- Wav2Vec2-large for general audio embeddings
- pyannote VAD for speech/non-speech segmentation
- CLAP for audio-text alignment embeddings
- emotion2vec for affective audio embeddings
- librosa for interpretable DSP features
- Non-verbal cue classifier for professional track

Usage:
    python audio_analyzer.py --output both          # Default: both outputs
    python audio_analyzer.py --output pro           # Professional only
    python audio_analyzer.py --output hobby         # Hobby/diffusion only
    python audio_analyzer.py --test 10              # Process first 10 clips
    python audio_analyzer.py --skip-clap            # Skip CLAP model
    python audio_analyzer.py --skip-emotion         # Skip emotion2vec

CRITICAL: This script processes files WITHOUT interpreting content.
All operations are based on model outputs and technical metrics only.
"""

import argparse
import json
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Local imports
from audio_utils import (
    extract_audio_robust,
    load_audio,
    get_scenes_for_processing,
    probe_audio_stream,
    AudioExtractionResult,
    AudioData,
    DEFAULT_SAMPLE_RATE,
    format_duration
)
from robust_processor import IncrementalJSONWriter, NumpyEncoder


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths
DEFAULT_SCENES_DIR = Path(__file__).parent.parent / "scenes"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
DEFAULT_EMBEDDINGS_DIR = Path(__file__).parent.parent / "audio_embeddings"
DEFAULT_DETECTIONS_FILE = Path(__file__).parent.parent / "analysis" / "detections.json"

# Model identifiers (HuggingFace)
WAV2VEC2_MODEL = "facebook/wav2vec2-large-960h"
CLAP_MODEL = "laion/larger_clap_general"
EMOTION2VEC_MODEL = "iic/emotion2vec_base"

# Processing settings
AUDIO_TEMP_DIR = Path(tempfile.gettempdir()) / "audio_analysis"
PROCESSING_TIMEOUT = 120  # Per-scene timeout

# Non-verbal cue categories (for professional track)
NON_VERBAL_CUES = [
    'sigh',       # Breath-based exhalation
    'groan',      # Low, strained vocalization
    'moan',       # Sustained vocalization
    'agitation',  # Rapid, erratic sounds
    'whimper',    # Soft, high-pitched
    'speech',     # Verbal content
    'silence',    # Absence of audio
    'ambient'     # Background noise only
]

# Valence hints for non-verbal cues
CUE_VALENCE_MAP = {
    'sigh': 'neutral',       # Context-dependent
    'groan': 'negative',     # Typically discomfort
    'moan': 'ambiguous',     # Requires visual context
    'agitation': 'negative', # Distress indicator
    'whimper': 'negative',   # Fear/sadness
    'speech': 'neutral',     # Context-dependent
    'silence': 'neutral',
    'ambient': 'neutral'
}


# =============================================================================
# MODEL LOADERS (Lazy Loading)
# =============================================================================

class ModelManager:
    """
    Manages lazy loading of audio models.
    Models are loaded on first use to reduce startup time.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self._wav2vec2 = None
        self._wav2vec2_processor = None
        self._vad_pipeline = None
        self._clap_model = None
        self._clap_processor = None
        self._emotion2vec = None
        
        # Track what's available
        self.wav2vec2_available = False
        self.vad_available = False
        self.clap_available = False
        self.emotion2vec_available = False
    
    def load_wav2vec2(self) -> bool:
        """Load Wav2Vec2 model."""
        if self._wav2vec2 is not None:
            return self.wav2vec2_available
        
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            import torch
            
            print(f"  Loading Wav2Vec2 ({WAV2VEC2_MODEL})...")
            self._wav2vec2_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL)
            self._wav2vec2 = Wav2Vec2Model.from_pretrained(WAV2VEC2_MODEL)
            self._wav2vec2.to(self.device)
            self._wav2vec2.eval()
            self.wav2vec2_available = True
            print("    Wav2Vec2 loaded successfully")
            return True
        except Exception as e:
            print(f"    Warning: Could not load Wav2Vec2: {e}")
            self.wav2vec2_available = False
            return False
    
    def load_vad(self) -> bool:
        """Load pyannote VAD pipeline."""
        if self._vad_pipeline is not None:
            return self.vad_available
        
        try:
            from pyannote.audio import Pipeline
            import torch
            
            print("  Loading pyannote VAD...")
            # Note: Requires HF token for pyannote models
            # User should set HF_TOKEN environment variable or use huggingface-cli login
            self._vad_pipeline = Pipeline.from_pretrained(
                "pyannote/voice-activity-detection",
                use_auth_token=os.environ.get('HF_TOKEN')
            )
            if torch.cuda.is_available() and 'cuda' in self.device:
                self._vad_pipeline.to(torch.device(self.device))
            self.vad_available = True
            print("    VAD loaded successfully")
            return True
        except Exception as e:
            print(f"    Warning: Could not load VAD: {e}")
            print("    (May need: huggingface-cli login or set HF_TOKEN)")
            self.vad_available = False
            return False
    
    def load_clap(self) -> bool:
        """Load CLAP model."""
        if self._clap_model is not None:
            return self.clap_available
        
        try:
            from transformers import ClapModel, ClapProcessor
            import torch
            
            print(f"  Loading CLAP ({CLAP_MODEL})...")
            self._clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)
            self._clap_model = ClapModel.from_pretrained(CLAP_MODEL)
            self._clap_model.to(self.device)
            self._clap_model.eval()
            self.clap_available = True
            print("    CLAP loaded successfully")
            return True
        except Exception as e:
            print(f"    Warning: Could not load CLAP: {e}")
            self.clap_available = False
            return False
    
    def load_emotion2vec(self) -> bool:
        """Load emotion2vec model."""
        if self._emotion2vec is not None:
            return self.emotion2vec_available
        
        try:
            from funasr import AutoModel
            
            print(f"  Loading emotion2vec ({EMOTION2VEC_MODEL})...")
            self._emotion2vec = AutoModel(model=EMOTION2VEC_MODEL)
            self.emotion2vec_available = True
            print("    emotion2vec loaded successfully")
            return True
        except Exception as e:
            print(f"    Warning: Could not load emotion2vec: {e}")
            self.emotion2vec_available = False
            return False
    
    @property
    def wav2vec2(self):
        if not self.wav2vec2_available:
            self.load_wav2vec2()
        return self._wav2vec2, self._wav2vec2_processor
    
    @property
    def vad(self):
        if not self.vad_available:
            self.load_vad()
        return self._vad_pipeline
    
    @property
    def clap(self):
        if not self.clap_available:
            self.load_clap()
        return self._clap_model, self._clap_processor
    
    @property
    def emotion2vec(self):
        if not self.emotion2vec_available:
            self.load_emotion2vec()
        return self._emotion2vec


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_wav2vec2_embedding(
    audio: AudioData,
    model_manager: ModelManager
) -> Optional[np.ndarray]:
    """
    Extract Wav2Vec2 embeddings from audio.
    
    Args:
        audio: Loaded audio data
        model_manager: Model manager instance
        
    Returns:
        Embedding array (T, 1024) or pooled (1024,), or None on failure
    """
    if not model_manager.wav2vec2_available:
        return None
    
    try:
        import torch
        model, processor = model_manager.wav2vec2
        
        # Process audio
        inputs = processor(
            audio.samples,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Get last hidden state
            hidden_states = outputs.last_hidden_state
            
            # Mean pooling over time
            embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding
        
    except Exception as e:
        print(f"    Wav2Vec2 extraction error: {e}")
        return None


def extract_vad_segments(
    wav_path: Path,
    model_manager: ModelManager
) -> Optional[List[Dict]]:
    """
    Extract VAD segments using pyannote.
    
    Args:
        wav_path: Path to WAV file
        model_manager: Model manager instance
        
    Returns:
        List of segment dicts with start, end, type
    """
    if not model_manager.vad_available:
        return None
    
    try:
        pipeline = model_manager.vad
        
        # Run VAD
        vad_result = pipeline(str(wav_path))
        
        segments = []
        for segment, _, label in vad_result.itertracks(yield_label=True):
            segments.append({
                'start': round(segment.start, 3),
                'end': round(segment.end, 3),
                'duration': round(segment.end - segment.start, 3),
                'type': 'speech' if label == 'SPEECH' else 'non_speech'
            })
        
        return segments
        
    except Exception as e:
        print(f"    VAD extraction error: {e}")
        return None


def extract_clap_embedding(
    audio: AudioData,
    model_manager: ModelManager
) -> Optional[np.ndarray]:
    """
    Extract CLAP audio embedding.
    
    Args:
        audio: Loaded audio data
        model_manager: Model manager instance
        
    Returns:
        CLAP embedding (512,) or None on failure
    """
    if not model_manager.clap_available:
        return None
    
    try:
        import torch
        model, processor = model_manager.clap
        
        # CLAP expects 48kHz audio, resample if needed
        target_sr = 48000
        if audio.sample_rate != target_sr:
            import librosa
            samples = librosa.resample(
                audio.samples,
                orig_sr=audio.sample_rate,
                target_sr=target_sr
            )
        else:
            samples = audio.samples
        
        # Process
        inputs = processor(
            audios=samples,
            sampling_rate=target_sr,
            return_tensors="pt"
        )
        inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            audio_embed = model.get_audio_features(**inputs)
            embedding = audio_embed.squeeze().cpu().numpy()
        
        return embedding
        
    except Exception as e:
        print(f"    CLAP extraction error: {e}")
        return None


def extract_emotion2vec_features(
    wav_path: Path,
    model_manager: ModelManager
) -> Optional[Dict]:
    """
    Extract emotion2vec features.
    
    Args:
        wav_path: Path to WAV file
        model_manager: Model manager instance
        
    Returns:
        Dict with emotion predictions or None on failure
    """
    if not model_manager.emotion2vec_available:
        return None
    
    try:
        model = model_manager.emotion2vec
        
        # Run inference
        result = model.generate(str(wav_path), granularity="utterance")
        
        if result and len(result) > 0:
            # Extract scores
            scores = result[0].get('scores', {})
            labels = result[0].get('labels', [])
            
            # Build emotion dict
            emotions = {}
            if isinstance(scores, (list, np.ndarray)) and labels:
                for i, label in enumerate(labels):
                    if i < len(scores):
                        emotions[label] = float(scores[i])
            
            # Get dominant emotion
            if emotions:
                dominant = max(emotions, key=emotions.get)
                return {
                    'emotions': emotions,
                    'dominant_emotion': dominant,
                    'confidence': emotions.get(dominant, 0.0)
                }
        
        return None
        
    except Exception as e:
        print(f"    emotion2vec extraction error: {e}")
        return None


def extract_librosa_features(audio: AudioData) -> Dict:
    """
    Extract interpretable audio features using librosa.
    
    Args:
        audio: Loaded audio data
        
    Returns:
        Dict of features
    """
    try:
        import librosa
        
        samples = audio.samples
        sr = audio.sample_rate
        
        features = {}
        
        # Pitch (F0) estimation
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                samples,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            if len(f0_valid) > 0:
                features['pitch'] = {
                    'mean': float(np.mean(f0_valid)),
                    'std': float(np.std(f0_valid)),
                    'min': float(np.min(f0_valid)),
                    'max': float(np.max(f0_valid)),
                    'voiced_ratio': float(np.mean(voiced_flag))
                }
                # Trend detection
                if len(f0_valid) > 10:
                    first_half = np.mean(f0_valid[:len(f0_valid)//2])
                    second_half = np.mean(f0_valid[len(f0_valid)//2:])
                    if second_half > first_half * 1.1:
                        features['pitch']['trend'] = 'rising'
                    elif second_half < first_half * 0.9:
                        features['pitch']['trend'] = 'falling'
                    else:
                        features['pitch']['trend'] = 'stable'
        except Exception:
            pass
        
        # Energy / RMS
        rms = librosa.feature.rms(y=samples)[0]
        features['energy'] = {
            'mean_rms': float(np.mean(rms)),
            'max_rms': float(np.max(rms)),
            'std_rms': float(np.std(rms))
        }
        # Energy trend
        if len(rms) > 10:
            first_half = np.mean(rms[:len(rms)//2])
            second_half = np.mean(rms[len(rms)//2:])
            if second_half > first_half * 1.2:
                features['energy']['trend'] = 'building'
            elif second_half < first_half * 0.8:
                features['energy']['trend'] = 'diminishing'
            else:
                features['energy']['trend'] = 'sustained'
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=samples, sr=sr)[0]
        features['spectral_centroid'] = {
            'mean': float(np.mean(spectral_centroid)),
            'std': float(np.std(spectral_centroid))
        }
        
        # Tempo / rhythm
        try:
            tempo, _ = librosa.beat.beat_track(y=samples, sr=sr)
            features['tempo_bpm'] = float(tempo)
        except Exception:
            features['tempo_bpm'] = None
        
        # Onset detection (vocal bursts)
        try:
            onsets = librosa.onset.onset_detect(y=samples, sr=sr)
            features['onset_count'] = int(len(onsets))
            features['onset_rate'] = float(len(onsets) / audio.duration_seconds)
        except Exception:
            features['onset_count'] = 0
            features['onset_rate'] = 0.0
        
        # Zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(samples)[0]
        features['zero_crossing_rate'] = float(np.mean(zcr))
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=samples, sr=sr)[0]
        features['spectral_rolloff'] = float(np.mean(rolloff))
        
        return features
        
    except Exception as e:
        print(f"    librosa feature extraction error: {e}")
        return {}


# =============================================================================
# NON-VERBAL CLASSIFIER (Professional Track)
# =============================================================================

def classify_non_verbal_cue(
    vad_segments: Optional[List[Dict]],
    librosa_features: Dict,
    audio: AudioData
) -> Dict:
    """
    Classify non-verbal cues based on VAD and acoustic features.
    
    This is a rule-based classifier. Could be replaced with a trained model.
    
    Args:
        vad_segments: VAD segmentation results
        librosa_features: Extracted librosa features
        audio: Audio data
        
    Returns:
        Classification result dict
    """
    result = {
        'dominant_cue': 'ambient',
        'confidence': 0.5,
        'valence_hint': 'neutral',
        'cue_segments': []
    }
    
    # Check for silence
    if audio.is_silent:
        result['dominant_cue'] = 'silence'
        result['confidence'] = 0.95
        return result
    
    # Analyze VAD results
    speech_ratio = 0.0
    non_speech_ratio = 0.0
    
    if vad_segments:
        total_duration = audio.duration_seconds
        speech_duration = sum(
            s['duration'] for s in vad_segments if s['type'] == 'speech'
        )
        non_speech_duration = sum(
            s['duration'] for s in vad_segments if s['type'] == 'non_speech'
        )
        
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
        non_speech_ratio = non_speech_duration / total_duration if total_duration > 0 else 0
    
    result['speech_ratio'] = round(speech_ratio, 3)
    result['non_verbal_ratio'] = round(non_speech_ratio, 3)
    
    # If mostly speech, classify as speech
    if speech_ratio > 0.6:
        result['dominant_cue'] = 'speech'
        result['confidence'] = min(0.95, speech_ratio)
        result['valence_hint'] = 'neutral'
        return result
    
    # Analyze acoustic features for non-verbal classification
    pitch = librosa_features.get('pitch', {})
    energy = librosa_features.get('energy', {})
    
    mean_pitch = pitch.get('mean', 0)
    pitch_std = pitch.get('std', 0)
    energy_trend = energy.get('trend', 'sustained')
    onset_rate = librosa_features.get('onset_rate', 0)
    
    # Rule-based classification
    # High pitch variability + building energy -> agitation
    if pitch_std > 50 and energy_trend == 'building' and onset_rate > 2:
        result['dominant_cue'] = 'agitation'
        result['confidence'] = 0.7
        result['valence_hint'] = 'negative'
    
    # Low pitch + sustained energy -> groan
    elif mean_pitch < 200 and mean_pitch > 0 and energy_trend == 'sustained':
        result['dominant_cue'] = 'groan'
        result['confidence'] = 0.65
        result['valence_hint'] = 'negative'
    
    # Medium pitch + rising trend -> moan
    elif 200 <= mean_pitch <= 400 and pitch.get('trend') == 'rising':
        result['dominant_cue'] = 'moan'
        result['confidence'] = 0.6
        result['valence_hint'] = 'ambiguous'
    
    # High pitch + diminishing energy -> whimper
    elif mean_pitch > 350 and energy_trend == 'diminishing':
        result['dominant_cue'] = 'whimper'
        result['confidence'] = 0.6
        result['valence_hint'] = 'negative'
    
    # Brief energy burst -> sigh
    elif energy.get('max_rms', 0) > 2 * energy.get('mean_rms', 0.1) and onset_rate < 1:
        result['dominant_cue'] = 'sigh'
        result['confidence'] = 0.55
        result['valence_hint'] = 'neutral'
    
    # Default: ambient
    else:
        result['dominant_cue'] = 'ambient'
        result['confidence'] = 0.5
        result['valence_hint'] = 'neutral'
    
    return result


# =============================================================================
# OUTPUT FORMATTERS
# =============================================================================

def format_pro_output(
    scene_path: str,
    audio: Optional[AudioData],
    vad_segments: Optional[List[Dict]],
    librosa_features: Dict,
    classification: Dict,
    error: Optional[Dict] = None
) -> Dict:
    """
    Format output for professional track (CoT reasoning context).
    
    Args:
        scene_path: Path to source scene
        audio: Loaded audio data
        vad_segments: VAD segmentation
        librosa_features: Librosa features
        classification: Non-verbal classification
        error: Error info if processing failed
        
    Returns:
        Professional track output dict
    """
    if error:
        return {
            'scene_path': scene_path,
            'audio_present': False,
            'processing_status': error.get('type', 'error'),
            'error': error,
            'cot_context': f"No audio analysis available: {error.get('message', 'unknown error')}"
        }
    
    if audio is None:
        return {
            'scene_path': scene_path,
            'audio_present': False,
            'processing_status': 'no_audio',
            'cot_context': "No audio track present. Emotional assessment relies solely on visual analysis."
        }
    
    # Build acoustic profile
    acoustic_profile = {}
    if 'pitch' in librosa_features:
        acoustic_profile['mean_pitch_hz'] = round(librosa_features['pitch'].get('mean', 0), 1)
        acoustic_profile['pitch_trend'] = librosa_features['pitch'].get('trend', 'unknown')
    if 'energy' in librosa_features:
        acoustic_profile['energy_trend'] = librosa_features['energy'].get('trend', 'unknown')
        acoustic_profile['intensity_rms'] = round(librosa_features['energy'].get('mean_rms', 0), 4)
    
    # Build segmentation summary
    segmentation = {
        'speech_ratio': classification.get('speech_ratio', 0),
        'non_verbal_ratio': classification.get('non_verbal_ratio', 0),
        'silence_ratio': round(1 - classification.get('speech_ratio', 0) - classification.get('non_verbal_ratio', 0), 3)
    }
    if vad_segments:
        segmentation['segment_count'] = len(vad_segments)
    
    # Generate CoT context string
    cot_parts = []
    
    dom_cue = classification.get('dominant_cue', 'ambient')
    confidence = classification.get('confidence', 0.5)
    
    if dom_cue == 'silence':
        cot_parts.append("Audio track is silent or near-silent.")
    elif dom_cue == 'speech':
        cot_parts.append(f"Primarily speech content ({segmentation['speech_ratio']*100:.0f}% of audio).")
    else:
        nv_pct = segmentation.get('non_verbal_ratio', 0) * 100
        cot_parts.append(f"Non-verbal vocalization detected: {dom_cue} ({confidence*100:.0f}% confidence).")
        if nv_pct > 50:
            cot_parts.append(f"Non-verbal content comprises {nv_pct:.0f}% of audio.")
    
    # Add acoustic context
    if acoustic_profile.get('pitch_trend') and acoustic_profile['pitch_trend'] != 'unknown':
        cot_parts.append(f"Pitch trend: {acoustic_profile['pitch_trend']}.")
    if acoustic_profile.get('energy_trend') and acoustic_profile['energy_trend'] != 'unknown':
        cot_parts.append(f"Intensity pattern: {acoustic_profile['energy_trend']}.")
    
    # Add valence hint
    valence = classification.get('valence_hint', 'neutral')
    if valence == 'negative':
        cot_parts.append("Acoustic profile suggests distress or discomfort.")
    elif valence == 'ambiguous':
        cot_parts.append("Valence ambiguous from audio alone; visual context required for disambiguation.")
    
    cot_context = " ".join(cot_parts)
    
    return {
        'scene_path': scene_path,
        'audio_present': True,
        'duration_seconds': round(audio.duration_seconds, 2),
        'processing_status': 'success',
        'segmentation': segmentation,
        'acoustic_profile': acoustic_profile,
        'classification': {
            'dominant_cue': dom_cue,
            'confidence': round(confidence, 3),
            'valence_hint': valence
        },
        'cot_context': cot_context
    }


def format_hobby_output(
    scene_path: str,
    scene_name: str,
    audio: Optional[AudioData],
    wav2vec_embedding: Optional[np.ndarray],
    clap_embedding: Optional[np.ndarray],
    emotion2vec_result: Optional[Dict],
    vad_segments: Optional[List[Dict]],
    librosa_features: Dict,
    embeddings_dir: Path,
    model_errors: Dict,
    error: Optional[Dict] = None
) -> Dict:
    """
    Format output for hobby/diffusion track.
    
    Args:
        scene_path: Path to source scene
        scene_name: Scene filename stem
        audio: Loaded audio data
        wav2vec_embedding: Wav2Vec2 embedding
        clap_embedding: CLAP embedding
        emotion2vec_result: emotion2vec results
        vad_segments: VAD segmentation
        librosa_features: Librosa features
        embeddings_dir: Directory to save embeddings
        model_errors: Dict of model name -> error info
        error: Error info if extraction failed
        
    Returns:
        Hobby track output dict
    """
    if error:
        return {
            'scene_path': scene_path,
            'audio_present': False,
            'processing_status': error.get('type', 'error'),
            'error': error,
            'embeddings': None,
            'librosa_features': None
        }
    
    if audio is None:
        return {
            'scene_path': scene_path,
            'audio_present': False,
            'processing_status': 'no_audio',
            'error': {'type': 'no_audio_stream', 'message': 'Video contains no audio track'},
            'embeddings': None,
            'librosa_features': None
        }
    
    # Save embeddings and build paths dict
    embeddings = {}
    embedding_stats = {}
    
    if wav2vec_embedding is not None:
        emb_path = embeddings_dir / f"{scene_name}_wav2vec2.npy"
        np.save(emb_path, wav2vec_embedding)
        embeddings['wav2vec2'] = str(emb_path)
        embedding_stats['wav2vec2_norm'] = float(np.linalg.norm(wav2vec_embedding))
    else:
        embeddings['wav2vec2'] = None
    
    if clap_embedding is not None:
        emb_path = embeddings_dir / f"{scene_name}_clap.npy"
        np.save(emb_path, clap_embedding)
        embeddings['clap'] = str(emb_path)
        embedding_stats['clap_norm'] = float(np.linalg.norm(clap_embedding))
    else:
        embeddings['clap'] = None
    
    if emotion2vec_result is not None:
        # Save emotion2vec embedding if available
        embeddings['emotion2vec'] = None  # emotion2vec returns scores, not embeddings
        embedding_stats['emotion2vec_dominant'] = emotion2vec_result.get('dominant_emotion')
        embedding_stats['emotion2vec_confidence'] = emotion2vec_result.get('confidence')
    else:
        embeddings['emotion2vec'] = None
    
    # VAD-based features
    vad_features = {}
    if vad_segments:
        speech_segs = [s for s in vad_segments if s['type'] == 'speech']
        non_speech_segs = [s for s in vad_segments if s['type'] == 'non_speech']
        
        speech_duration = sum(s['duration'] for s in speech_segs)
        non_speech_duration = sum(s['duration'] for s in non_speech_segs)
        total = audio.duration_seconds
        
        vad_features = {
            'speech_ratio': round(speech_duration / total, 3) if total > 0 else 0,
            'non_speech_ratio': round(non_speech_duration / total, 3) if total > 0 else 0,
            'segment_count': len(vad_segments),
            'speech_segments': len(speech_segs),
            'non_speech_segments': len(non_speech_segs)
        }
    
    # Determine processing status
    if model_errors:
        status = 'partial'
    else:
        status = 'success'
    
    # Build diffusion conditioning hints
    energy_trend = librosa_features.get('energy', {}).get('trend', 'unknown')
    intensity_class = 'medium'
    if librosa_features.get('energy', {}).get('mean_rms', 0) > 0.15:
        intensity_class = 'high'
    elif librosa_features.get('energy', {}).get('mean_rms', 0) < 0.05:
        intensity_class = 'low'
    
    diffusion_conditioning = {
        'intensity_class': intensity_class,
        'trajectory': energy_trend if energy_trend != 'unknown' else 'stable',
        'has_speech': vad_features.get('speech_ratio', 0) > 0.3,
        'vocalization_density': librosa_features.get('onset_rate', 0)
    }
    
    output = {
        'scene_path': scene_path,
        'audio_present': True,
        'duration_seconds': round(audio.duration_seconds, 2),
        'processing_status': status,
        'embeddings': embeddings,
        'embedding_stats': embedding_stats,
        'vad_features': vad_features,
        'librosa_features': librosa_features,
        'diffusion_conditioning': diffusion_conditioning
    }
    
    if model_errors:
        output['model_errors'] = model_errors
    
    return output


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_scene(
    scene_path: Path,
    model_manager: ModelManager,
    embeddings_dir: Path,
    output_mode: str = 'both',
    temp_dir: Path = AUDIO_TEMP_DIR
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Process a single scene through the audio pipeline.
    
    Args:
        scene_path: Path to video file
        model_manager: Model manager instance
        embeddings_dir: Directory for embedding files
        output_mode: 'pro', 'hobby', or 'both'
        temp_dir: Directory for temporary audio files
        
    Returns:
        (pro_output, hobby_output) - either can be None based on output_mode
    """
    scene_name = scene_path.stem
    
    # Step 1: Extract audio
    extraction = extract_audio_robust(
        scene_path,
        output_dir=temp_dir,
        timeout_seconds=PROCESSING_TIMEOUT
    )
    
    if not extraction.success:
        error = {
            'type': extraction.error_type,
            'message': extraction.error_message
        }
        
        pro_out = None
        hobby_out = None
        
        if output_mode in ['pro', 'both']:
            pro_out = format_pro_output(str(scene_path), None, None, {}, {}, error)
        if output_mode in ['hobby', 'both']:
            hobby_out = format_hobby_output(
                str(scene_path), scene_name, None, None, None, None, None, {},
                embeddings_dir, {}, error
            )
        
        return pro_out, hobby_out
    
    # Step 2: Load audio
    audio = load_audio(extraction.wav_path)
    if audio is None:
        error = {'type': 'load_failed', 'message': 'Could not load extracted audio'}
        
        pro_out = None
        hobby_out = None
        
        if output_mode in ['pro', 'both']:
            pro_out = format_pro_output(str(scene_path), None, None, {}, {}, error)
        if output_mode in ['hobby', 'both']:
            hobby_out = format_hobby_output(
                str(scene_path), scene_name, None, None, None, None, None, {},
                embeddings_dir, {}, error
            )
        
        # Cleanup
        extraction.wav_path.unlink(missing_ok=True)
        return pro_out, hobby_out
    
    # Step 3: Extract features (all models)
    model_errors = {}
    
    # Wav2Vec2
    wav2vec_emb = None
    if model_manager.wav2vec2_available:
        wav2vec_emb = extract_wav2vec2_embedding(audio, model_manager)
        if wav2vec_emb is None:
            model_errors['wav2vec2'] = {'type': 'inference_failed', 'message': 'Extraction returned None'}
    
    # VAD
    vad_segments = None
    if model_manager.vad_available:
        vad_segments = extract_vad_segments(extraction.wav_path, model_manager)
        if vad_segments is None:
            model_errors['vad'] = {'type': 'inference_failed', 'message': 'VAD returned None'}
    
    # CLAP
    clap_emb = None
    if model_manager.clap_available:
        clap_emb = extract_clap_embedding(audio, model_manager)
        if clap_emb is None:
            model_errors['clap'] = {'type': 'inference_failed', 'message': 'Extraction returned None'}
    
    # emotion2vec
    emotion_result = None
    if model_manager.emotion2vec_available:
        emotion_result = extract_emotion2vec_features(extraction.wav_path, model_manager)
        if emotion_result is None:
            model_errors['emotion2vec'] = {'type': 'inference_failed', 'message': 'Extraction returned None'}
    
    # librosa features (always available)
    librosa_features = extract_librosa_features(audio)
    
    # Step 4: Classification (for pro track)
    classification = classify_non_verbal_cue(vad_segments, librosa_features, audio)
    
    # Step 5: Format outputs
    pro_out = None
    hobby_out = None
    
    if output_mode in ['pro', 'both']:
        pro_out = format_pro_output(
            str(scene_path), audio, vad_segments, librosa_features, classification
        )
    
    if output_mode in ['hobby', 'both']:
        hobby_out = format_hobby_output(
            str(scene_path), scene_name, audio, wav2vec_emb, clap_emb,
            emotion_result, vad_segments, librosa_features, embeddings_dir, model_errors
        )
    
    # Cleanup temp audio
    extraction.wav_path.unlink(missing_ok=True)
    
    return pro_out, hobby_out


def process_all_scenes(
    scenes_dir: Path,
    output_dir: Path,
    embeddings_dir: Path,
    output_mode: str = 'both',
    skip_clap: bool = False,
    skip_emotion: bool = False,
    skip_vad: bool = False,
    max_clips: Optional[int] = None,
    device: str = 'cuda'
) -> Dict:
    """
    Process all scenes through the audio pipeline.
    
    Args:
        scenes_dir: Directory containing scene clips
        output_dir: Directory for JSON outputs
        embeddings_dir: Directory for embedding files
        output_mode: 'pro', 'hobby', or 'both'
        skip_clap: Skip CLAP model
        skip_emotion: Skip emotion2vec
        skip_vad: Skip VAD
        max_clips: Limit number of clips (for testing)
        device: PyTorch device
        
    Returns:
        Processing summary
    """
    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Output files
    pro_output_file = output_dir / "audio_analysis_pro.json"
    hobby_output_file = output_dir / "audio_analysis_hobby.json"
    
    # Initialize model manager
    print("Initializing models...")
    model_manager = ModelManager(device=device)
    
    # Load required models
    model_manager.load_wav2vec2()
    
    if not skip_vad:
        model_manager.load_vad()
    
    if not skip_clap:
        model_manager.load_clap()
    
    if not skip_emotion:
        model_manager.load_emotion2vec()
    
    print()
    
    # Get scenes to process
    print("Discovering scenes...")
    scenes = get_scenes_for_processing(scenes_dir)
    
    if max_clips:
        scenes = scenes[:max_clips]
        print(f"Limited to first {max_clips} clips")
    
    print(f"Found {len(scenes)} scenes to process")
    print()
    
    if not scenes:
        return {'total_scenes': 0}
    
    # Initialize writers
    pro_writer = None
    hobby_writer = None
    
    if output_mode in ['pro', 'both']:
        pro_writer = IncrementalJSONWriter(pro_output_file, backup_interval=5)
        pro_writer.set_config({
            'output_mode': 'professional',
            'non_verbal_cues': NON_VERBAL_CUES,
            'models': {
                'wav2vec2': model_manager.wav2vec2_available,
                'vad': model_manager.vad_available
            }
        })
    
    if output_mode in ['hobby', 'both']:
        hobby_writer = IncrementalJSONWriter(hobby_output_file, backup_interval=5)
        hobby_writer.set_config({
            'output_mode': 'hobby',
            'models': {
                'wav2vec2': model_manager.wav2vec2_available,
                'vad': model_manager.vad_available,
                'clap': model_manager.clap_available,
                'emotion2vec': model_manager.emotion2vec_available
            },
            'embeddings_dir': str(embeddings_dir)
        })
    
    # Process scenes
    successful = 0
    failed = 0
    no_audio = 0
    
    for scene_path in tqdm(scenes, desc="Processing audio"):
        try:
            pro_out, hobby_out = process_scene(
                scene_path,
                model_manager,
                embeddings_dir,
                output_mode
            )
            
            # Track results
            if pro_out:
                if pro_out.get('processing_status') == 'success':
                    successful += 1
                elif pro_out.get('processing_status') == 'no_audio':
                    no_audio += 1
                else:
                    failed += 1
                pro_writer.add_analysis(pro_out)
            elif hobby_out:
                if hobby_out.get('processing_status') in ['success', 'partial']:
                    successful += 1
                elif hobby_out.get('processing_status') == 'no_audio':
                    no_audio += 1
                else:
                    failed += 1
            
            if hobby_out and hobby_writer:
                hobby_writer.add_analysis(hobby_out)
                
        except Exception as e:
            print(f"\n  Error processing {scene_path.name}: {e}")
            failed += 1
            
            if pro_writer:
                pro_writer.add_error(str(scene_path), 'exception', str(e))
            if hobby_writer:
                hobby_writer.add_error(str(scene_path), 'exception', str(e))
    
    # Finalize
    summary = {
        'total_scenes': len(scenes),
        'successful': successful,
        'failed': failed,
        'no_audio': no_audio,
        'output_mode': output_mode
    }
    
    if pro_writer:
        pro_writer.update_summary(summary)
        pro_writer.finalize()
        print(f"\nPro output: {pro_output_file}")
    
    if hobby_writer:
        hobby_writer.update_summary(summary)
        hobby_writer.finalize()
        print(f"Hobby output: {hobby_output_file}")
        print(f"Embeddings: {embeddings_dir}")
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Audio Analyzer - Professional and Hobby tracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all scenes, both outputs
  python audio_analyzer.py
  
  # Professional output only
  python audio_analyzer.py --output pro
  
  # Hobby output only, skip CLAP
  python audio_analyzer.py --output hobby --skip-clap
  
  # Test on 10 clips
  python audio_analyzer.py --test 10
        """
    )
    
    parser.add_argument(
        '--scenes-dir', '-s',
        type=Path,
        default=DEFAULT_SCENES_DIR,
        help=f"Directory with scene clips (default: {DEFAULT_SCENES_DIR})"
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSON files (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        '--embeddings-dir', '-e',
        type=Path,
        default=DEFAULT_EMBEDDINGS_DIR,
        help=f"Directory for embedding files (default: {DEFAULT_EMBEDDINGS_DIR})"
    )
    
    parser.add_argument(
        '--output',
        choices=['pro', 'hobby', 'both'],
        default='both',
        help="Output mode: pro, hobby, or both (default: both)"
    )
    
    parser.add_argument(
        '--test',
        type=int,
        metavar='N',
        help="Process only first N clips (for testing)"
    )
    
    parser.add_argument(
        '--skip-clap',
        action='store_true',
        help="Skip CLAP model (faster, less memory)"
    )
    
    parser.add_argument(
        '--skip-emotion',
        action='store_true',
        help="Skip emotion2vec model"
    )
    
    parser.add_argument(
        '--skip-vad',
        action='store_true',
        help="Skip pyannote VAD"
    )
    
    parser.add_argument(
        '--device',
        default='cuda',
        help="PyTorch device (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Print config
    print("=" * 70)
    print("Unified Audio Analyzer")
    print("=" * 70)
    print(f"Scenes Dir:     {args.scenes_dir}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"Embeddings Dir: {args.embeddings_dir}")
    print(f"Output Mode:    {args.output}")
    print(f"Device:         {args.device}")
    if args.test:
        print(f"Test Mode:      First {args.test} clips")
    print(f"Skip CLAP:      {args.skip_clap}")
    print(f"Skip emotion:   {args.skip_emotion}")
    print(f"Skip VAD:       {args.skip_vad}")
    print("=" * 70)
    print()
    
    # Process
    summary = process_all_scenes(
        scenes_dir=args.scenes_dir,
        output_dir=args.output_dir,
        embeddings_dir=args.embeddings_dir,
        output_mode=args.output,
        skip_clap=args.skip_clap,
        skip_emotion=args.skip_emotion,
        skip_vad=args.skip_vad,
        max_clips=args.test,
        device=args.device
    )
    
    # Report
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Scenes:  {summary.get('total_scenes', 0)}")
    print(f"Successful:    {summary.get('successful', 0)}")
    print(f"No Audio:      {summary.get('no_audio', 0)}")
    print(f"Failed:        {summary.get('failed', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
