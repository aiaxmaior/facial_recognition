#!/usr/bin/env python3
"""
Pipeline Data Portal - FastAPI Backend

Serves pipeline analysis data from JSON files to the React frontend.
Aggregates detection, emotion, and caption data into a unified API.

Author: ajax
Date: 2026-01-23
"""

import json
import re
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Import age correction module
from age_correction import AgeCorrector, correct_age, get_age_category, CorrectionResult

app = FastAPI(title="Pipeline Data Portal API", version="1.0.0")

# Global age corrector instance (can be configured via API)
age_corrector = AgeCorrector()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path configuration
BASE_DIR = Path(__file__).parent.parent.parent  # training_data directory
ANALYSIS_DIR = BASE_DIR / "analysis"
PROCESSED_DIR = BASE_DIR / "processed" / "data"
PROCESSED_VIDEOS_DIR = BASE_DIR / "processed"  # Source videos
SCENES_DIR = BASE_DIR / "scenes"  # Scene clips
VLM_COPIES_DIR = BASE_DIR / "vlm_copies"  # VLM video copies


def load_json_file(filepath: Path) -> Optional[Dict]:
    """Load JSON file with error handling."""
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {filepath}: {e}")
        return None


def extract_video_name(scene_path: str) -> str:
    """Extract video name from scene path (e.g., '2-Scene-001' -> '2')."""
    # Path like: /path/to/scenes/VideoName-Scene-001.mp4
    filename = Path(scene_path).stem
    # Try to extract video name (everything before -Scene-)
    match = re.match(r'^(.+?)-Scene-\d+', filename)
    if match:
        return match.group(1)
    return filename


def extract_scene_name(scene_path: str) -> str:
    """Extract scene name from path."""
    return Path(scene_path).stem


def build_clip_data(
    detection: Dict,
    emotions: Dict[str, Dict],
    captions: Dict[str, Dict]
) -> Dict:
    """Build unified clip data from multiple sources."""
    scene_path = detection.get('scene_path', '')
    scene_name = extract_scene_name(scene_path)
    video_name = extract_video_name(scene_path)
    
    # Get emotion data for this scene
    emotion = emotions.get(scene_path, {})
    
    # Get caption data for this scene
    caption = captions.get(scene_name, {})
    context = caption.get('context', {})
    
    return {
        'id': scene_name,
        'sceneName': scene_name,
        'videoName': video_name,
        'scenePath': scene_path,
        'vlmPath': caption.get('vlm_path'),
        
        # Detection data
        'durationSeconds': detection.get('duration_seconds', 0),
        'totalFrames': detection.get('total_frames', 0),
        'sampledFrames': detection.get('sampled_frames', 0),
        'personPresent': detection.get('person_present', False),
        'detectionCoverage': detection.get('detection_coverage', 0),
        'avgConfidence': detection.get('avg_confidence', 0),
        'maxPersons': detection.get('max_persons_detected', 0),
        
        # Emotion data
        'dominantEmotion': emotion.get('dominant_emotion', context.get('dominant_emotion', 'unknown')),
        'emotionDistribution': emotion.get('emotion_distribution', {}),
        'meanValence': emotion.get('mean_valence', context.get('valence', 0)),
        'meanArousal': emotion.get('mean_arousal', context.get('arousal', 0)),
        'painPleasureScore': emotion.get('pain_pleasure_score', 0),
        
        # Caption data
        'caption': caption.get('caption'),
        'captionError': caption.get('error'),
        
        # Context data from captions
        'ageEstimates': context.get('age_estimates', []),
        'genderEstimates': context.get('gender_estimates', []),
        'motionIntensity': context.get('motion_intensity', 0),
        'nudenetLabels': context.get('nudenet_labels', {}),
        'nudenetDetectionRate': context.get('nudenet_detection_rate', 0),
        
        # HAR (Human Action Recognition) data
        'harAction': detection.get('har_action'),
        'harScore': detection.get('har_score'),
        'harModelsUsed': detection.get('har_models_used'),
        'dominantActions': detection.get('dominant_actions', []),
    }


def build_video_data(clips: List[Dict]) -> List[Dict]:
    """Aggregate clip data into video summaries."""
    videos: Dict[str, Dict] = {}
    
    for clip in clips:
        video_name = clip['videoName']
        
        if video_name not in videos:
            videos[video_name] = {
                'name': video_name,
                'totalScenes': 0,
                'scenesWithPerson': 0,
                'totalDetectionCoverage': 0,
                'totalConfidence': 0,
                'emotionCounts': {},
                'totalValence': 0,
                'totalArousal': 0,
                'totalDuration': 0,
                'captionsGenerated': 0,
                'captionsFailed': 0,
                'clipCount': 0,
            }
        
        v = videos[video_name]
        v['totalScenes'] += 1
        v['clipCount'] += 1
        
        if clip['personPresent']:
            v['scenesWithPerson'] += 1
            v['totalDetectionCoverage'] += clip['detectionCoverage']
            v['totalConfidence'] += clip['avgConfidence']
            v['totalValence'] += clip['meanValence']
            v['totalArousal'] += clip['meanArousal']
        
        emotion = clip['dominantEmotion']
        v['emotionCounts'][emotion] = v['emotionCounts'].get(emotion, 0) + 1
        
        v['totalDuration'] += clip['durationSeconds']
        
        if clip['caption']:
            v['captionsGenerated'] += 1
        elif clip['captionError']:
            v['captionsFailed'] += 1
    
    # Calculate averages
    result = []
    for video_name, v in videos.items():
        with_person = v['scenesWithPerson'] or 1  # Avoid division by zero
        result.append({
            'name': v['name'],
            'totalScenes': v['totalScenes'],
            'scenesWithPerson': v['scenesWithPerson'],
            'avgDetectionCoverage': v['totalDetectionCoverage'] / with_person,
            'avgConfidence': v['totalConfidence'] / with_person,
            'emotionBreakdown': v['emotionCounts'],
            'avgValence': v['totalValence'] / with_person,
            'avgArousal': v['totalArousal'] / with_person,
            'totalDuration': v['totalDuration'],
            'captionsGenerated': v['captionsGenerated'],
            'captionsFailed': v['captionsFailed'],
        })
    
    return sorted(result, key=lambda x: x['name'])


def build_insights(clips: List[Dict], processing_summary: Optional[Dict]) -> Dict:
    """Generate pipeline insights from clip data."""
    total_clips = len(clips)
    clips_with_person = sum(1 for c in clips if c['personPresent'])
    clips_with_caption = sum(1 for c in clips if c['caption'])
    clips_failed = sum(1 for c in clips if c['captionError'])
    
    # Emotion distribution
    emotions: Dict[str, int] = {}
    for clip in clips:
        e = clip['dominantEmotion']
        emotions[e] = emotions.get(e, 0) + 1
    
    # Averages (only from clips with person)
    person_clips = [c for c in clips if c['personPresent']]
    avg_valence = sum(c['meanValence'] for c in person_clips) / len(person_clips) if person_clips else 0
    avg_arousal = sum(c['meanArousal'] for c in person_clips) / len(person_clips) if person_clips else 0
    avg_coverage = sum(c['detectionCoverage'] for c in person_clips) / len(person_clips) if person_clips else 0
    
    # Video count
    videos = set(c['videoName'] for c in clips)
    
    return {
        'totalClips': total_clips,
        'clipsWithPerson': clips_with_person,
        'clipsWithCaption': clips_with_caption,
        'clipsFailed': clips_failed,
        'emotionDistribution': emotions,
        'averageValence': avg_valence,
        'averageArousal': avg_arousal,
        'averageDetectionCoverage': avg_coverage,
        'totalVideos': len(videos),
        'processingTime': processing_summary.get('total_processing_time', 0) if processing_summary else 0,
        'processedAt': processing_summary.get('processed_at', '') if processing_summary else '',
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/pipeline-data")
async def get_pipeline_data():
    """
    Get all pipeline data aggregated for the frontend.
    Returns clips, videos, and insights.
    Merges HAR (Human Action Recognition) data when available.
    """
    # Load all data sources
    detections_data = load_json_file(ANALYSIS_DIR / "detections.json")
    emotions_data = load_json_file(ANALYSIS_DIR / "emotions.json")
    captions_data = load_json_file(ANALYSIS_DIR / "captions.json")
    processing_summary = load_json_file(PROCESSED_DIR / "processing_summary.json")
    
    # Load HAR data for merging (optional)
    har_detections = load_json_file(ANALYSIS_DIR / "har_detections.json")
    har_actions = load_json_file(ANALYSIS_DIR / "har_actions.json")
    
    # Build HAR lookup by scene_path for merging
    har_lookup: Dict[str, Dict] = {}
    if har_detections:
        for analysis in har_detections.get('analyses', []):
            scene_path = analysis.get('scene_path', '')
            har_lookup[scene_path] = {
                'har_action': analysis.get('har_action'),
                'har_score': analysis.get('har_score'),
                'har_models_used': analysis.get('har_models_used'),
                'dominant_actions': analysis.get('dominant_actions', []),
            }
    # Also check har_actions.json clips array
    if har_actions and not har_lookup:
        for clip in har_actions.get('clips', []):
            scene_path = clip.get('clip_path', '')
            top_k = clip.get('predictions', {}).get('top_k', [])
            har_lookup[scene_path] = {
                'har_action': top_k[0]['label'] if top_k else None,
                'har_score': top_k[0]['score'] if top_k else None,
                'har_models_used': clip.get('models_used', []),
                'dominant_actions': [
                    {'label': p['label'], 'score': p['score'], 'rank': i+1}
                    for i, p in enumerate(top_k[:3])
                ],
            }
    
    # Build lookup dicts
    emotions_lookup: Dict[str, Dict] = {}
    if emotions_data:
        for analysis in emotions_data.get('analyses', []):
            scene_path = analysis.get('scene_path', '')
            emotions_lookup[scene_path] = analysis
    
    captions_lookup: Dict[str, Dict] = {}
    if captions_data:
        for result in captions_data.get('results', []):
            scene_name = result.get('scene_name', '')
            captions_lookup[scene_name] = result
    
    # Build clip data
    clips = []
    if detections_data:
        for detection in detections_data.get('analyses', []):
            # Merge HAR data into detection if available
            scene_path = detection.get('scene_path', '')
            if scene_path in har_lookup:
                detection = {**detection, **har_lookup[scene_path]}
            
            clip = build_clip_data(detection, emotions_lookup, captions_lookup)
            clips.append(clip)
    
    # Build video summaries
    videos = build_video_data(clips)
    
    # Build insights
    insights = build_insights(clips, processing_summary)
    
    return {
        'clips': clips,
        'videos': videos,
        'insights': insights,
    }


@app.get("/api/clips")
async def get_clips():
    """Get all clips data."""
    data = await get_pipeline_data()
    return data['clips']


@app.get("/api/videos")
async def get_videos():
    """Get all video summaries."""
    data = await get_pipeline_data()
    return data['videos']


@app.get("/api/insights")
async def get_insights():
    """Get pipeline insights."""
    data = await get_pipeline_data()
    return data['insights']


@app.get("/api/clip/{scene_name}")
async def get_clip(scene_name: str):
    """Get a specific clip by scene name."""
    data = await get_pipeline_data()
    for clip in data['clips']:
        if clip['sceneName'] == scene_name:
            return clip
    raise HTTPException(status_code=404, detail="Clip not found")


@app.get("/api/video/{video_name}")
async def get_video(video_name: str):
    """Get a specific video summary."""
    data = await get_pipeline_data()
    for video in data['videos']:
        if video['name'] == video_name:
            return video
    raise HTTPException(status_code=404, detail="Video not found")


@app.get("/api/raw/detections")
async def get_raw_detections():
    """Get raw detections JSON."""
    data = load_json_file(ANALYSIS_DIR / "detections.json")
    if data is None:
        raise HTTPException(status_code=404, detail="Detections data not found")
    return data


@app.get("/api/raw/emotions")
async def get_raw_emotions():
    """Get raw emotions JSON."""
    data = load_json_file(ANALYSIS_DIR / "emotions.json")
    if data is None:
        raise HTTPException(status_code=404, detail="Emotions data not found")
    return data


@app.get("/api/raw/captions")
async def get_raw_captions():
    """Get raw captions JSON."""
    data = load_json_file(ANALYSIS_DIR / "captions.json")
    if data is None:
        raise HTTPException(status_code=404, detail="Captions data not found")
    return data


@app.get("/api/raw/demographics")
async def get_raw_demographics():
    """Get raw demographics JSON."""
    data = load_json_file(ANALYSIS_DIR / "demographics.json")
    if data is None:
        raise HTTPException(status_code=404, detail="Demographics data not found")
    return data


@app.get("/api/video-demographics")
async def get_video_demographics():
    """
    Get aggregated video demographics with per-person age estimates.
    
    This data comes from the video benchmark analysis and includes:
    - Per-scene confirmed ages (aggregated from clip-level estimates)
    - Age corrections applied for female subjects
    - Source video grouping
    """
    data = load_json_file(ANALYSIS_DIR / "video_demographics.json")
    if data is None:
        raise HTTPException(status_code=404, detail="Video demographics data not found. Run aggregate_video_ages.py first.")
    
    analyses = data.get('analyses', [])
    
    # Apply age corrections and build enhanced output
    enhanced_analyses = []
    for analysis in analyses:
        confirmed_age = analysis.get('confirmed_age')
        gender = analysis.get('confirmed_gender', 'unknown')
        
        # Apply correction
        corrected_age = confirmed_age
        skew = 0.0
        if confirmed_age is not None:
            result = age_corrector.correct(confirmed_age, gender)
            corrected_age = result.corrected_age
            skew = result.skew
        
        enhanced = {
            **analysis,
            'confirmed_age_raw': confirmed_age,
            'confirmed_age_corrected': corrected_age,
            'age_correction_skew': skew,
            'confirmed_category_corrected': get_age_category(corrected_age) if corrected_age else None,
        }
        enhanced_analyses.append(enhanced)
    
    return {
        'config': data.get('config', {}),
        'summary': {
            **data.get('summary', {}),
            'age_correction_applied': True,
            'age_correction_coefficients': age_corrector.coefficients,
        },
        'analyses': enhanced_analyses,
    }


@app.get("/api/raw/video-demographics")
async def get_raw_video_demographics():
    """Get raw video demographics JSON (without corrections)."""
    data = load_json_file(ANALYSIS_DIR / "video_demographics.json")
    if data is None:
        raise HTTPException(status_code=404, detail="Video demographics data not found")
    return data


@app.get("/api/raw/nudenet")
async def get_raw_nudenet():
    """Get raw nudenet JSON."""
    data = load_json_file(ANALYSIS_DIR / "nudenet.json")
    if data is None:
        raise HTTPException(status_code=404, detail="NudeNet data not found")
    return data


@app.get("/api/nudenet")
async def get_nudenet_data():
    """
    Get aggregated NudeNet data for visualization.
    Returns label distributions, detection rates, and per-scene stats.
    """
    nudenet_data = load_json_file(ANALYSIS_DIR / "nudenet.json")
    if nudenet_data is None:
        raise HTTPException(status_code=404, detail="NudeNet data not found")
    
    analyses = nudenet_data.get('analyses', {})
    
    # Aggregate label counts across all scenes
    total_label_counts: Dict[str, int] = {}
    exposed_counts: Dict[str, int] = {}
    covered_counts: Dict[str, int] = {}
    
    # Per-scene stats for scatter plots
    scene_stats = []
    detection_rates = []
    
    for scene_name, scene_data in analyses.items():
        label_counts = scene_data.get('label_counts', {})
        detection_rate = scene_data.get('detection_rate', 0.0)
        frames_sampled = scene_data.get('total_frames_sampled', 0)
        frames_with_detections = scene_data.get('frames_with_detections', 0)
        
        # Aggregate labels
        for label, count in label_counts.items():
            total_label_counts[label] = total_label_counts.get(label, 0) + count
            if 'EXPOSED' in label:
                exposed_counts[label] = exposed_counts.get(label, 0) + count
            elif 'COVERED' in label:
                covered_counts[label] = covered_counts.get(label, 0) + count
        
        detection_rates.append(detection_rate)
        
        scene_stats.append({
            'sceneName': scene_name,
            'scenePath': scene_data.get('scene_path', ''),
            'framesSampled': frames_sampled,
            'framesWithDetections': frames_with_detections,
            'detectionRate': detection_rate,
            'labelCounts': label_counts,
            'totalDetections': sum(label_counts.values()),
            'hasExposed': any('EXPOSED' in l for l in label_counts.keys()),
        })
    
    # Create detection rate histogram
    rate_histogram = []
    bin_size = 0.1
    for start in range(0, 10):
        s = start / 10
        e = (start + 1) / 10
        count = sum(1 for r in detection_rates if s <= r < e)
        rate_histogram.append({
            'range': f'{int(s*100)}-{int(e*100)}%',
            'min': s,
            'max': e,
            'count': count
        })
    
    # Summary stats
    total_scenes = len(analyses)
    scenes_with_detections = sum(1 for s in scene_stats if s['framesWithDetections'] > 0)
    scenes_with_exposed = sum(1 for s in scene_stats if s['hasExposed'])
    
    return {
        'summary': {
            'totalScenes': total_scenes,
            'scenesWithDetections': scenes_with_detections,
            'scenesWithExposed': scenes_with_exposed,
            'avgDetectionRate': sum(detection_rates) / len(detection_rates) if detection_rates else 0,
            'totalDetections': sum(total_label_counts.values()),
        },
        'labelDistribution': total_label_counts,
        'exposedDistribution': exposed_counts,
        'coveredDistribution': covered_counts,
        'detectionRateHistogram': rate_histogram,
        'sceneStats': scene_stats,
    }


@app.get("/api/explorer-data")
async def get_explorer_data():
    """
    Get unified data for the Data Explorer with all features merged per scene.
    Enables X vs Y comparisons across demographics, emotions, nudenet, etc.
    """
    # Load all data sources
    demographics_data = load_json_file(ANALYSIS_DIR / "demographics.json")
    emotions_data = load_json_file(ANALYSIS_DIR / "emotions.json")
    nudenet_data = load_json_file(ANALYSIS_DIR / "nudenet.json")
    detections_data = load_json_file(ANALYSIS_DIR / "detections.json")
    
    # Build unified scene data
    unified_scenes: Dict[str, Dict] = {}
    
    # Process demographics
    if demographics_data:
        for analysis in demographics_data.get('analyses', []):
            scene_path = analysis.get('scene_path', '')
            scene_name = Path(scene_path).stem
            
            if scene_name not in unified_scenes:
                unified_scenes[scene_name] = {'sceneName': scene_name, 'scenePath': scene_path}
            
            # Get raw age stats
            raw_mean_age = analysis.get('mean_age')
            raw_min_age = analysis.get('age_range', [None])[0] if analysis.get('age_range') else None
            raw_max_age = analysis.get('age_range', [None, None])[1] if analysis.get('age_range') and len(analysis.get('age_range', [])) > 1 else None
            
            # Compute corrected ages from frame demographics
            frame_demos = analysis.get('frame_demographics', [])
            corrected_ages = []
            for frame in frame_demos:
                for person in frame.get('persons', []):
                    raw_age = person.get('age_combined')
                    gender = person.get('gender_combined', 'unknown')
                    if raw_age is not None:
                        result = age_corrector.correct(raw_age, gender)
                        corrected_ages.append(result.corrected_age)
            
            unified_scenes[scene_name].update({
                # Raw ages
                'meanAge': raw_mean_age,
                'minAge': raw_min_age,
                'maxAge': raw_max_age,
                # Corrected ages
                'meanAgeCorrected': sum(corrected_ages) / len(corrected_ages) if corrected_ages else None,
                'minAgeCorrected': min(corrected_ages) if corrected_ages else None,
                'maxAgeCorrected': max(corrected_ages) if corrected_ages else None,
                # Other stats
                'meanMotion': analysis.get('mean_motion', 0),
                'erraticMotionRatio': analysis.get('erratic_motion_ratio', 0),
                'framesWithPersons': analysis.get('frames_with_persons', 0),
                'ageDistribution': analysis.get('age_distribution', {}),
                'genderDistribution': analysis.get('gender_distribution', {}),
                'raceDistribution': analysis.get('race_distribution', {}),
                # Derived features
                'maleCount': analysis.get('gender_distribution', {}).get('Male', 0),
                'femaleCount': analysis.get('gender_distribution', {}).get('Female', 0),
                'adultCount': analysis.get('age_distribution', {}).get('adult', 0) + analysis.get('age_distribution', {}).get('young_adult', 0),
            })
    
    # Process emotions
    if emotions_data:
        for analysis in emotions_data.get('analyses', []):
            scene_path = analysis.get('scene_path', '')
            scene_name = Path(scene_path).stem
            
            if scene_name not in unified_scenes:
                unified_scenes[scene_name] = {'sceneName': scene_name, 'scenePath': scene_path}
            
            unified_scenes[scene_name].update({
                'dominantEmotion': analysis.get('dominant_emotion', 'unknown'),
                'meanValence': analysis.get('mean_valence', 0),
                'meanArousal': analysis.get('mean_arousal', 0),
                'painPleasureScore': analysis.get('pain_pleasure_score', 0),
                'emotionDistribution': analysis.get('emotion_distribution', {}),
            })
    
    # Process nudenet
    if nudenet_data:
        for scene_name, scene_data in nudenet_data.get('analyses', {}).items():
            if scene_name not in unified_scenes:
                unified_scenes[scene_name] = {'sceneName': scene_name, 'scenePath': scene_data.get('scene_path', '')}
            
            label_counts = scene_data.get('label_counts', {})
            exposed_count = sum(v for k, v in label_counts.items() if 'EXPOSED' in k)
            covered_count = sum(v for k, v in label_counts.items() if 'COVERED' in k)
            
            unified_scenes[scene_name].update({
                'nudenetDetectionRate': scene_data.get('detection_rate', 0),
                'nudenetFramesSampled': scene_data.get('total_frames_sampled', 0),
                'nudenetFramesWithDetections': scene_data.get('frames_with_detections', 0),
                'nudenetTotalDetections': sum(label_counts.values()),
                'nudenetExposedCount': exposed_count,
                'nudenetCoveredCount': covered_count,
                'nudenetLabelCounts': label_counts,
            })
    
    # Process detections (person detection)
    if detections_data:
        for analysis in detections_data.get('analyses', []):
            scene_path = analysis.get('scene_path', '')
            scene_name = Path(scene_path).stem
            
            if scene_name not in unified_scenes:
                unified_scenes[scene_name] = {'sceneName': scene_name, 'scenePath': scene_path}
            
            unified_scenes[scene_name].update({
                'personPresent': analysis.get('person_present', False),
                'detectionCoverage': analysis.get('detection_coverage', 0),
                'avgConfidence': analysis.get('avg_confidence', 0),
                'maxPersons': analysis.get('max_persons_detected', 0),
                'durationSeconds': analysis.get('duration_seconds', 0),
            })
    
    # Convert to list and filter out scenes with no useful data
    scene_list = [s for s in unified_scenes.values() if s.get('meanAge') is not None or s.get('nudenetDetectionRate') is not None or s.get('meanValence') is not None]
    
    # Define available features for the explorer
    numeric_features = [
        {'key': 'meanAge', 'label': 'Mean Age (Raw)', 'category': 'demographics'},
        {'key': 'minAge', 'label': 'Min Age (Raw)', 'category': 'demographics'},
        {'key': 'maxAge', 'label': 'Max Age (Raw)', 'category': 'demographics'},
        {'key': 'meanAgeCorrected', 'label': 'Mean Age (Corrected)', 'category': 'demographics'},
        {'key': 'minAgeCorrected', 'label': 'Min Age (Corrected)', 'category': 'demographics'},
        {'key': 'maxAgeCorrected', 'label': 'Max Age (Corrected)', 'category': 'demographics'},
        {'key': 'meanMotion', 'label': 'Mean Motion', 'category': 'demographics'},
        {'key': 'erraticMotionRatio', 'label': 'Erratic Motion Ratio', 'category': 'demographics'},
        {'key': 'framesWithPersons', 'label': 'Frames with Persons', 'category': 'demographics'},
        {'key': 'maleCount', 'label': 'Male Count', 'category': 'demographics'},
        {'key': 'femaleCount', 'label': 'Female Count', 'category': 'demographics'},
        {'key': 'meanValence', 'label': 'Valence', 'category': 'emotions'},
        {'key': 'meanArousal', 'label': 'Arousal', 'category': 'emotions'},
        {'key': 'painPleasureScore', 'label': 'Pain/Pleasure Score', 'category': 'emotions'},
        {'key': 'nudenetDetectionRate', 'label': 'NudeNet Detection Rate', 'category': 'nudenet'},
        {'key': 'nudenetTotalDetections', 'label': 'NudeNet Total Detections', 'category': 'nudenet'},
        {'key': 'nudenetExposedCount', 'label': 'Exposed Content Count', 'category': 'nudenet'},
        {'key': 'nudenetCoveredCount', 'label': 'Covered Content Count', 'category': 'nudenet'},
        {'key': 'detectionCoverage', 'label': 'Person Detection Coverage', 'category': 'detection'},
        {'key': 'avgConfidence', 'label': 'Detection Confidence', 'category': 'detection'},
        {'key': 'maxPersons', 'label': 'Max Persons Detected', 'category': 'detection'},
        {'key': 'durationSeconds', 'label': 'Duration (seconds)', 'category': 'detection'},
    ]
    
    categorical_features = [
        {'key': 'dominantEmotion', 'label': 'Dominant Emotion', 'category': 'emotions'},
        {'key': 'personPresent', 'label': 'Person Present', 'category': 'detection'},
    ]
    
    return {
        'scenes': scene_list,
        'numericFeatures': numeric_features,
        'categoricalFeatures': categorical_features,
        'totalScenes': len(scene_list),
    }


@app.get("/api/har-actions")
async def get_har_actions():
    """
    Get HAR (Human Action Recognition) data for visualization.
    Returns action distributions, per-clip predictions, and video summaries.
    """
    har_data = load_json_file(ANALYSIS_DIR / "har_actions.json")
    if har_data is None:
        raise HTTPException(status_code=404, detail="HAR actions data not found")
    
    return har_data


@app.get("/api/har-pipeline-data")
async def get_har_pipeline_data():
    """
    Get all HAR pipeline data aggregated for the frontend.
    Similar to /api/pipeline-data but for HAR-specific JSONs.
    """
    # Load HAR-specific data sources
    detections_data = load_json_file(ANALYSIS_DIR / "har_detections.json")
    emotions_data = load_json_file(ANALYSIS_DIR / "har_emotions.json")
    captions_data = load_json_file(ANALYSIS_DIR / "har_captions.json")
    processing_summary = load_json_file(ANALYSIS_DIR / "har_processing_summary.json")
    
    # Build lookup dicts
    emotions_lookup: Dict[str, Dict] = {}
    if emotions_data:
        for analysis in emotions_data.get('analyses', []):
            scene_path = analysis.get('scene_path', '')
            emotions_lookup[scene_path] = analysis
    
    captions_lookup: Dict[str, Dict] = {}
    if captions_data:
        for result in captions_data.get('results', []):
            scene_name = result.get('scene_name', '')
            captions_lookup[scene_name] = result
    
    # Build clip data
    clips = []
    if detections_data:
        for detection in detections_data.get('analyses', []):
            clip = build_clip_data(detection, emotions_lookup, captions_lookup)
            # Add HAR-specific fields
            clip['harAction'] = detection.get('har_action', 'unknown')
            clip['harScore'] = detection.get('har_score', 0.0)
            clip['harModelsUsed'] = detection.get('har_models_used', [])
            clips.append(clip)
    
    # Build video summaries
    videos = build_video_data(clips)
    
    # Build insights
    insights = build_insights(clips, processing_summary)
    
    return {
        'clips': clips,
        'videos': videos,
        'insights': insights,
        'dataSource': 'HAR'
    }


@app.get("/api/demographics")
async def get_demographics_data():
    """
    Get aggregated demographics data for visualization.
    Returns age (raw + corrected), gender, race distributions and per-scene stats.
    
    Age correction is applied to female subjects based on regression model.
    """
    demographics_data = load_json_file(ANALYSIS_DIR / "demographics.json")
    if demographics_data is None:
        raise HTTPException(status_code=404, detail="Demographics data not found")
    
    analyses = demographics_data.get('analyses', [])
    config = demographics_data.get('config', {})
    age_categories = config.get('age_categories', {})
    
    # Aggregate statistics
    total_scenes = len(analyses)
    scenes_with_persons = sum(1 for a in analyses if a.get('frames_with_persons', 0) > 0)
    
    # Age distribution - raw (aggregate across all scenes)
    raw_age_distribution = {
        "infant": 0, "toddler": 0, "child": 0, "adolescent": 0, "early_teen": 0,
        "late_teen": 0, "young_adult": 0, "adult": 0, "middle_aged": 0, "senior": 0, "unknown": 0
    }
    
    # Age distribution - corrected
    corrected_age_distribution = {
        "infant": 0, "toddler": 0, "child": 0, "adolescent": 0, "early_teen": 0,
        "late_teen": 0, "young_adult": 0, "adult": 0, "middle_aged": 0, "senior": 0, "unknown": 0
    }
    
    # Gender distribution
    total_gender_distribution = {"Male": 0, "Female": 0, "unknown": 0}
    
    # Race distribution
    total_race_distribution: Dict[str, int] = {}
    
    # Per-scene data for scatter plots
    scene_stats = []
    all_raw_ages = []
    all_corrected_ages = []
    
    for analysis in analyses:
        scene_path = analysis.get('scene_path', '')
        scene_name = Path(scene_path).stem
        
        # Collect individual ages from frame demographics and apply corrections
        frame_demos = analysis.get('frame_demographics', [])
        scene_raw_ages = []
        scene_corrected_ages = []
        
        for frame in frame_demos:
            for person in frame.get('persons', []):
                raw_age = person.get('age_combined')
                gender = person.get('gender_combined', 'unknown')
                
                if raw_age is not None:
                    scene_raw_ages.append(raw_age)
                    all_raw_ages.append(raw_age)
                    
                    # Apply age correction
                    result = age_corrector.correct(raw_age, gender)
                    corrected_age = result.corrected_age
                    scene_corrected_ages.append(corrected_age)
                    all_corrected_ages.append(corrected_age)
                    
                    # Update raw distribution
                    raw_cat = get_age_category(raw_age)
                    if raw_cat in raw_age_distribution:
                        raw_age_distribution[raw_cat] += 1
                    
                    # Update corrected distribution
                    corrected_cat = get_age_category(corrected_age)
                    if corrected_cat in corrected_age_distribution:
                        corrected_age_distribution[corrected_cat] += 1
                
                # Gender
                if gender:
                    g = gender.lower()
                    if g in ('male', 'man'):
                        total_gender_distribution['Male'] += 1
                    elif g in ('female', 'woman'):
                        total_gender_distribution['Female'] += 1
                    else:
                        total_gender_distribution['unknown'] += 1
                
                # Race
                race = person.get('race')
                if race:
                    total_race_distribution[race] = total_race_distribution.get(race, 0) + 1
        
        # Scene-level stats with both raw and corrected
        scene_stats.append({
            'sceneName': scene_name,
            'scenePath': scene_path,
            'framesAnalyzed': analysis.get('total_frames_analyzed', 0),
            'framesWithPersons': analysis.get('frames_with_persons', 0),
            # Raw ages
            'meanAge': sum(scene_raw_ages) / len(scene_raw_ages) if scene_raw_ages else None,
            'minAge': min(scene_raw_ages) if scene_raw_ages else None,
            'maxAge': max(scene_raw_ages) if scene_raw_ages else None,
            # Corrected ages
            'meanAgeCorrected': sum(scene_corrected_ages) / len(scene_corrected_ages) if scene_corrected_ages else None,
            'minAgeCorrected': min(scene_corrected_ages) if scene_corrected_ages else None,
            'maxAgeCorrected': max(scene_corrected_ages) if scene_corrected_ages else None,
            # Other stats
            'meanMotion': analysis.get('mean_motion', 0),
            'erraticMotionRatio': analysis.get('erratic_motion_ratio', 0),
            'genderDistribution': analysis.get('gender_distribution', {}),
            'raceDistribution': analysis.get('race_distribution', {}),
        })
    
    # Create age histogram bins for both raw and corrected
    def make_histogram(ages, bin_size=5, max_age=80):
        histogram = []
        for start in range(0, max_age, bin_size):
            end = start + bin_size
            count = sum(1 for age in ages if start <= age < end)
            histogram.append({
                'range': f'{start}-{end}',
                'min': start,
                'max': end,
                'count': count
            })
        return histogram
    
    # Build histograms
    raw_histogram = make_histogram(all_raw_ages)
    corrected_histogram = make_histogram(all_corrected_ages)
    
    return {
        'summary': {
            'totalScenes': total_scenes,
            'scenesWithPersons': scenes_with_persons,
            'totalPersonsDetected': sum(total_gender_distribution.values()),
            # Both raw and corrected mean ages
            'meanAgeRaw': sum(all_raw_ages) / len(all_raw_ages) if all_raw_ages else None,
            'meanAgeCorrected': sum(all_corrected_ages) / len(all_corrected_ages) if all_corrected_ages else None,
            # Backward-compatible: frontend expects meanAgeOverall
            'meanAgeOverall': sum(all_corrected_ages) / len(all_corrected_ages) if all_corrected_ages else None,
        },
        'ageCategories': age_categories,
        # Raw and corrected distributions
        'ageDistributionRaw': raw_age_distribution,
        'ageDistributionCorrected': corrected_age_distribution,
        # Backward-compatible: frontend expects ageDistribution (use corrected as default)
        'ageDistribution': corrected_age_distribution,
        'genderDistribution': total_gender_distribution,
        'raceDistribution': total_race_distribution,
        # Raw and corrected histograms
        'ageHistogramRaw': raw_histogram,
        'ageHistogramCorrected': corrected_histogram,
        # Backward-compatible: frontend expects ageHistogram (use corrected as default)
        'ageHistogram': corrected_histogram,
        'sceneStats': scene_stats,
        'ageCorrectionApplied': True,
        'ageCorrectionCoefficients': age_corrector.coefficients,
    }


# ============ Per-Person Detail Endpoints ============

@app.get("/api/scene/{scene_name}/persons")
async def get_scene_persons(scene_name: str):
    """
    Get detailed per-person data for a specific scene.
    
    Returns all individuals detected across frames with their:
    - Demographics (age, gender, race)
    - Emotions (per-face if available)
    - Bounding boxes
    - Frame appearances
    
    This supports multi-person tracking and analysis.
    """
    # Load demographics data
    demographics_data = load_json(ANALYSIS_DIR / "demographics.json") if (ANALYSIS_DIR / "demographics.json").exists() else {}
    emotions_data = load_json(ANALYSIS_DIR / "emotions.json") if (ANALYSIS_DIR / "emotions.json").exists() else {}
    
    # Find the scene in demographics
    scene_demographics = None
    for analysis in demographics_data.get('analyses', []):
        if scene_name in analysis.get('scene_path', ''):
            scene_demographics = analysis
            break
    
    # Find the scene in emotions
    scene_emotions = None
    for analysis in emotions_data.get('analyses', []):
        if scene_name in analysis.get('scene_path', ''):
            scene_emotions = analysis
            break
    
    if not scene_demographics:
        raise HTTPException(status_code=404, detail=f"Scene not found: {scene_name}")
    
    # Collect all persons across frames
    persons_timeline = []  # List of all person detections with frame info
    unique_person_stats = {}  # Aggregated stats per "person slot" (e.g., person_0, person_1)
    
    frame_demos = scene_demographics.get('frame_demographics', [])
    frame_emotions_list = scene_emotions.get('frame_emotions', []) if scene_emotions else []
    
    # Build emotion lookup by frame_idx
    emotion_by_frame = {}
    for fe in frame_emotions_list:
        frame_idx = fe.get('frame_idx')
        if frame_idx is not None:
            emotion_by_frame[frame_idx] = fe
    
    for frame in frame_demos:
        frame_idx = frame.get('frame_idx', 0)
        persons = frame.get('persons', [])
        
        # Get emotion data for this frame (if available)
        frame_emotion = emotion_by_frame.get(frame_idx, {})
        faces_emotions = frame_emotion.get('faces', [])  # NEW: per-face emotions
        
        for person_idx, person in enumerate(persons):
            # Try to match person to face by bbox overlap (if faces_emotions available)
            matched_face_emotion = None
            if faces_emotions and person.get('bbox'):
                person_bbox = person.get('bbox')
                for face in faces_emotions:
                    face_bbox = face.get('bbox')
                    if face_bbox and _bbox_overlap(person_bbox, face_bbox) > 0.3:
                        matched_face_emotion = face
                        break
            
            person_record = {
                'frame_idx': frame_idx,
                'person_idx': person_idx,
                'bbox': person.get('bbox'),
                'age': person.get('age_combined'),
                'age_source': person.get('age_source'),
                'gender': person.get('gender_combined'),
                'gender_confidence': person.get('gender_confidence'),
                'race': person.get('race'),
                'race_confidence': person.get('race_confidence'),
                'needs_review': person.get('needs_review', False),
                'age_disagreement': person.get('age_disagreement', False),
                'gender_disagreement': person.get('gender_disagreement', False),
                # Per-face emotion (if matched)
                'emotion': matched_face_emotion.get('dominant_emotion') if matched_face_emotion else None,
                'emotion_scores': matched_face_emotion.get('emotion_scores') if matched_face_emotion else None,
                'valence': matched_face_emotion.get('valence') if matched_face_emotion else None,
                'arousal': matched_face_emotion.get('arousal') if matched_face_emotion else None,
            }
            persons_timeline.append(person_record)
            
            # Aggregate by person slot
            slot_key = f'person_{person_idx}'
            if slot_key not in unique_person_stats:
                unique_person_stats[slot_key] = {
                    'person_idx': person_idx,
                    'appearances': 0,
                    'ages': [],
                    'genders': [],
                    'emotions': [],
                    'races': [],
                    'needs_review_count': 0,
                }
            
            stats = unique_person_stats[slot_key]
            stats['appearances'] += 1
            if person.get('age_combined'):
                stats['ages'].append(person['age_combined'])
            if person.get('gender_combined'):
                stats['genders'].append(person['gender_combined'])
            if person.get('race'):
                stats['races'].append(person['race'])
            if matched_face_emotion:
                stats['emotions'].append(matched_face_emotion.get('dominant_emotion'))
            if person.get('needs_review'):
                stats['needs_review_count'] += 1
    
    # Compute aggregated stats for each person slot
    person_summaries = []
    for slot_key, stats in unique_person_stats.items():
        ages = stats['ages']
        genders = stats['genders']
        emotions = stats['emotions']
        races = stats['races']
        
        summary = {
            'person_idx': stats['person_idx'],
            'appearances': stats['appearances'],
            'mean_age': sum(ages) / len(ages) if ages else None,
            'age_range': [min(ages), max(ages)] if ages else None,
            'dominant_gender': max(set(genders), key=genders.count) if genders else None,
            'gender_distribution': {g: genders.count(g) for g in set(genders)} if genders else {},
            'dominant_emotion': max(set(emotions), key=emotions.count) if emotions else None,
            'emotion_distribution': {e: emotions.count(e) for e in set(emotions)} if emotions else {},
            'dominant_race': max(set(races), key=races.count) if races else None,
            'needs_review_ratio': stats['needs_review_count'] / stats['appearances'] if stats['appearances'] > 0 else 0,
        }
        person_summaries.append(summary)
    
    return {
        'sceneName': scene_name,
        'totalFramesAnalyzed': len(frame_demos),
        'maxPersonsInFrame': max(len(f.get('persons', [])) for f in frame_demos) if frame_demos else 0,
        'uniquePersonSlots': len(person_summaries),
        'personSummaries': person_summaries,
        'timeline': persons_timeline,  # Full frame-by-frame data
    }


def _bbox_overlap(bbox1, bbox2) -> float:
    """Calculate IoU overlap between two bboxes (x1,y1,x2,y2)."""
    if not bbox1 or not bbox2:
        return 0.0
    
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


# ============ Video Serving Endpoints ============

def find_video_file(directory: Path, name: str) -> Optional[Path]:
    """Find a video file by name (with or without extension)."""
    # Try exact match first
    for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
        path = directory / f"{name}{ext}"
        if path.exists():
            return path
    # Try with the name as-is (might already have extension)
    path = directory / name
    if path.exists():
        return path
    # Search for partial match
    for f in directory.iterdir():
        if f.is_file() and f.stem == name:
            return f
    return None


@app.get("/api/video/scene/{scene_name}")
async def get_scene_video(scene_name: str):
    """Serve a scene video file."""
    video_path = find_video_file(SCENES_DIR, scene_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail=f"Scene video not found: {scene_name}")
    
    media_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    return FileResponse(
        video_path,
        media_type=media_type,
        filename=video_path.name
    )


@app.get("/api/video/source/{video_name}")
async def get_source_video(video_name: str):
    """Serve a source video file from processed directory."""
    video_path = find_video_file(PROCESSED_VIDEOS_DIR, video_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail=f"Source video not found: {video_name}")
    
    media_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    return FileResponse(
        video_path,
        media_type=media_type,
        filename=video_path.name
    )


@app.get("/api/video/vlm/{scene_name}")
async def get_vlm_video(scene_name: str):
    """Serve a VLM copy video file."""
    # VLM copies have _vlm suffix
    vlm_name = f"{scene_name}_vlm"
    video_path = find_video_file(VLM_COPIES_DIR, vlm_name)
    if video_path is None:
        # Try without suffix
        video_path = find_video_file(VLM_COPIES_DIR, scene_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail=f"VLM video not found: {scene_name}")
    
    media_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"
    return FileResponse(
        video_path,
        media_type=media_type,
        filename=video_path.name
    )


@app.get("/api/video/exists/{video_type}/{name}")
async def check_video_exists(video_type: str, name: str):
    """Check if a video file exists."""
    if video_type == "scene":
        video_path = find_video_file(SCENES_DIR, name)
    elif video_type == "source":
        video_path = find_video_file(PROCESSED_VIDEOS_DIR, name)
    elif video_type == "vlm":
        vlm_name = f"{name}_vlm"
        video_path = find_video_file(VLM_COPIES_DIR, vlm_name)
        if video_path is None:
            video_path = find_video_file(VLM_COPIES_DIR, name)
    else:
        raise HTTPException(status_code=400, detail="Invalid video type")
    
    return {
        "exists": video_path is not None,
        "path": str(video_path) if video_path else None
    }


# ============ Caption Reviewer Endpoints ============

# Image dataset directories to scan
IMAGE_DATASET_DIRS = [
    BASE_DIR / "asdataset" / "scraped_images",
    BASE_DIR / "datasets",  # Future datasets
]


class CaptionUpdate(BaseModel):
    """Request body for updating a caption."""
    caption: str
    negative_prompt: Optional[str] = None


def scan_image_datasets() -> List[Dict]:
    """Scan configured directories for image datasets."""
    datasets = []
    
    for dataset_dir in IMAGE_DATASET_DIRS:
        if not dataset_dir.exists():
            continue
            
        # Count images in directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        images = [f for f in dataset_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
        
        if images:
            # Check for metadata.json
            metadata_path = dataset_dir / "metadata.json"
            has_metadata = metadata_path.exists()
            
            datasets.append({
                'name': dataset_dir.name,
                'path': str(dataset_dir),
                'imageCount': len(images),
                'hasMetadata': has_metadata,
            })
    
    return datasets


def load_dataset_images(dataset_path: Path) -> List[Dict]:
    """Load all images from a dataset directory with their captions."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    images = []
    
    # Get all image files
    image_files = sorted([f for f in dataset_path.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions])
    
    for img_file in image_files:
        image_id = img_file.stem
        
        # Look for caption file
        caption_path = dataset_path / f"{image_id}.txt"
        caption = ""
        negative_prompt = ""
        
        if caption_path.exists():
            try:
                content = caption_path.read_text(encoding='utf-8')
                # Check for negative prompt separator
                if "### Negative Prompt ###" in content:
                    parts = content.split("### Negative Prompt ###")
                    caption = parts[0].strip()
                    negative_prompt = parts[1].strip() if len(parts) > 1 else ""
                else:
                    caption = content.strip()
            except Exception as e:
                print(f"Error reading caption for {image_id}: {e}")
        
        # Extract tags from caption (comma-separated items)
        tags = [t.strip() for t in caption.split(',') if t.strip()] if caption else []
        
        images.append({
            'id': image_id,
            'filename': img_file.name,
            'path': str(img_file),
            'caption': caption,
            'negativePrompt': negative_prompt,
            'tags': tags[:20],  # Limit displayed tags
            'hasCaption': bool(caption),
        })
    
    return images


@app.get("/api/caption-datasets")
async def get_caption_datasets():
    """Get list of available image datasets for caption review."""
    datasets = scan_image_datasets()
    return {
        'datasets': datasets,
        'totalDatasets': len(datasets),
    }


@app.get("/api/caption-dataset/{dataset_name}")
async def get_caption_dataset(dataset_name: str, skip: int = 0, limit: int = 50):
    """Get images from a specific dataset with pagination."""
    # Find the dataset
    dataset_path = None
    for dir_path in IMAGE_DATASET_DIRS:
        if dir_path.name == dataset_name and dir_path.exists():
            dataset_path = dir_path
            break
    
    if dataset_path is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_name}")
    
    # Load all images
    all_images = load_dataset_images(dataset_path)
    total = len(all_images)
    
    # Apply pagination
    paginated = all_images[skip:skip + limit]
    
    return {
        'datasetName': dataset_name,
        'images': paginated,
        'total': total,
        'skip': skip,
        'limit': limit,
        'hasMore': skip + limit < total,
    }


@app.get("/api/caption-image/{dataset_name}/{image_id}")
async def get_caption_image_file(dataset_name: str, image_id: str):
    """Serve an image file from a dataset."""
    # Find the dataset
    dataset_path = None
    for dir_path in IMAGE_DATASET_DIRS:
        if dir_path.name == dataset_name and dir_path.exists():
            dataset_path = dir_path
            break
    
    if dataset_path is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_name}")
    
    # Find the image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
    for ext in image_extensions:
        img_path = dataset_path / f"{image_id}{ext}"
        if img_path.exists():
            media_type = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
            return FileResponse(img_path, media_type=media_type)
    
    raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")


@app.put("/api/caption-dataset/{dataset_name}/{image_id}")
async def update_caption(dataset_name: str, image_id: str, update: CaptionUpdate):
    """Update the caption for an image."""
    # Find the dataset
    dataset_path = None
    for dir_path in IMAGE_DATASET_DIRS:
        if dir_path.name == dataset_name and dir_path.exists():
            dataset_path = dir_path
            break
    
    if dataset_path is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_name}")
    
    # Write caption file
    caption_path = dataset_path / f"{image_id}.txt"
    
    try:
        content = update.caption
        if update.negative_prompt:
            content += f"\n\n### Negative Prompt ###\n{update.negative_prompt}"
        
        caption_path.write_text(content, encoding='utf-8')
        
        return {
            'success': True,
            'imageId': image_id,
            'caption': update.caption,
            'negativePrompt': update.negative_prompt,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save caption: {str(e)}")


@app.delete("/api/caption-dataset/{dataset_name}/{image_id}")
async def delete_image(dataset_name: str, image_id: str):
    """Delete an image and its caption from a dataset."""
    # Find the dataset
    dataset_path = None
    for dir_path in IMAGE_DATASET_DIRS:
        if dir_path.name == dataset_name and dir_path.exists():
            dataset_path = dir_path
            break
    
    if dataset_path is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_name}")
    
    deleted_files = []
    
    # Delete image file
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
    for ext in image_extensions:
        img_path = dataset_path / f"{image_id}{ext}"
        if img_path.exists():
            img_path.unlink()
            deleted_files.append(img_path.name)
            break
    
    # Delete caption file
    caption_path = dataset_path / f"{image_id}.txt"
    if caption_path.exists():
        caption_path.unlink()
        deleted_files.append(caption_path.name)
    
    if not deleted_files:
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
    
    return {
        'success': True,
        'imageId': image_id,
        'deletedFiles': deleted_files,
    }


# =============================================================================
# AUDIO ANALYSIS ENDPOINTS
# =============================================================================

@app.get("/api/audio")
async def get_audio_data():
    """
    Get aggregated audio analysis data for visualization.
    Returns non-verbal cue distributions, acoustic profiles, and per-scene stats.
    Loads from both audio_analysis_pro.json and audio_analysis_hobby.json.
    """
    # Try to load both audio analysis files
    pro_data = load_json_file(ANALYSIS_DIR / "audio_analysis_pro.json")
    hobby_data = load_json_file(ANALYSIS_DIR / "audio_analysis_hobby.json")
    
    # If neither exists, return empty dataset instead of 404 (prevents frontend crashes)
    if pro_data is None and hobby_data is None:
        return {
            'summary': {
                'totalScenes': 0,
                'scenesWithAudio': 0,
                'scenesWithSpeech': 0,
                'avgDurationSeconds': 0,
                'avgSpeechRatio': 0,
                'avgSilenceRatio': 0,
                'dataAvailable': False,
                'message': 'Audio analysis data not found. Run audio_analyzer.py first.'
            },
            'cueDistribution': {},
            'valenceHintDistribution': {},
            'pitchTrendDistribution': {},
            'energyTrendDistribution': {},
            'durationHistogram': [],
            'speechRatioHistogram': [],
            'sceneStats': []
        }
    
    # Use pro data as primary source (has classification info)
    analyses = []
    if pro_data:
        analyses = pro_data.get('analyses', [])
    elif hobby_data:
        analyses = hobby_data.get('analyses', [])
    
    # Aggregate statistics
    total_scenes = len(analyses)
    scenes_with_audio = sum(1 for a in analyses if a.get('audio_present', False))
    scenes_with_speech = 0
    
    # Distribution counters
    cue_distribution: Dict[str, int] = {}
    valence_hint_distribution: Dict[str, int] = {}
    pitch_trend_distribution: Dict[str, int] = {}
    energy_trend_distribution: Dict[str, int] = {}
    
    # Duration data for histogram
    durations = []
    speech_ratios = []
    silence_ratios = []
    
    # Per-scene stats
    scene_stats = []
    
    for analysis in analyses:
        scene_path = analysis.get('scene_path', '')
        scene_name = Path(scene_path).stem
        
        audio_present = analysis.get('audio_present', False)
        
        if audio_present:
            duration = analysis.get('duration_seconds', 0)
            durations.append(duration)
            
            # Segmentation data
            segmentation = analysis.get('segmentation', {})
            speech_ratio = segmentation.get('speech_ratio', 0)
            non_verbal_ratio = segmentation.get('non_verbal_ratio', 0)
            silence_ratio = segmentation.get('silence_ratio', 0)
            
            speech_ratios.append(speech_ratio)
            silence_ratios.append(silence_ratio)
            
            if speech_ratio > 0.3:
                scenes_with_speech += 1
            
            # Classification data
            classification = analysis.get('classification', {})
            dominant_cue = classification.get('dominant_cue', 'unknown')
            cue_distribution[dominant_cue] = cue_distribution.get(dominant_cue, 0) + 1
            
            valence_hint = classification.get('valence_hint', 'neutral')
            valence_hint_distribution[valence_hint] = valence_hint_distribution.get(valence_hint, 0) + 1
            
            # Acoustic profile
            acoustic = analysis.get('acoustic_profile', {})
            pitch_trend = acoustic.get('pitch_trend', 'unknown')
            energy_trend = acoustic.get('energy_trend', 'unknown')
            
            pitch_trend_distribution[pitch_trend] = pitch_trend_distribution.get(pitch_trend, 0) + 1
            energy_trend_distribution[energy_trend] = energy_trend_distribution.get(energy_trend, 0) + 1
        
        # Build scene stat
        scene_stats.append({
            'sceneName': scene_name,
            'scenePath': scene_path,
            'audioPresent': audio_present,
            'durationSeconds': analysis.get('duration_seconds', 0),
            'processingStatus': analysis.get('processing_status', 'unknown'),
            'segmentation': analysis.get('segmentation'),
            'acousticProfile': analysis.get('acoustic_profile'),
            'classification': analysis.get('classification'),
            'cotContext': analysis.get('cot_context', ''),
        })
    
    # Create duration histogram
    duration_histogram = []
    if durations:
        max_dur = max(durations) if durations else 10
        bin_size = max(1, int(max_dur / 10))
        for start in range(0, int(max_dur) + bin_size, bin_size):
            end = start + bin_size
            count = sum(1 for d in durations if start <= d < end)
            duration_histogram.append({
                'range': f'{start}-{end}s',
                'min': start,
                'max': end,
                'count': count
            })
    
    # Create speech ratio histogram
    speech_ratio_histogram = []
    for start in range(0, 10):
        s = start / 10
        e = (start + 1) / 10
        count = sum(1 for r in speech_ratios if s <= r < e)
        speech_ratio_histogram.append({
            'range': f'{int(s*100)}-{int(e*100)}%',
            'min': s,
            'max': e,
            'count': count
        })
    
    return {
        'summary': {
            'totalScenes': total_scenes,
            'scenesWithAudio': scenes_with_audio,
            'scenesWithSpeech': scenes_with_speech,
            'avgDurationSeconds': sum(durations) / len(durations) if durations else 0,
            'avgSpeechRatio': sum(speech_ratios) / len(speech_ratios) if speech_ratios else 0,
            'avgSilenceRatio': sum(silence_ratios) / len(silence_ratios) if silence_ratios else 0,
        },
        'cueDistribution': cue_distribution,
        'valenceHintDistribution': valence_hint_distribution,
        'pitchTrendDistribution': pitch_trend_distribution,
        'energyTrendDistribution': energy_trend_distribution,
        'durationHistogram': duration_histogram,
        'speechRatioHistogram': speech_ratio_histogram,
        'sceneStats': scene_stats,
    }


@app.get("/api/audio/scene/{scene_name}")
async def get_scene_audio(scene_name: str):
    """
    Get detailed audio analysis for a single scene.
    Combines data from both pro and hobby tracks.
    """
    pro_data = load_json_file(ANALYSIS_DIR / "audio_analysis_pro.json")
    hobby_data = load_json_file(ANALYSIS_DIR / "audio_analysis_hobby.json")
    
    if pro_data is None and hobby_data is None:
        raise HTTPException(status_code=404, detail="Audio analysis data not found")
    
    # Find scene in pro data
    pro_analysis = None
    if pro_data:
        for analysis in pro_data.get('analyses', []):
            if scene_name in analysis.get('scene_path', ''):
                pro_analysis = analysis
                break
    
    # Find scene in hobby data
    hobby_analysis = None
    if hobby_data:
        for analysis in hobby_data.get('analyses', []):
            if scene_name in analysis.get('scene_path', ''):
                hobby_analysis = analysis
                break
    
    if pro_analysis is None and hobby_analysis is None:
        raise HTTPException(status_code=404, detail=f"Scene not found: {scene_name}")
    
    return {
        'sceneName': scene_name,
        'proTrack': pro_analysis,
        'hobbyTrack': hobby_analysis,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
