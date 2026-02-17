#!/usr/bin/env python3
"""
Robust Processing Utilities
===========================

Common utilities for fault-tolerant video processing:
- Incremental JSON writing (crash-resistant)
- Error handling with skip & flag
- Timeout handling for stuck processes
- Resume capability
- Progress tracking

Used by all detector scripts to ensure pipeline doesn't break.
"""

import json
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from contextlib import contextmanager
import traceback
import numpy as np


# =============================================================================
# NUMPY-SAFE JSON ENCODER
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_safe_dump(data: Any, f, **kwargs):
    """JSON dump that handles numpy types."""
    json.dump(data, f, cls=NumpyEncoder, **kwargs)


# =============================================================================
# TIMEOUT HANDLING
# =============================================================================

class TimeoutError(Exception):
    """Raised when a processing operation times out."""
    pass


@contextmanager
def timeout(seconds: int, error_message: str = "Operation timed out"):
    """
    Context manager for timeout handling.
    
    Usage:
        with timeout(30, "Video processing timed out"):
            process_video(path)
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# =============================================================================
# INCREMENTAL JSON WRITER
# =============================================================================

class IncrementalJSONWriter:
    """
    Writes JSON incrementally to prevent data loss on crashes.
    
    Maintains a valid JSON file at all times by:
    1. Writing a temp file with new data
    2. Atomically replacing the main file
    3. Keeping a backup of the last good state
    """
    
    def __init__(self, output_path: Path, backup_interval: int = 10):
        self.output_path = Path(output_path)
        self.temp_path = self.output_path.with_suffix('.tmp')
        self.backup_path = self.output_path.with_suffix('.backup.json')
        self.backup_interval = backup_interval
        self.write_count = 0
        
        # Initialize structure
        self.data = {
            'config': {},
            'summary': {},
            'analyses': [],
            'errors': [],
            'metadata': {
                'started_at': datetime.now().isoformat(),
                'last_updated': None,
                'total_processed': 0,
                'total_errors': 0,
                'total_skipped': 0
            }
        }
        
        # Try to resume from existing file
        self._try_resume()
    
    def _try_resume(self):
        """Load existing data if available for resume capability."""
        if self.output_path.exists():
            try:
                with open(self.output_path) as f:
                    existing = json.load(f)
                
                # Merge existing data with expected structure
                # (handles old files that don't have metadata/errors keys)
                self.data['config'] = existing.get('config', {})
                self.data['summary'] = existing.get('summary', {})
                self.data['analyses'] = existing.get('analyses', [])
                
                # Preserve or initialize errors list
                if 'errors' in existing:
                    self.data['errors'] = existing['errors']
                # else: keep the empty list from __init__
                
                # Preserve or initialize metadata
                if 'metadata' in existing:
                    self.data['metadata'] = existing['metadata']
                else:
                    # Set total_processed based on existing analyses
                    self.data['metadata']['total_processed'] = len(self.data['analyses'])
                
                print(f"  Resuming from existing file: {len(self.data.get('analyses', []))} entries")
            except Exception as e:
                print(f"  Starting fresh (existing file invalid: {e})")
    
    def get_processed_paths(self) -> set:
        """Get set of already-processed scene paths for resume."""
        processed = set()
        for analysis in self.data.get('analyses', []):
            if 'scene_path' in analysis:
                processed.add(analysis['scene_path'])
        return processed
    
    def set_config(self, config: Dict):
        """Set configuration section."""
        self.data['config'] = config
        self._write()
    
    def add_analysis(self, analysis: Dict):
        """Add a successful analysis result."""
        self.data['analyses'].append(analysis)
        self.data['metadata']['total_processed'] += 1
        self.data['metadata']['last_updated'] = datetime.now().isoformat()
        self.write_count += 1
        
        # Write periodically
        if self.write_count % self.backup_interval == 0:
            self._write()
    
    def add_error(self, scene_path: str, error_type: str, error_message: str):
        """Record an error for a scene (scene is skipped but logged)."""
        self.data['errors'].append({
            'scene_path': scene_path,
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        })
        self.data['metadata']['total_errors'] += 1
        self.data['metadata']['last_updated'] = datetime.now().isoformat()
        self._write()
    
    def add_skipped(self, scene_path: str, reason: str):
        """Record a skipped scene."""
        self.data['errors'].append({
            'scene_path': scene_path,
            'error_type': 'skipped',
            'error_message': reason,
            'timestamp': datetime.now().isoformat()
        })
        self.data['metadata']['total_skipped'] += 1
    
    def update_summary(self, summary: Dict):
        """Update summary statistics."""
        self.data['summary'] = summary
        self._write()
    
    def _write(self):
        """Write current state to file atomically."""
        try:
            # Write to temp file first (using numpy-safe encoder)
            with open(self.temp_path, 'w') as f:
                json.dump(self.data, f, indent=2, cls=NumpyEncoder)
            
            # Backup existing file
            if self.output_path.exists():
                self.output_path.rename(self.backup_path)
            
            # Atomic rename
            self.temp_path.rename(self.output_path)
            
        except Exception as e:
            print(f"  Warning: Could not write to {self.output_path}: {e}")
    
    def finalize(self):
        """Final write and cleanup."""
        self.data['metadata']['completed_at'] = datetime.now().isoformat()
        self._write()
        
        # Clean up backup if successful
        if self.backup_path.exists():
            self.backup_path.unlink()
        
        return self.data


# =============================================================================
# ROBUST SCENE PROCESSOR
# =============================================================================

@dataclass
class ProcessingResult:
    """Result of processing a single scene."""
    success: bool
    scene_path: str
    data: Optional[Dict] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0


def process_scene_robust(
    scene_path: Path,
    processor_func: Callable,
    timeout_seconds: int = 120,
    **processor_kwargs
) -> ProcessingResult:
    """
    Process a single scene with full error handling.
    
    Args:
        scene_path: Path to scene video
        processor_func: Function that takes scene_path and returns dict
        timeout_seconds: Max time before timeout
        **processor_kwargs: Additional args for processor_func
        
    Returns:
        ProcessingResult with success/failure info
    """
    start_time = datetime.now()
    
    try:
        # Check file exists
        if not scene_path.exists():
            return ProcessingResult(
                success=False,
                scene_path=str(scene_path),
                error_type='file_not_found',
                error_message=f"File does not exist: {scene_path}"
            )
        
        # Check file is readable
        if scene_path.stat().st_size == 0:
            return ProcessingResult(
                success=False,
                scene_path=str(scene_path),
                error_type='empty_file',
                error_message=f"File is empty: {scene_path}"
            )
        
        # Process with timeout
        with timeout(timeout_seconds, f"Timeout after {timeout_seconds}s"):
            result = processor_func(scene_path, **processor_kwargs)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            success=True,
            scene_path=str(scene_path),
            data=result,
            processing_time=elapsed
        )
        
    except TimeoutError as e:
        return ProcessingResult(
            success=False,
            scene_path=str(scene_path),
            error_type='timeout',
            error_message=str(e)
        )
    except MemoryError:
        return ProcessingResult(
            success=False,
            scene_path=str(scene_path),
            error_type='memory_error',
            error_message="Out of memory"
        )
    except Exception as e:
        return ProcessingResult(
            success=False,
            scene_path=str(scene_path),
            error_type=type(e).__name__,
            error_message=f"{str(e)}\n{traceback.format_exc()}"
        )


def process_all_scenes_robust(
    scene_paths: List[Path],
    processor_func: Callable,
    output_writer: IncrementalJSONWriter,
    timeout_seconds: int = 120,
    progress_callback: Optional[Callable] = None,
    **processor_kwargs
) -> Dict:
    """
    Process all scenes with robust error handling.
    
    Features:
    - Skips already-processed scenes (resume)
    - Catches and logs all errors
    - Writes incrementally
    - Reports progress
    
    Args:
        scene_paths: List of scene paths to process
        processor_func: Function to process each scene
        output_writer: IncrementalJSONWriter instance
        timeout_seconds: Max time per scene
        progress_callback: Optional callback(current, total, scene_path)
        **processor_kwargs: Additional args for processor_func
        
    Returns:
        Summary statistics
    """
    # Get already processed for resume
    processed = output_writer.get_processed_paths()
    
    total = len(scene_paths)
    successful = 0
    failed = 0
    skipped = 0
    
    for i, scene_path in enumerate(scene_paths):
        # Progress callback
        if progress_callback:
            progress_callback(i + 1, total, str(scene_path))
        
        # Skip if already processed
        if str(scene_path) in processed:
            skipped += 1
            continue
        
        # Process with error handling
        result = process_scene_robust(
            scene_path,
            processor_func,
            timeout_seconds,
            **processor_kwargs
        )
        
        if result.success:
            output_writer.add_analysis(result.data)
            successful += 1
        else:
            output_writer.add_error(
                result.scene_path,
                result.error_type,
                result.error_message
            )
            failed += 1
            print(f"  ⚠ Error on {scene_path.name}: {result.error_type}")
    
    return {
        'total': total,
        'successful': successful,
        'failed': failed,
        'skipped_resume': skipped
    }


# =============================================================================
# FILE VALIDATION
# =============================================================================

def validate_video_file(path: Path) -> tuple:
    """
    Quick validation of video file without full processing.
    
    Returns:
        (is_valid, error_message)
    """
    import cv2
    
    if not path.exists():
        return False, "File does not exist"
    
    if path.stat().st_size == 0:
        return False, "File is empty"
    
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, "Cannot open video"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            return False, "Video has no frames"
        
        # Try to read first frame
        ret, _ = cap.read()
        cap.release()
        
        if not ret:
            return False, "Cannot read first frame"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def prevalidate_scenes(scene_paths: List[Path], quick: bool = True) -> tuple:
    """
    Pre-validate scene files before processing.
    
    Args:
        scene_paths: List of paths to validate
        quick: If True, only check file existence and size
        
    Returns:
        (valid_paths, invalid_paths_with_reasons)
    """
    valid = []
    invalid = []
    
    for path in scene_paths:
        if quick:
            if not path.exists():
                invalid.append((path, "File does not exist"))
            elif path.stat().st_size == 0:
                invalid.append((path, "File is empty"))
            else:
                valid.append(path)
        else:
            is_valid, error = validate_video_file(path)
            if is_valid:
                valid.append(path)
            else:
                invalid.append((path, error))
    
    return valid, invalid
