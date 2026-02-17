#!/usr/bin/env python3
"""
Pipeline Status Checker
=======================

Reads pipeline status and log files to provide diagnostic information.
Designed for AI assistant to review without accessing raw data.

Outputs:
- Pipeline progress (which steps completed/failed)
- Error summaries (last N lines of failed step logs)
- Timing information
- Recommendations for next steps

Usage:
    python check_pipeline_status.py
    python check_pipeline_status.py --errors-only
    python check_pipeline_status.py --step 3
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent.parent
STATUS_FILE = SCRIPT_DIR / "analysis" / "pipeline_status.json"
LOG_DIR = SCRIPT_DIR / "analysis" / "logs"

def load_status() -> dict:
    """Load pipeline status file."""
    if STATUS_FILE.exists():
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {}

def get_step_logs(step_num: int, lines: int = 30) -> str:
    """Get last N lines from a step's status file."""
    status_file = LOG_DIR / f"step{step_num}_status.txt"
    if status_file.exists():
        with open(status_file) as f:
            content = f.read()
        return content
    return "No status file found"

def get_latest_log(step_num: int, lines: int = 50) -> str:
    """Get last N lines from a step's full log."""
    # Find most recent log file for this step
    log_files = sorted(LOG_DIR.glob(f"step{step_num}_*.log"), reverse=True)
    if log_files:
        with open(log_files[0]) as f:
            all_lines = f.readlines()
        return ''.join(all_lines[-lines:])
    return "No log file found"

def print_status_summary():
    """Print overall pipeline status."""
    status = load_status()
    
    if not status:
        print("=" * 60)
        print("PIPELINE STATUS: NOT STARTED")
        print("=" * 60)
        print("No pipeline_status.json found.")
        print("Run: ./run_pipeline.sh")
        return
    
    print("=" * 60)
    print("PIPELINE STATUS SUMMARY")
    print("=" * 60)
    print(f"Started: {status.get('pipeline_start', 'N/A')}")
    print(f"Last Updated: {status.get('last_updated', 'N/A')}")
    print(f"Current Step: {status.get('current_step', 'N/A')}")
    print()
    
    steps = status.get('steps', {})
    
    # Step names
    step_names = {
        '1': 'Emotion Detection',
        '2': 'Demographics Detection', 
        '3': 'NudeNet Processing',
        '4': 'VLM Captioning',
        '5': 'Data Sanitizer',
        '6': 'Final Assembly'
    }
    
    print("STEP STATUS:")
    print("-" * 60)
    
    failed_steps = []
    completed_steps = []
    
    for step_num in ['1', '2', '3', '4', '5', '6']:
        step_data = steps.get(step_num, {})
        name = step_names.get(step_num, 'Unknown')
        status_val = step_data.get('status', 'pending')
        duration = step_data.get('duration_seconds', 0)
        message = step_data.get('message', '')
        
        # Status emoji
        if status_val == 'completed':
            emoji = '✓'
            completed_steps.append(step_num)
        elif status_val == 'failed':
            emoji = '✗'
            failed_steps.append(step_num)
        elif status_val == 'running':
            emoji = '⟳'
        elif status_val == 'skipped':
            emoji = '⊘'
        else:
            emoji = '○'
        
        duration_str = f"({duration}s)" if duration > 0 else ""
        print(f"  {emoji} Step {step_num}: {name:<25} {status_val:<10} {duration_str}")
        if message and status_val == 'failed':
            print(f"       └─ {message}")
    
    print()
    
    # Overall status
    pipeline_status = steps.get('0', {}).get('status', 'unknown')
    print(f"OVERALL: {pipeline_status.upper()}")
    print("=" * 60)
    
    # Recommendations
    print()
    print("RECOMMENDATIONS:")
    print("-" * 60)
    
    if failed_steps:
        print(f"  Failed steps: {', '.join(failed_steps)}")
        print(f"  To view errors: python scripts/check_pipeline_status.py --step {failed_steps[0]}")
        print(f"  To retry: ./run_pipeline.sh --from {failed_steps[0]}")
    elif pipeline_status == 'completed':
        print("  ✓ Pipeline completed successfully!")
        print("  View results: analysis/sanitized_stats.json (AI-safe)")
        print("  User CSV: curated/unified_dataset.csv")
    else:
        next_step = str(len(completed_steps) + 1)
        print(f"  Continue from step {next_step}: ./run_pipeline.sh --from {next_step}")
    
    print()

def print_step_errors(step_num: int):
    """Print error details for a specific step."""
    print("=" * 60)
    print(f"STEP {step_num} ERROR DETAILS")
    print("=" * 60)
    
    # Status file (concise)
    print("\nStatus Summary:")
    print("-" * 40)
    print(get_step_logs(step_num))
    
    # Full log (last 50 lines)
    print("\nFull Log (last 50 lines):")
    print("-" * 40)
    print(get_latest_log(step_num, 50))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check pipeline status")
    parser.add_argument('--step', '-s', type=int, help="Show details for specific step")
    parser.add_argument('--errors-only', '-e', action='store_true', help="Only show failed steps")
    
    args = parser.parse_args()
    
    if args.step:
        print_step_errors(args.step)
    else:
        print_status_summary()

if __name__ == "__main__":
    main()
