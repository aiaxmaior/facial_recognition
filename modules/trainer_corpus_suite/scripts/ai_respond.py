#!/usr/bin/env python3
"""
AI Response Writer for Interactive Pipeline
============================================

Helper script for the AI assistant to write responses to the pipeline.
This creates the ai_response.json file that the pipeline script reads.

Actions:
- retry: Retry the failed step
- skip: Skip the step and continue
- run_command: Run a shell command before retrying
- apply_fix: Write content to a file before retrying
- abort: Stop the pipeline

Usage (by AI):
    python ai_respond.py retry
    python ai_respond.py skip
    python ai_respond.py run_command "pip install missing-package"
    python ai_respond.py apply_fix path/to/file.py "file content here"
    python ai_respond.py abort
"""

import json
import sys
from pathlib import Path
from datetime import datetime

AI_INBOX = Path(__file__).parent.parent / "analysis" / "ai_inbox"
AI_RESPONSE = AI_INBOX / "ai_response.json"

def write_response(action: str, **kwargs):
    """Write AI response file."""
    AI_INBOX.mkdir(parents=True, exist_ok=True)
    
    response = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        **kwargs
    }
    
    with open(AI_RESPONSE, 'w') as f:
        json.dump(response, f, indent=2)
    
    print(f"AI response written: {action}")
    print(f"File: {AI_RESPONSE}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python ai_respond.py <action> [args...]")
        print("Actions: retry, skip, run_command, apply_fix, abort")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == "retry":
        write_response("retry")
    
    elif action == "skip":
        write_response("skip")
    
    elif action == "run_command":
        if len(sys.argv) < 3:
            print("Usage: python ai_respond.py run_command 'command here'")
            sys.exit(1)
        write_response("run_command", command=sys.argv[2])
    
    elif action == "apply_fix":
        if len(sys.argv) < 4:
            print("Usage: python ai_respond.py apply_fix path/to/file 'content'")
            sys.exit(1)
        write_response("apply_fix", file=sys.argv[2], content=sys.argv[3])
    
    elif action == "abort":
        write_response("abort")
    
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: retry, skip, run_command, apply_fix, abort")
        sys.exit(1)

if __name__ == "__main__":
    main()
