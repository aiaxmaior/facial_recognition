#!/bin/bash
# =============================================================================
# Interactive Pipeline with AI Feedback Loop
# =============================================================================
# 
# This script runs the pipeline and on errors:
# 1. Writes error details to analysis/ai_inbox/error_report.json
# 2. Waits for AI response in analysis/ai_inbox/ai_response.json
# 3. Executes AI's instructions and continues
#
# The AI assistant can:
# - Read error_report.json to diagnose issues
# - Write ai_response.json with fix instructions
# - Script automatically picks up and executes the fix
#
# Usage:
#   ./run_pipeline_interactive.sh
#
# =============================================================================

set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/analysis/logs"
AI_INBOX="${SCRIPT_DIR}/analysis/ai_inbox"
ERROR_REPORT="${AI_INBOX}/error_report.json"
AI_RESPONSE="${AI_INBOX}/ai_response.json"
STATUS_FILE="${SCRIPT_DIR}/analysis/pipeline_status.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Timing
WAIT_INTERVAL=15        # Check for AI response every 15 seconds
MAX_WAIT_TIME=300       # Maximum wait time (5 minutes)

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${AI_INBOX}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Logging
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "${LOG_DIR}/pipeline_${TIMESTAMP}.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "${LOG_DIR}/pipeline_${TIMESTAMP}.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "${LOG_DIR}/pipeline_${TIMESTAMP}.log"
}

log_ai() {
    echo -e "${CYAN}[AI-COMM]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AI-COMM] $1" >> "${LOG_DIR}/pipeline_${TIMESTAMP}.log"
}

# =============================================================================
# AI Communication
# =============================================================================

write_error_report() {
    local step_num=$1
    local step_name=$2
    local exit_code=$3
    local log_file=$4
    
    # Get last 100 lines of log for context
    local log_tail=""
    if [ -f "$log_file" ]; then
        log_tail=$(tail -100 "$log_file" | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    fi
    
    # Get any Python traceback
    local traceback=""
    if [ -f "$log_file" ]; then
        traceback=$(grep -A 20 "Traceback" "$log_file" 2>/dev/null | tail -25 | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
    fi
    
    cat > "${ERROR_REPORT}" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "step_number": ${step_num},
    "step_name": "${step_name}",
    "exit_code": ${exit_code},
    "log_file": "${log_file}",
    "status": "awaiting_ai_response",
    "error_summary": {
        "last_100_lines": "${log_tail}",
        "traceback": "${traceback}"
    },
    "available_actions": [
        "retry - Retry the same step",
        "skip - Skip this step and continue",
        "run_command - Execute a custom command before retrying",
        "apply_fix - Apply a code fix (provide file path and content)",
        "abort - Stop the pipeline"
    ],
    "instructions": "AI: Write your response to ${AI_RESPONSE}"
}
EOF

    log_ai "Error report written to ${ERROR_REPORT}"
    log_ai "Waiting for AI response at ${AI_RESPONSE}"
}

wait_for_ai_response() {
    local waited=0
    
    # Clear any old response
    rm -f "${AI_RESPONSE}"
    
    log_ai "=================================================="
    log_ai "WAITING FOR AI ASSISTANT RESPONSE"
    log_ai "Error report: ${ERROR_REPORT}"
    log_ai "Write response to: ${AI_RESPONSE}"
    log_ai "Max wait time: ${MAX_WAIT_TIME} seconds"
    log_ai "=================================================="
    
    while [ $waited -lt $MAX_WAIT_TIME ]; do
        if [ -f "${AI_RESPONSE}" ]; then
            log_ai "AI response received!"
            return 0
        fi
        
        # Progress indicator
        local remaining=$((MAX_WAIT_TIME - waited))
        echo -ne "\r${CYAN}[AI-COMM]${NC} Waiting for AI response... ${remaining}s remaining    "
        
        sleep $WAIT_INTERVAL
        waited=$((waited + WAIT_INTERVAL))
    done
    
    echo ""
    log_warn "Timeout waiting for AI response"
    return 1
}

process_ai_response() {
    if [ ! -f "${AI_RESPONSE}" ]; then
        log_error "No AI response file found"
        return 1
    fi
    
    # Parse response
    local action=$(python3 -c "import json; print(json.load(open('${AI_RESPONSE}'))['action'])" 2>/dev/null)
    
    log_ai "AI action: ${action}"
    
    case "$action" in
        "retry")
            log_ai "AI requested retry"
            rm -f "${AI_RESPONSE}"
            return 0  # Will retry
            ;;
        "skip")
            log_ai "AI requested skip"
            rm -f "${AI_RESPONSE}"
            return 2  # Skip code
            ;;
        "run_command")
            local cmd=$(python3 -c "import json; print(json.load(open('${AI_RESPONSE}'))['command'])" 2>/dev/null)
            log_ai "AI requested command: ${cmd}"
            eval "$cmd"
            local cmd_result=$?
            log_ai "Command result: ${cmd_result}"
            rm -f "${AI_RESPONSE}"
            return 0  # Retry after command
            ;;
        "apply_fix")
            local fix_file=$(python3 -c "import json; print(json.load(open('${AI_RESPONSE}'))['file'])" 2>/dev/null)
            local fix_content=$(python3 -c "import json; print(json.load(open('${AI_RESPONSE}'))['content'])" 2>/dev/null)
            log_ai "AI applying fix to: ${fix_file}"
            echo "$fix_content" > "$fix_file"
            rm -f "${AI_RESPONSE}"
            return 0  # Retry after fix
            ;;
        "abort")
            log_ai "AI requested abort"
            rm -f "${AI_RESPONSE}"
            return 3  # Abort code
            ;;
        *)
            log_error "Unknown AI action: ${action}"
            rm -f "${AI_RESPONSE}"
            return 1
            ;;
    esac
}

# =============================================================================
# Step Execution with AI Loop
# =============================================================================

run_step_with_ai() {
    local step_num=$1
    local step_name=$2
    local command=$3
    local log_file="${LOG_DIR}/step${step_num}_${step_name}_${TIMESTAMP}.log"
    local max_retries=3
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        log_info "=========================================="
        log_info "Step ${step_num}: ${step_name} (attempt $((retry_count + 1)))"
        log_info "=========================================="
        
        # Run command
        set +e
        eval "${command}" > "${log_file}" 2>&1
        local exit_code=$?
        set -e
        
        if [ $exit_code -eq 0 ]; then
            log_info "Step ${step_num} completed successfully!"
            echo "SUCCESS" > "${LOG_DIR}/step${step_num}_status.txt"
            return 0
        fi
        
        log_error "Step ${step_num} failed with exit code ${exit_code}"
        
        # Write error report for AI
        write_error_report $step_num "$step_name" $exit_code "$log_file"
        
        # Wait for AI response
        if wait_for_ai_response; then
            process_ai_response
            local ai_action=$?
            
            case $ai_action in
                0)  # Retry
                    retry_count=$((retry_count + 1))
                    log_info "Retrying step ${step_num}..."
                    continue
                    ;;
                2)  # Skip
                    log_warn "Skipping step ${step_num} as requested by AI"
                    echo "SKIPPED (AI request)" > "${LOG_DIR}/step${step_num}_status.txt"
                    return 0
                    ;;
                3)  # Abort
                    log_error "Aborting pipeline as requested by AI"
                    return 1
                    ;;
                *)  # Error
                    log_error "AI response processing failed"
                    retry_count=$((retry_count + 1))
                    ;;
            esac
        else
            # Timeout - ask user what to do
            log_warn "AI response timeout. Continuing with default action (skip)..."
            echo "SKIPPED (timeout)" > "${LOG_DIR}/step${step_num}_status.txt"
            return 0
        fi
    done
    
    log_error "Max retries exceeded for step ${step_num}"
    return 1
}

# =============================================================================
# Pipeline Steps
# =============================================================================

run_all_steps() {
    log_info "Starting interactive pipeline..."
    
    # Step 0: Person Detection (regenerate detections.json if needed)
    if [ ! -s "${SCRIPT_DIR}/analysis/detections.json" ] || ! python3 -c "import json; json.load(open('${SCRIPT_DIR}/analysis/detections.json'))" 2>/dev/null; then
        log_warn "detections.json missing or invalid - running person_detector first"
        if ! run_step_with_ai 0 "person_detector" \
            "python ${SCRIPT_DIR}/scripts/person_detector.py"; then
            return 1
        fi
    else
        log_info "detections.json valid - skipping person_detector"
    fi
    
    # Step 1: Emotion Detection
    if ! run_step_with_ai 1 "emotion_detector" \
        "python ${SCRIPT_DIR}/scripts/emotion_detector.py"; then
        return 1
    fi
    
    # Step 2: Demographics Detection
    if ! run_step_with_ai 2 "demographics_detector" \
        "python ${SCRIPT_DIR}/scripts/demographics_detector.py"; then
        return 1
    fi
    
    # Step 3: NudeNet Processing
    if ! run_step_with_ai 3 "nudenet_processor" \
        "python ${SCRIPT_DIR}/scripts/nudenet_batch_processor.py"; then
        return 1
    fi
    
    # Step 4: VLM Captioning (check server first)
    if curl -s --connect-timeout 5 http://localhost:8000/health > /dev/null 2>&1; then
        if ! run_step_with_ai 4 "vlm_captioner" \
            "python ${SCRIPT_DIR}/scripts/vlm_captioner.py"; then
            return 1
        fi
    else
        log_warn "VLM server not available - skipping step 4"
        echo "SKIPPED (server unavailable)" > "${LOG_DIR}/step4_status.txt"
    fi
    
    # Step 5: Data Sanitizer
    if ! run_step_with_ai 5 "data_sanitizer" \
        "python ${SCRIPT_DIR}/scripts/data_sanitizer.py"; then
        return 1
    fi
    
    # Step 6: Final Assembly
    if ! run_step_with_ai 6 "final_assembler" \
        "python ${SCRIPT_DIR}/scripts/final_assembler.py"; then
        return 1
    fi
    
    return 0
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo ""
    echo "============================================================"
    echo "  INTERACTIVE PIPELINE WITH AI FEEDBACK LOOP"
    echo "============================================================"
    echo "  Log directory: ${LOG_DIR}"
    echo "  AI inbox: ${AI_INBOX}"
    echo "  "
    echo "  On errors, this script will:"
    echo "    1. Write error details to: ${ERROR_REPORT}"
    echo "    2. Wait up to ${MAX_WAIT_TIME}s for AI response"
    echo "    3. Execute AI instructions from: ${AI_RESPONSE}"
    echo "============================================================"
    echo ""
    
    # Clean up old AI communication files
    rm -f "${ERROR_REPORT}" "${AI_RESPONSE}"
    
    if run_all_steps; then
        log_info "============================================================"
        log_info "PIPELINE COMPLETED SUCCESSFULLY!"
        log_info "============================================================"
        log_info "Output files:"
        log_info "  - analysis/emotions.json"
        log_info "  - analysis/demographics.json"
        log_info "  - analysis/nudenet.json"
        log_info "  - analysis/captions.json"
        log_info "  - analysis/sanitized_stats.json (AI-safe)"
        log_info "  - curated/unified_dataset.csv"
        log_info "============================================================"
    else
        log_error "============================================================"
        log_error "PIPELINE FAILED"
        log_error "Check logs in: ${LOG_DIR}"
        log_error "============================================================"
        exit 1
    fi
}

main "$@"
