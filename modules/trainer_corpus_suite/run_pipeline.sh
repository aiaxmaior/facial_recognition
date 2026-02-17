#!/bin/bash
# =============================================================================
# Video Analysis Pipeline Orchestrator
# =============================================================================
# Runs all detector scripts in sequence with comprehensive logging.
# Logs are written to analysis/logs/ for AI assistant review.
#
# Usage:
#   ./run_pipeline.sh              # Run full pipeline
#   ./run_pipeline.sh --from 3     # Resume from step 3
#   ./run_pipeline.sh --only 2     # Run only step 2
#   ./run_pipeline.sh --skip-vlm   # Skip VLM captioning (requires server)
#
# =============================================================================

set -e  # Exit on error (can be overridden per-step)

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/analysis/logs"
STATUS_FILE="${SCRIPT_DIR}/analysis/pipeline_status.json"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p "${LOG_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
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

# =============================================================================
# Status Tracking
# =============================================================================

update_status() {
    local step=$1
    local name=$2
    local status=$3
    local message=$4
    local duration=$5
    
    # Create/update status JSON
    python3 << EOF
import json
from pathlib import Path
from datetime import datetime

status_file = Path("${STATUS_FILE}")

# Load existing or create new
if status_file.exists():
    with open(status_file) as f:
        data = json.load(f)
else:
    data = {"pipeline_start": datetime.now().isoformat(), "steps": {}}

# Update step
data["steps"]["${step}"] = {
    "name": "${name}",
    "status": "${status}",
    "message": "${message}",
    "duration_seconds": ${duration:-0},
    "timestamp": datetime.now().isoformat()
}

data["last_updated"] = datetime.now().isoformat()
data["current_step"] = "${step}"

# Write back
with open(status_file, 'w') as f:
    json.dump(data, f, indent=2)
EOF
}

# =============================================================================
# Step Execution
# =============================================================================

run_step() {
    local step_num=$1
    local step_name=$2
    local command=$3
    local log_file="${LOG_DIR}/step${step_num}_${step_name}_${TIMESTAMP}.log"
    
    log_info "=========================================="
    log_info "Step ${step_num}: ${step_name}"
    log_info "=========================================="
    log_info "Command: ${command}"
    log_info "Log file: ${log_file}"
    
    update_status "${step_num}" "${step_name}" "running" "Started" "0"
    
    local start_time=$(date +%s)
    
    # Run command and capture output
    set +e  # Don't exit on error for this command
    eval "${command}" > "${log_file}" 2>&1
    local exit_code=$?
    set -e
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        log_info "Step ${step_num} completed successfully (${duration}s)"
        update_status "${step_num}" "${step_name}" "completed" "Success" "${duration}"
        
        # Write success marker
        echo "SUCCESS" > "${LOG_DIR}/step${step_num}_status.txt"
        return 0
    else
        log_error "Step ${step_num} FAILED with exit code ${exit_code}"
        log_error "Check log file: ${log_file}"
        update_status "${step_num}" "${step_name}" "failed" "Exit code: ${exit_code}" "${duration}"
        
        # Write failure marker with last 50 lines of error
        echo "FAILED (exit code: ${exit_code})" > "${LOG_DIR}/step${step_num}_status.txt"
        echo "=== Last 50 lines of log ===" >> "${LOG_DIR}/step${step_num}_status.txt"
        tail -50 "${log_file}" >> "${LOG_DIR}/step${step_num}_status.txt"
        
        return $exit_code
    fi
}

# =============================================================================
# Pipeline Steps
# =============================================================================

step1_emotion() {
    run_step 1 "emotion_detector" "python ${SCRIPT_DIR}/scripts/emotion_detector.py"
}

step2_demographics() {
    run_step 2 "demographics_detector" "python ${SCRIPT_DIR}/scripts/demographics_detector.py"
}

step3_nudenet() {
    run_step 3 "nudenet_processor" "python ${SCRIPT_DIR}/scripts/nudenet_batch_processor.py"
}

step4_vlm() {
    # Check if VLM server is reachable
    if curl -s --connect-timeout 5 http://localhost:8000/health > /dev/null 2>&1; then
        run_step 4 "vlm_captioner" "python ${SCRIPT_DIR}/scripts/vlm_captioner.py"
    else
        log_warn "VLM server not reachable at localhost:8000"
        log_warn "Skipping VLM captioning - start server and rerun with --from 4"
        update_status "4" "vlm_captioner" "skipped" "Server not available" "0"
        echo "SKIPPED (server not available)" > "${LOG_DIR}/step4_status.txt"
    fi
}

step5_sanitizer() {
    run_step 5 "data_sanitizer" "python ${SCRIPT_DIR}/scripts/data_sanitizer.py"
}

step6_assembler() {
    run_step 6 "final_assembler" "python ${SCRIPT_DIR}/scripts/final_assembler.py"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local start_from=1
    local only_step=0
    local skip_vlm=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --from)
                start_from=$2
                shift 2
                ;;
            --only)
                only_step=$2
                shift 2
                ;;
            --skip-vlm)
                skip_vlm=true
                shift
                ;;
            --help)
                echo "Usage: $0 [--from N] [--only N] [--skip-vlm]"
                echo "  --from N    Resume from step N"
                echo "  --only N    Run only step N"
                echo "  --skip-vlm  Skip VLM captioning step"
                echo ""
                echo "Steps:"
                echo "  1: Emotion Detection"
                echo "  2: Demographics Detection"
                echo "  3: NudeNet Processing"
                echo "  4: VLM Captioning (requires server)"
                echo "  5: Data Sanitizer"
                echo "  6: Final Assembly"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    log_info "=========================================="
    log_info "Video Analysis Pipeline"
    log_info "=========================================="
    log_info "Timestamp: ${TIMESTAMP}"
    log_info "Log directory: ${LOG_DIR}"
    log_info "Starting from step: ${start_from}"
    if [ $only_step -ne 0 ]; then
        log_info "Running only step: ${only_step}"
    fi
    log_info "=========================================="
    
    # Initialize status
    update_status "0" "pipeline" "started" "Initializing" "0"
    
    # Run steps
    local failed=false
    
    if [ $only_step -ne 0 ]; then
        # Run only specified step
        case $only_step in
            1) step1_emotion || failed=true ;;
            2) step2_demographics || failed=true ;;
            3) step3_nudenet || failed=true ;;
            4) step4_vlm || failed=true ;;
            5) step5_sanitizer || failed=true ;;
            6) step6_assembler || failed=true ;;
            *) log_error "Invalid step: $only_step"; exit 1 ;;
        esac
    else
        # Run from start_from to end
        [ $start_from -le 1 ] && { step1_emotion || failed=true; }
        [ $start_from -le 2 ] && [ "$failed" = false ] && { step2_demographics || failed=true; }
        [ $start_from -le 3 ] && [ "$failed" = false ] && { step3_nudenet || failed=true; }
        [ $start_from -le 4 ] && [ "$failed" = false ] && [ "$skip_vlm" = false ] && { step4_vlm || failed=true; }
        [ $start_from -le 5 ] && [ "$failed" = false ] && { step5_sanitizer || failed=true; }
        [ $start_from -le 6 ] && [ "$failed" = false ] && { step6_assembler || failed=true; }
    fi
    
    # Final status
    log_info "=========================================="
    if [ "$failed" = true ]; then
        log_error "Pipeline FAILED - check logs in ${LOG_DIR}"
        update_status "0" "pipeline" "failed" "Check logs" "0"
        exit 1
    else
        log_info "Pipeline COMPLETED successfully!"
        update_status "0" "pipeline" "completed" "All steps done" "0"
        
        # Summary
        log_info ""
        log_info "Output files:"
        log_info "  - analysis/emotions.json"
        log_info "  - analysis/demographics.json"
        log_info "  - analysis/nudenet.json"
        log_info "  - analysis/captions.json (if VLM ran)"
        log_info "  - analysis/sanitized_stats.json (AI-safe)"
        log_info "  - curated/unified_dataset.csv"
        log_info ""
        log_info "For AI review: cat analysis/pipeline_status.json"
    fi
    log_info "=========================================="
}

# Run main
main "$@"
