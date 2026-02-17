#!/bin/bash
# =============================================================================
# Unified Audio Analysis Pipeline
# =============================================================================
# Processes scene clips through comprehensive audio analysis pipeline.
# Produces outputs for both professional (CoT reasoning) and hobby (diffusion) tracks.
#
# Outputs:
#   - analysis/audio_analysis_pro.json    (Professional track - CoT context)
#   - analysis/audio_analysis_hobby.json  (Hobby track - embeddings + features)
#   - audio_embeddings/*.npy              (Embedding files for diffusion)
#
# Usage:
#   ./run_audio_analysis.sh                    # Both outputs (default)
#   ./run_audio_analysis.sh --pro              # Professional context only
#   ./run_audio_analysis.sh --hobby            # Hobby/diffusion context only
#   ./run_audio_analysis.sh --test             # First 10 clips
#   ./run_audio_analysis.sh --skip-clap        # Skip CLAP model (faster)
#   ./run_audio_analysis.sh --skip-emotion     # Skip emotion2vec
#   ./run_audio_analysis.sh --skip-vad         # Skip VAD
#
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/analysis/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create log directory
mkdir -p "${LOG_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Logging Functions
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "${LOG_DIR}/audio_analysis_${TIMESTAMP}.log"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "${LOG_DIR}/audio_analysis_${TIMESTAMP}.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "${LOG_DIR}/audio_analysis_${TIMESTAMP}.log"
}

# =============================================================================
# Help
# =============================================================================

show_help() {
    echo "Unified Audio Analysis Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Output Modes:"
    echo "  --pro              Professional track only (CoT context for VLM)"
    echo "  --hobby            Hobby/diffusion track only (embeddings + features)"
    echo "  (default)          Both outputs"
    echo ""
    echo "Processing Options:"
    echo "  --test             Process only first 10 clips (quick test)"
    echo "  --test N           Process only first N clips"
    echo "  --skip-clap        Skip CLAP model (faster, less VRAM)"
    echo "  --skip-emotion     Skip emotion2vec model"
    echo "  --skip-vad         Skip pyannote VAD"
    echo "  --cpu              Use CPU instead of CUDA"
    echo ""
    echo "Other:"
    echo "  --help             Show this help message"
    echo ""
    echo "Outputs:"
    echo "  analysis/audio_analysis_pro.json    - Professional track"
    echo "  analysis/audio_analysis_hobby.json  - Hobby track"
    echo "  audio_embeddings/*.npy              - Embedding files"
    echo ""
    echo "Examples:"
    echo "  $0                          # Full run, both outputs"
    echo "  $0 --pro                    # Professional output only"
    echo "  $0 --hobby --skip-clap      # Hobby output, no CLAP"
    echo "  $0 --test 5                 # Test on 5 clips"
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

OUTPUT_MODE="both"
TEST_CLIPS=""
SKIP_CLAP=""
SKIP_EMOTION=""
SKIP_VAD=""
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --pro)
            OUTPUT_MODE="pro"
            shift
            ;;
        --hobby)
            OUTPUT_MODE="hobby"
            shift
            ;;
        --test)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                TEST_CLIPS="$2"
                shift 2
            else
                TEST_CLIPS="10"
                shift
            fi
            ;;
        --skip-clap)
            SKIP_CLAP="--skip-clap"
            shift
            ;;
        --skip-emotion)
            SKIP_EMOTION="--skip-emotion"
            shift
            ;;
        --skip-vad)
            SKIP_VAD="--skip-vad"
            shift
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Pre-flight Checks
# =============================================================================

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}Unified Audio Analysis Pipeline${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

# Check Python
if ! command -v python &> /dev/null; then
    log_error "Python not found in PATH"
    exit 1
fi

# Check if scenes directory exists
if [ ! -d "${SCRIPT_DIR}/scenes" ]; then
    log_error "scenes/ directory not found"
    echo "Have you run person_detector.py yet?"
    exit 1
fi

# Check scene count
SCENE_COUNT=$(find "${SCRIPT_DIR}/scenes" -name "*.mp4" -o -name "*.MP4" 2>/dev/null | wc -l)
if [ "$SCENE_COUNT" -eq 0 ]; then
    log_error "No video files found in scenes/"
    exit 1
fi

log_info "Found ${SCENE_COUNT} scene files"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    log_error "FFmpeg not found - required for audio extraction"
    exit 1
fi

if ! command -v ffprobe &> /dev/null; then
    log_error "FFprobe not found - required for audio stream detection"
    exit 1
fi

log_info "FFmpeg available"

# Check CUDA if requested
if [ "$DEVICE" = "cuda" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        log_info "CUDA available: ${GPU_NAME}"
    else
        log_warn "CUDA not available, falling back to CPU"
        DEVICE="cpu"
    fi
fi

# =============================================================================
# Build Command
# =============================================================================

CMD="python ${SCRIPT_DIR}/scripts/audio_analyzer.py"
CMD="$CMD --output ${OUTPUT_MODE}"
CMD="$CMD --device ${DEVICE}"

if [ -n "$TEST_CLIPS" ]; then
    CMD="$CMD --test ${TEST_CLIPS}"
    log_info "Test mode: processing first ${TEST_CLIPS} clips"
fi

if [ -n "$SKIP_CLAP" ]; then
    CMD="$CMD ${SKIP_CLAP}"
    log_info "Skipping CLAP model"
fi

if [ -n "$SKIP_EMOTION" ]; then
    CMD="$CMD ${SKIP_EMOTION}"
    log_info "Skipping emotion2vec model"
fi

if [ -n "$SKIP_VAD" ]; then
    CMD="$CMD ${SKIP_VAD}"
    log_info "Skipping VAD model"
fi

echo ""
log_info "Output mode: ${OUTPUT_MODE}"
log_info "Device: ${DEVICE}"
echo ""
log_info "Command: ${CMD}"
echo ""

# =============================================================================
# Run Analysis
# =============================================================================

LOG_FILE="${LOG_DIR}/audio_analysis_${TIMESTAMP}.log"
log_info "Log file: ${LOG_FILE}"
echo ""

START_TIME=$(date +%s)

# Run with output to both console and log
set +e  # Don't exit on error
eval "${CMD}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""

# =============================================================================
# Report Results
# =============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}======================================================================${NC}"
    echo -e "${GREEN}Audio Analysis Complete!${NC}"
    echo -e "${GREEN}======================================================================${NC}"
    echo ""
    
    if [ "$OUTPUT_MODE" = "pro" ] || [ "$OUTPUT_MODE" = "both" ]; then
        if [ -f "${SCRIPT_DIR}/analysis/audio_analysis_pro.json" ]; then
            PRO_COUNT=$(python -c "import json; print(len(json.load(open('${SCRIPT_DIR}/analysis/audio_analysis_pro.json')).get('analyses', [])))" 2>/dev/null || echo "?")
            echo "Professional output: analysis/audio_analysis_pro.json (${PRO_COUNT} scenes)"
        fi
    fi
    
    if [ "$OUTPUT_MODE" = "hobby" ] || [ "$OUTPUT_MODE" = "both" ]; then
        if [ -f "${SCRIPT_DIR}/analysis/audio_analysis_hobby.json" ]; then
            HOBBY_COUNT=$(python -c "import json; print(len(json.load(open('${SCRIPT_DIR}/analysis/audio_analysis_hobby.json')).get('analyses', [])))" 2>/dev/null || echo "?")
            echo "Hobby output:        analysis/audio_analysis_hobby.json (${HOBBY_COUNT} scenes)"
        fi
        
        EMB_COUNT=$(find "${SCRIPT_DIR}/audio_embeddings" -name "*.npy" 2>/dev/null | wc -l)
        if [ "$EMB_COUNT" -gt 0 ]; then
            echo "Embeddings:          audio_embeddings/ (${EMB_COUNT} files)"
        fi
    fi
    
    echo ""
    echo "Processing time: ${DURATION}s"
    echo ""
    echo "Next steps:"
    echo "  - View pro summary:   jq '.summary' analysis/audio_analysis_pro.json"
    echo "  - View hobby summary: jq '.summary' analysis/audio_analysis_hobby.json"
    echo ""
else
    echo -e "${RED}======================================================================${NC}"
    echo -e "${RED}Audio Analysis FAILED${NC}"
    echo -e "${RED}======================================================================${NC}"
    echo ""
    echo "Exit code: ${EXIT_CODE}"
    echo "Check log file: ${LOG_FILE}"
    echo ""
    
    # Show last 20 lines of log
    echo "Last 20 lines of log:"
    echo "----------------------------------------------------------------------"
    tail -20 "${LOG_FILE}"
    echo "----------------------------------------------------------------------"
    
    exit $EXIT_CODE
fi
