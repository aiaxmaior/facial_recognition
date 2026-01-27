import React, { useEffect, useCallback, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { useCamera } from '../hooks/useCamera';
import { useFacePose } from '../hooks/useFacePose';
import { useEnrollment } from '../hooks/useEnrollment';
import { getAudioService } from '../services/audioService';
import { CameraFeed } from './CameraFeed';
import { CaptureProgress } from './CaptureProgress';
import { CAPTURE_TARGETS, STABLE_FRAME_THRESHOLD, AUTO_CLOSE_DELAY } from '../constants';
import type { EnrollmentModalProps, EnrollmentStatus } from '../types/enrollment';

/**
 * Main enrollment modal component
 */
export function EnrollmentModal({
  isOpen,
  onClose,
  userId,
  enrollmentStatus,
  apiEndpoint,
  onEnrollmentComplete,
  userName,
  enableAudio = true,
  className,
}: EnrollmentModalProps) {
  const audioService = getAudioService();
  const prevStepRef = useRef(-1);
  const autoCloseTimerRef = useRef<NodeJS.Timeout | null>(null);
  const [autoCloseCountdown, setAutoCloseCountdown] = useState(AUTO_CLOSE_DELAY);
  const [isStabilizing, setIsStabilizing] = useState(false);
  const stableCountRef = useRef(0);
  
  // Re-enrollment confirmation state
  const [showReenrollConfirm, setShowReenrollConfirm] = useState(false);
  const [reenrollConfirmed, setReenrollConfirmed] = useState(false);
  
  // Check if user already has enrollment data
  const hasExistingEnrollment = enrollmentStatus === 'enrolled' || enrollmentStatus === 'pending';

  // Camera hook
  const {
    videoRef,
    canvasRef,
    isReady: cameraReady,
    error: cameraError,
    startCamera,
    stopCamera,
    captureFrame,
    getVideoElement,
  } = useCamera();

  // Face pose detection hook
  const {
    isLoading: poseLoading,
    isReady: poseReady,
    error: poseError,
    currentPose,
    initialize: initializePose,
    startDetection,
    stopDetection,
    checkPoseMatch,
  } = useFacePose();

  // Enrollment state machine
  const {
    captureState,
    currentStep,
    captures,
    currentTarget,
    error: enrollmentError,
    result,
    isCountingDown,
    countdownSeconds,
    isPoseValid,
    startCapture,
    resetCapture,
    processPose,
    captureFrame: doCapture,
  } = useEnrollment({
    userId,
    apiEndpoint,
    onComplete: onEnrollmentComplete,
  });

  // Initialize on open
  useEffect(() => {
    if (isOpen) {
      console.log('[EnrollmentModal] Modal opened, initializing...');
      audioService.setEnabled(enableAudio);
      initializePose();
      startCamera();
      
      // Show re-enrollment confirmation if user already has enrollment
      if (hasExistingEnrollment && !reenrollConfirmed) {
        setShowReenrollConfirm(true);
      }
    } else {
      console.log('[EnrollmentModal] Modal closed, cleaning up...');
      stopCamera();
      stopDetection();
      resetCapture();
      setShowReenrollConfirm(false);
      setReenrollConfirmed(false);
      
      if (autoCloseTimerRef.current) {
        clearInterval(autoCloseTimerRef.current);
        autoCloseTimerRef.current = null;
      }
    }

    return () => {
      stopCamera();
      stopDetection();
    };
  }, [isOpen, hasExistingEnrollment, reenrollConfirmed]);

  // Start detection when camera and pose model are ready
  useEffect(() => {
    if (cameraReady && poseReady) {
      const video = getVideoElement();
      if (video) {
        startDetection(video);
      }
    }
  }, [cameraReady, poseReady, getVideoElement, startDetection]);

  // Process pose updates
  useEffect(() => {
    if (captureState === 'capturing') {
      processPose(currentPose);
      
      // Track stabilization
      if (currentTarget && checkPoseMatch(currentTarget) && !isCountingDown) {
        stableCountRef.current++;
        setIsStabilizing(stableCountRef.current < STABLE_FRAME_THRESHOLD);
      } else {
        stableCountRef.current = 0;
        setIsStabilizing(false);
      }
    }
  }, [currentPose, captureState, processPose, currentTarget, checkPoseMatch, isCountingDown]);

  // Secondary audio guidance when pose doesn't match
  useEffect(() => {
    if (captureState !== 'capturing' || !currentTarget || isCountingDown || !currentPose.detected) {
      return;
    }

    // Don't play guidance if pose is valid
    const poseMatches = checkPoseMatch(currentTarget);
    if (poseMatches) {
      return;
    }

    // Get appropriate guidance audio based on how user needs to adjust
    const guidanceKey = audioService.getGuidanceAudioKey(
      currentPose.yaw,
      currentPose.pitch,
      currentTarget.name,
      currentTarget.yawRange,
      currentTarget.pitchRange
    );

    // Log guidance decision periodically
    if (Math.random() < 0.05 && guidanceKey) {
      console.log(`[Guidance] Target: ${currentTarget.name} | Yaw: ${currentPose.yaw.toFixed(1)} (need ${currentTarget.yawRange[0]} to ${currentTarget.yawRange[1]}) | Pitch: ${currentPose.pitch.toFixed(1)} (need ${currentTarget.pitchRange[0]} to ${currentTarget.pitchRange[1]}) | Key: ${guidanceKey}`);
    }

    if (guidanceKey) {
      audioService.playAdvisory(guidanceKey);
    }
  }, [captureState, currentTarget, isCountingDown, currentPose, checkPoseMatch]);

  // Handle frame capture
  useEffect(() => {
    if (captureState === 'capturing' && isCountingDown && countdownSeconds === 0) {
      console.log('[EnrollmentModal] 📸 Countdown reached 0 - capturing frame...');
      const imageData = captureFrame();
      if (imageData) {
        console.log('[EnrollmentModal] ✅ Frame captured, processing...');
        doCapture(imageData);
        
        // Only play completion audio on the LAST capture (step 4 = 5th capture, index 0-4)
        const isLastCapture = currentStep >= 4;
        if (isLastCapture) {
          console.log('[EnrollmentModal] 🔊 Playing capture complete audio (final capture)');
          audioService.playComplete();
        } else {
          // Play a quick beep for intermediate captures
          audioService.playBeep();
        }
      } else {
        console.error('[EnrollmentModal] ❌ Failed to capture frame!');
      }
    }
  }, [captureState, isCountingDown, countdownSeconds, captureFrame, doCapture, currentStep]);

  // Audio cues for step changes
  useEffect(() => {
    if (currentStep !== prevStepRef.current && captureState === 'capturing') {
      prevStepRef.current = currentStep;
      audioService.resetAdvisoryState();
      
      if (currentTarget) {
        console.log(`[EnrollmentModal] 🔊 Playing direction audio for: ${currentTarget.name}`);
        audioService.playDirection(currentTarget.name);
      }
    }
  }, [currentStep, captureState, currentTarget]);

  // Audio cue for countdown beeps
  useEffect(() => {
    if (isCountingDown && countdownSeconds > 0) {
      console.log(`[EnrollmentModal] 🔔 Countdown beep: ${countdownSeconds}`);
      audioService.playBeep();
    }
  }, [isCountingDown, countdownSeconds]);

  // Auto-close after completion
  useEffect(() => {
    if (captureState === 'complete') {
      setAutoCloseCountdown(AUTO_CLOSE_DELAY);
      
      autoCloseTimerRef.current = setInterval(() => {
        setAutoCloseCountdown((prev) => {
          if (prev <= 1) {
            if (autoCloseTimerRef.current) {
              clearInterval(autoCloseTimerRef.current);
              autoCloseTimerRef.current = null;
            }
            onClose();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }

    return () => {
      if (autoCloseTimerRef.current) {
        clearInterval(autoCloseTimerRef.current);
        autoCloseTimerRef.current = null;
      }
    };
  }, [captureState, onClose]);

  const handleStart = useCallback(() => {
    startCapture();
  }, [startCapture]);

  const handleClose = useCallback(() => {
    resetCapture();
    onClose();
  }, [resetCapture, onClose]);

  const handleRetry = useCallback(() => {
    resetCapture();
    startCapture();
  }, [resetCapture, startCapture]);

  // Re-enrollment confirmation handlers
  const handleConfirmReenroll = useCallback(() => {
    console.log('[EnrollmentModal] User confirmed re-enrollment');
    setShowReenrollConfirm(false);
    setReenrollConfirmed(true);
  }, []);

  const handleCancelReenroll = useCallback(() => {
    console.log('[EnrollmentModal] User cancelled re-enrollment');
    setShowReenrollConfirm(false);
    onClose();
  }, [onClose]);

  // Don't render if not open
  if (!isOpen) return null;

  const error = cameraError || poseError || enrollmentError;
  const isLoading = poseLoading || (!cameraReady && !cameraError);

  return (
    <div className={clsx('enrollment-modal-overlay', className)} style={styles.overlay}>
      <div style={styles.modal}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>Biometric & Photo Capture</h2>
          <button onClick={handleClose} style={styles.closeButton}>
            <CloseIcon />
          </button>
        </div>

        {/* Content */}
        <div style={styles.content}>
          {/* Left side - Progress */}
          <div style={styles.progressPanel}>
            <CaptureProgress captures={captures} currentStep={currentStep} />
          </div>

          {/* Right side - Camera */}
          <div style={styles.cameraPanel}>
            {/* Camera feed - render first so overlays appear on top */}
            {(captureState === 'capturing' || captureState === 'idle') && (
              <CameraFeed
                videoRef={videoRef}
                canvasRef={canvasRef}
                isActive={captureState === 'capturing'}
                targetPose={currentTarget}
                currentPose={currentPose}
                isCountingDown={isCountingDown}
                countdownSeconds={countdownSeconds}
                isPoseValid={isPoseValid}
                isStabilizing={isStabilizing}
                onCapture={() => {}}
              />
            )}

            {/* Loading state */}
            {isLoading && (
              <div style={styles.loadingOverlay}>
                <div style={styles.spinner} />
                <p>
                  {!cameraReady ? 'Initializing camera...' : 
                   poseLoading ? 'Loading face detection model (10-20s)...' : 
                   'Initializing...'}
                </p>
              </div>
            )}
            
            {/* Model loading indicator (shown even after camera ready) */}
            {cameraReady && poseLoading && !isLoading && (
              <div style={{...styles.loadingOverlay, backgroundColor: 'rgba(0,0,0,0.7)'}}>
                <div style={styles.spinner} />
                <p>Loading face detection model...</p>
                <p style={{fontSize: '12px', opacity: 0.7}}>This may take 10-20 seconds on first load</p>
              </div>
            )}

            {/* Error state */}
            {error && (
              <div style={styles.errorOverlay}>
                <p style={styles.errorText}>{error}</p>
                <button onClick={handleRetry} style={styles.retryButton}>
                  Try Again
                </button>
              </div>
            )}

            {/* Idle state - show start button */}
            {captureState === 'idle' && !error && !isLoading && cameraReady && (
              <div style={styles.startOverlay}>
                <div style={styles.infoBox}>
                  <h3>Face Enrollment</h3>
                  <p>We'll capture 5 photos from different angles:</p>
                  <ul style={styles.angleList}>
                    {CAPTURE_TARGETS.map((target) => (
                      <li key={target.name}>{target.displayName}</li>
                    ))}
                  </ul>
                  <button onClick={handleStart} style={styles.startButton}>
                    Start Capture
                  </button>
                </div>
              </div>
            )}

            {/* Processing state */}
            {captureState === 'processing' && (
              <div style={styles.processingOverlay}>
                <div style={styles.spinner} />
                <p>Processing enrollment...</p>
              </div>
            )}

            {/* Complete state */}
            {captureState === 'complete' && result && (
              <div style={styles.completeOverlay}>
                <div style={styles.successIcon}>
                  <CheckCircleIcon />
                </div>
                <h3 style={styles.successTitle}>Enrollment Complete!</h3>
                <p style={styles.successMessage}>{result.message}</p>
                <button onClick={handleClose} style={styles.doneButton}>
                  Done ({autoCloseCountdown}s)
                </button>
              </div>
            )}

            {/* Error state after processing */}
            {captureState === 'error' && (
              <div style={styles.errorOverlay}>
                <p style={styles.errorText}>{enrollmentError}</p>
                <button onClick={handleRetry} style={styles.retryButton}>
                  Try Again
                </button>
              </div>
            )}

            {/* Re-enrollment confirmation overlay */}
            {showReenrollConfirm && (
              <div style={styles.reenrollOverlay}>
                <div style={styles.reenrollBox}>
                  <WarningIcon />
                  <h3 style={styles.reenrollTitle}>Existing Profile Found</h3>
                  <p style={styles.reenrollText}>
                    {userName ? `${userName} already` : 'This employee already'} has a facial recognition profile
                    {enrollmentStatus === 'enrolled' 
                      ? ' that has been published to devices.' 
                      : ' with pending embeddings.'}
                  </p>
                  <p style={styles.reenrollQuestion}>
                    Would you like to recreate the facial profile for this employee?
                  </p>
                  <p style={styles.reenrollWarning}>
                    This will replace the existing biometric data.
                  </p>
                  <div style={styles.reenrollButtons}>
                    <button onClick={handleCancelReenroll} style={styles.cancelButton}>
                      Cancel
                    </button>
                    <button onClick={handleConfirmReenroll} style={styles.confirmButton}>
                      Yes, Recreate Profile
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer info bar */}
        <div style={styles.footer}>
          <InfoIcon />
          <span>
            Proper face alignment in the image capture box. System will capture automatically.
          </span>
        </div>

        {/* Action buttons */}
        <div style={styles.actions}>
          <button onClick={handleClose} style={styles.backButton}>
            Back
          </button>
          {captureState === 'complete' && (
            <button onClick={handleClose} style={styles.saveButton}>
              Save
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Icons
function CloseIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

function CheckCircleIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <polyline points="9 12 12 15 16 9" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

function WarningIcon() {
  return (
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}

const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modal: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    width: '90%',
    maxWidth: '900px',
    maxHeight: '90vh',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 24px',
    borderBottom: '1px solid #e0e0e0',
  },
  title: {
    margin: 0,
    fontSize: '20px',
    fontWeight: 600,
    color: '#1a1a1a',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    padding: '8px',
    color: '#666',
    borderRadius: '4px',
  },
  content: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  progressPanel: {
    width: '280px',
    borderRight: '1px solid #e0e0e0',
    overflowY: 'auto',
    padding: '16px',
  },
  cameraPanel: {
    flex: 1,
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
    minHeight: '400px',
  },
  loadingOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
    color: '#fff',
    gap: '16px',
    zIndex: 10,
  },
  spinner: {
    width: '40px',
    height: '40px',
    border: '3px solid #333',
    borderTopColor: '#3b82f6',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  errorOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    color: '#fff',
    gap: '16px',
    padding: '24px',
    zIndex: 10,
  },
  errorText: {
    color: '#ff6b6b',
    textAlign: 'center',
    maxWidth: '300px',
  },
  retryButton: {
    padding: '10px 24px',
    backgroundColor: '#3b82f6',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
  },
  startOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    color: '#fff',
    zIndex: 10,
  },
  infoBox: {
    textAlign: 'center',
    padding: '32px',
    maxWidth: '400px',
  },
  angleList: {
    textAlign: 'left',
    margin: '16px auto',
    paddingLeft: '24px',
    lineHeight: 1.8,
  },
  startButton: {
    marginTop: '24px',
    padding: '14px 48px',
    backgroundColor: '#22c55e',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  processingOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    color: '#fff',
    gap: '16px',
    zIndex: 10,
  },
  completeOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 50, 0, 0.95)',
    color: '#fff',
    gap: '16px',
    padding: '24px',
    zIndex: 10,
  },
  successIcon: {
    marginBottom: '8px',
  },
  successTitle: {
    margin: 0,
    fontSize: '24px',
    fontWeight: 600,
    color: '#22c55e',
  },
  successMessage: {
    color: '#ccc',
    textAlign: 'center',
    maxWidth: '300px',
  },
  doneButton: {
    marginTop: '16px',
    padding: '14px 48px',
    backgroundColor: '#22c55e',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    fontWeight: 600,
    cursor: 'pointer',
  },
  footer: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '12px 24px',
    backgroundColor: '#d1fae5',
    color: '#065f46',
    fontSize: '14px',
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    gap: '12px',
    padding: '16px 24px',
    borderTop: '1px solid #e0e0e0',
  },
  backButton: {
    padding: '10px 24px',
    backgroundColor: '#f3f4f6',
    color: '#374151',
    border: '1px solid #d1d5db',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
  },
  saveButton: {
    padding: '10px 24px',
    backgroundColor: '#22c55e',
    color: '#fff',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
  },
  // Re-enrollment confirmation styles
  reenrollOverlay: {
    position: 'absolute',
    inset: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.95)',
    color: '#fff',
    zIndex: 20,
  },
  reenrollBox: {
    textAlign: 'center',
    padding: '32px',
    maxWidth: '420px',
    backgroundColor: '#1f2937',
    borderRadius: '12px',
    border: '1px solid #374151',
  },
  reenrollTitle: {
    margin: '16px 0 8px 0',
    fontSize: '22px',
    fontWeight: 600,
    color: '#f59e0b',
  },
  reenrollText: {
    color: '#d1d5db',
    fontSize: '14px',
    lineHeight: 1.6,
    margin: '8px 0',
  },
  reenrollQuestion: {
    color: '#fff',
    fontSize: '16px',
    fontWeight: 500,
    margin: '16px 0 8px 0',
  },
  reenrollWarning: {
    color: '#ef4444',
    fontSize: '13px',
    fontStyle: 'italic',
    margin: '8px 0 24px 0',
  },
  reenrollButtons: {
    display: 'flex',
    gap: '12px',
    justifyContent: 'center',
  },
  cancelButton: {
    padding: '12px 24px',
    backgroundColor: '#374151',
    color: '#fff',
    border: '1px solid #4b5563',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
  },
  confirmButton: {
    padding: '12px 24px',
    backgroundColor: '#f59e0b',
    color: '#000',
    border: 'none',
    borderRadius: '6px',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
  },
};

// Add keyframe animation via style tag
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = `
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `;
  document.head.appendChild(styleSheet);
}
