import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  CaptureState,
  CapturedFrame,
  PoseTarget,
  FacePose,
  EnrollmentResult,
  EnrollmentApiResponse,
} from '../types/enrollment';
import {
  CAPTURE_TARGETS,
  COUNTDOWN_SECONDS,
  POSE_HOLD_TIME,
  STABLE_FRAME_THRESHOLD,
  COUNTDOWN_TOLERANCE,
} from '../constants';

export interface UseEnrollmentOptions {
  userId: string;
  apiEndpoint: string;
  onComplete?: (result: EnrollmentResult) => void;
}

export interface UseEnrollmentReturn {
  // State
  captureState: CaptureState;
  currentStep: number;
  captures: CapturedFrame[];
  currentTarget: PoseTarget | null;
  error: string | null;
  result: EnrollmentResult | null;
  
  // Countdown state
  isCountingDown: boolean;
  countdownSeconds: number;
  isPoseValid: boolean;
  
  // Actions
  startCapture: () => void;
  resetCapture: () => void;
  processPose: (pose: FacePose) => void;
  captureFrame: (imageData: string) => void;
  submitCaptures: () => Promise<void>;
}

/**
 * State machine hook for managing the enrollment capture flow
 */
export function useEnrollment(options: UseEnrollmentOptions): UseEnrollmentReturn {
  const { userId, apiEndpoint, onComplete } = options;

  // Core state
  const [captureState, setCaptureState] = useState<CaptureState>('idle');
  const [currentStep, setCurrentStep] = useState(0);
  const [captures, setCaptures] = useState<CapturedFrame[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EnrollmentResult | null>(null);

  // Countdown state
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdownSeconds, setCountdownSeconds] = useState(COUNTDOWN_SECONDS);
  const [isPoseValid, setIsPoseValid] = useState(false);

  // Refs for timing
  const stableFrameCountRef = useRef(0);
  const poseHeldSinceRef = useRef(0);
  const countdownStartRef = useRef(0);
  const countdownIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lockedPoseRef = useRef<{ yaw: number; pitch: number } | null>(null);
  const shouldCaptureRef = useRef(false);

  // Current target pose
  const currentTarget = currentStep < CAPTURE_TARGETS.length ? CAPTURE_TARGETS[currentStep] : null;

  const startCapture = useCallback(() => {
    console.log('[Enrollment] 🚀 Starting capture sequence');
    setCaptureState('capturing');
    setCurrentStep(0);
    setCaptures([]);
    setError(null);
    setResult(null);
    setIsCountingDown(false);
    setCountdownSeconds(COUNTDOWN_SECONDS);
    setIsPoseValid(false);
    stableFrameCountRef.current = 0;
    poseHeldSinceRef.current = 0;
    lockedPoseRef.current = null;
    shouldCaptureRef.current = false;
  }, []);

  const resetCapture = useCallback(() => {
    setCaptureState('idle');
    setCurrentStep(0);
    setCaptures([]);
    setError(null);
    setResult(null);
    setIsCountingDown(false);
    setCountdownSeconds(COUNTDOWN_SECONDS);
    setIsPoseValid(false);
    
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    
    stableFrameCountRef.current = 0;
    poseHeldSinceRef.current = 0;
    lockedPoseRef.current = null;
    shouldCaptureRef.current = false;
  }, []);

  const checkPoseMatch = useCallback((pose: FacePose, target: PoseTarget): boolean => {
    if (!pose.detected) return false;

    const { yaw, pitch } = pose;
    const [yawMin, yawMax] = target.yawRange;
    const [pitchMin, pitchMax] = target.pitchRange;

    return yaw >= yawMin && yaw <= yawMax && pitch >= pitchMin && pitch <= pitchMax;
  }, []);

  const checkLockedPose = useCallback((pose: FacePose): boolean => {
    if (!lockedPoseRef.current || !pose.detected) return false;

    const yawDiff = Math.abs(pose.yaw - lockedPoseRef.current.yaw);
    const pitchDiff = Math.abs(pose.pitch - lockedPoseRef.current.pitch);

    return yawDiff < COUNTDOWN_TOLERANCE && pitchDiff < COUNTDOWN_TOLERANCE;
  }, []);

  const startCountdown = useCallback(() => {
    console.log('[Enrollment] ⏱️ Starting countdown...');
    setIsCountingDown(true);
    setCountdownSeconds(COUNTDOWN_SECONDS);
    countdownStartRef.current = Date.now();

    // Clear any existing interval
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
    }

    countdownIntervalRef.current = setInterval(() => {
      const elapsed = (Date.now() - countdownStartRef.current) / 1000;
      const remaining = Math.max(0, COUNTDOWN_SECONDS - elapsed);
      setCountdownSeconds(Math.ceil(remaining));

      if (remaining <= 0) {
        // Trigger capture
        console.log('[Enrollment] ⏱️ Countdown complete - triggering capture');
        shouldCaptureRef.current = true;
        
        if (countdownIntervalRef.current) {
          clearInterval(countdownIntervalRef.current);
          countdownIntervalRef.current = null;
        }
      }
    }, 100);
  }, []);

  const cancelCountdown = useCallback(() => {
    setIsCountingDown(false);
    setCountdownSeconds(COUNTDOWN_SECONDS);
    lockedPoseRef.current = null;
    stableFrameCountRef.current = 0;
    poseHeldSinceRef.current = 0;

    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
  }, []);

  const processPose = useCallback((pose: FacePose) => {
    if (captureState !== 'capturing' || !currentTarget) return;

    // Check if we should capture (countdown finished)
    if (shouldCaptureRef.current) {
      return; // Wait for captureFrame to be called
    }

    // If counting down, check if still in locked position
    if (isCountingDown) {
      if (!checkLockedPose(pose)) {
        cancelCountdown();
      }
      return;
    }

    // Check if pose matches target
    const poseMatches = checkPoseMatch(pose, currentTarget);
    setIsPoseValid(poseMatches);

    if (poseMatches) {
      stableFrameCountRef.current++;

      // Need stable frames before starting countdown
      if (stableFrameCountRef.current >= STABLE_FRAME_THRESHOLD) {
        if (poseHeldSinceRef.current === 0) {
          poseHeldSinceRef.current = Date.now();
        }

        const holdDuration = (Date.now() - poseHeldSinceRef.current) / 1000;

        if (holdDuration >= POSE_HOLD_TIME) {
          // Lock the pose and start countdown
          lockedPoseRef.current = { yaw: pose.yaw, pitch: pose.pitch };
          startCountdown();
        }
      }
    } else {
      // Reset stability tracking
      stableFrameCountRef.current = 0;
      poseHeldSinceRef.current = 0;
    }
  }, [captureState, currentTarget, isCountingDown, checkPoseMatch, checkLockedPose, startCountdown, cancelCountdown]);

  const captureFrame = useCallback((imageData: string) => {
    if (!shouldCaptureRef.current || !currentTarget) {
      console.log('[Enrollment] captureFrame called but not ready:', { shouldCapture: shouldCaptureRef.current, hasTarget: !!currentTarget });
      return;
    }

    shouldCaptureRef.current = false;

    console.log(`[Enrollment] 📸 CAPTURED: ${currentTarget.name} (step ${currentStep + 1}/${CAPTURE_TARGETS.length})`);
    console.log(`[Enrollment] Image data length: ${imageData.length} bytes`);

    const newCapture: CapturedFrame = {
      pose: currentTarget.name,
      imageData,
      timestamp: Date.now(),
    };

    const newCaptures = [...captures, newCapture];
    setCaptures(newCaptures);

    // Move to next step
    const nextStep = currentStep + 1;
    setCurrentStep(nextStep);
    console.log(`[Enrollment] Moving to step ${nextStep + 1}`);

    // Reset state for next capture
    setIsCountingDown(false);
    setCountdownSeconds(COUNTDOWN_SECONDS);
    setIsPoseValid(false);
    stableFrameCountRef.current = 0;
    poseHeldSinceRef.current = 0;
    lockedPoseRef.current = null;

    // Check if all captures complete
    if (nextStep >= CAPTURE_TARGETS.length) {
      console.log('[Enrollment] ✅ All captures complete! Processing...');
      setCaptureState('processing');
    }
  }, [currentTarget, captures, currentStep]);

  const submitCaptures = useCallback(async () => {
    if (captures.length < CAPTURE_TARGETS.length) {
      setError('Not all poses captured');
      return;
    }

    setCaptureState('processing');

    try {
      const response = await fetch(`${apiEndpoint}/capture`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId,
          captures: captures.map((c) => ({
            pose: c.pose,
            imageData: c.imageData,
          })),
        }),
      });

      const data: EnrollmentApiResponse = await response.json();

      if (data.success) {
        const enrollmentResult: EnrollmentResult = {
          success: true,
          userId,
          message: data.message,
          embeddingCount: data.data?.embeddingCount,
          profileImageUrl: data.data?.profileImagePath,
        };
        
        setResult(enrollmentResult);
        setCaptureState('complete');
        onComplete?.(enrollmentResult);
      } else {
        throw new Error(data.error || data.message || 'Enrollment failed');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to submit enrollment';
      setError(message);
      setCaptureState('error');
    }
  }, [captures, apiEndpoint, userId, onComplete]);

  // Auto-submit when all captures are done
  useEffect(() => {
    if (captureState === 'processing' && captures.length === CAPTURE_TARGETS.length) {
      submitCaptures();
    }
  }, [captureState, captures.length, submitCaptures]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (countdownIntervalRef.current) {
        clearInterval(countdownIntervalRef.current);
      }
    };
  }, []);

  return {
    captureState,
    currentStep,
    captures,
    currentTarget,
    error,
    result,
    isCountingDown,
    countdownSeconds,
    isPoseValid,
    startCapture,
    resetCapture,
    processPose,
    captureFrame,
    submitCaptures,
  };
}
