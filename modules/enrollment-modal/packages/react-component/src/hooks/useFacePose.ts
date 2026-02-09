import { useRef, useState, useCallback, useEffect } from 'react';
import type { FacePose, PoseTarget } from '../types/enrollment';

// TensorFlow.js imports - loaded dynamically to reduce bundle size
let tf: typeof import('@tensorflow/tfjs-core') | null = null;
let faceLandmarksDetection: typeof import('@tensorflow-models/face-landmarks-detection') | null = null;

export interface UseFacePoseOptions {
  /** Detection interval in ms (default: 66 = ~15fps) */
  detectionInterval?: number;
  /** Minimum detection confidence (0-1) */
  minConfidence?: number;
}

export interface UseFacePoseReturn {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  currentPose: FacePose;
  initialize: () => Promise<void>;
  startDetection: (video: HTMLVideoElement) => void;
  stopDetection: () => void;
  checkPoseMatch: (target: PoseTarget) => boolean;
}

const DEFAULT_POSE: FacePose = {
  detected: false,
  yaw: 0,
  pitch: 0,
  roll: 0,
  confidence: 0,
};

// MediaPipe FaceMesh landmark indices for key facial points
const LANDMARK_INDICES = {
  noseTip: 1,
  chin: 152,
  forehead: 10,
  leftEyeOuter: 33,
  rightEyeOuter: 263,
  leftEyeInner: 133,
  rightEyeInner: 362,
};

/**
 * Calculate head pose from MediaPipe face landmarks
 * Uses the 3D landmark positions (x, y, z) provided by FaceMesh
 * 
 * MediaPipe provides Z coordinates representing depth - we use these
 * to compute face orientation without needing full PnP solving.
 * 
 * Convention:
 * - yaw: 0 = centered, positive = looking left, negative = looking right
 * - pitch: 0 = straight, positive = looking up, negative = looking down
 * - roll: 0 = level, positive = tilting right
 */
function calculateHeadPose(
  landmarks: Array<{ x: number; y: number; z: number }>,
  videoWidth: number,
  videoHeight: number
): { yaw: number; pitch: number; roll: number } {
  
  // Get key landmarks
  const noseTip = landmarks[LANDMARK_INDICES.noseTip];
  const chin = landmarks[LANDMARK_INDICES.chin];
  const forehead = landmarks[LANDMARK_INDICES.forehead];
  const leftEye = landmarks[LANDMARK_INDICES.leftEyeOuter];
  const rightEye = landmarks[LANDMARK_INDICES.rightEyeOuter];
  
  // ===== ROLL (head tilt) =====
  // Angle of the eye line from horizontal
  const eyeDx = rightEye.x - leftEye.x;
  const eyeDy = rightEye.y - leftEye.y;
  const roll = Math.atan2(eyeDy, eyeDx) * (180 / Math.PI);
  
  // ===== Compute face scale for normalization =====
  // Use eye width as reference to normalize Z values (accounts for distance from camera)
  const eyeWidth = Math.abs(eyeDx);
  const faceHeight = Math.abs(chin.y - forehead.y);
  const faceScale = Math.max(eyeWidth, faceHeight, 0.001);
  
  // ===== YAW (left/right rotation) =====
  // Use the Z-depth difference between left and right eyes
  // MediaPipe Z: more negative = closer to camera
  // When looking LEFT (from user's perspective): right eye comes forward (more negative Z)
  // When looking RIGHT: left eye comes forward (more negative Z)
  const leftZ = leftEye.z;
  const rightZ = rightEye.z;
  const zDiffLR = (rightZ - leftZ) / faceScale;  // FIXED: inverted for correct left/right
  
  // Also use nose position relative to eye center
  const eyeCenterX = (leftEye.x + rightEye.x) / 2;
  const noseOffsetX = (eyeCenterX - noseTip.x) / faceScale;  // FIXED: inverted
  
  // Z-based method (primary) - reduced by 50% for better calibration
  const yawFromZ = zDiffLR * 300;
  // X-based method (secondary) - nose offset indicates rotation direction
  const yawFromX = noseOffsetX * 30;
  
  // Weighted combination
  let yaw = yawFromZ * 0.6 + yawFromX * 0.4;
  
  // ===== PITCH (up/down rotation) =====
  // Use Z-depth difference between forehead and chin
  // When looking UP: forehead goes back (less negative Z), chin comes forward (more negative Z)
  //   So foreheadZ > chinZ, meaning (foreheadZ - chinZ) is positive
  // When looking DOWN: forehead comes forward (more negative Z), chin goes back
  //   So foreheadZ < chinZ, meaning (foreheadZ - chinZ) is negative
  const foreheadZ = forehead.z;
  const chinZ = chin.z;
  const zDiffUD = (foreheadZ - chinZ) / faceScale;
  
  // Scale to degrees - calibrated for accurate readings
  let pitch = zDiffUD * 80;
  
  // Clamp to reasonable range
  yaw = Math.max(-90, Math.min(90, yaw));
  pitch = Math.max(-90, Math.min(90, pitch));
  
  // Debug logging
  if (Math.random() < 0.05) {
    console.log(`[Pose] zDiffLR=${zDiffLR.toFixed(4)} zDiffUD=${zDiffUD.toFixed(4)} faceScale=${faceScale.toFixed(3)} | YAW=${yaw.toFixed(1)}° PITCH=${pitch.toFixed(1)}° ROLL=${roll.toFixed(1)}°`);
  }
  
  return { yaw, pitch, roll };
}

/**
 * Hook for face pose detection using TensorFlow.js
 */
export function useFacePose(options: UseFacePoseOptions = {}): UseFacePoseReturn {
  const { detectionInterval = 66, minConfidence = 0.5 } = options;

  const [isLoading, setIsLoading] = useState(false);
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPose, setCurrentPose] = useState<FacePose>(DEFAULT_POSE);

  const detectorRef = useRef<any>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastDetectionRef = useRef<number>(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const initialize = useCallback(async () => {
    if (isReady || isLoading) return;

    setIsLoading(true);
    setError(null);

    // Add timeout for initialization
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('Model loading timeout (30s)')), 30000);
    });

    try {
      console.log('[FacePose] Starting initialization...');
      
      await Promise.race([
        (async () => {
          // Dynamically import TensorFlow.js
          if (!tf) {
            console.log('[FacePose] Loading TensorFlow.js core...');
            tf = await import('@tensorflow/tfjs-core');
            console.log('[FacePose] Loading WebGL backend...');
            await import('@tensorflow/tfjs-backend-webgl');
            console.log('[FacePose] Setting backend to webgl...');
            await tf.setBackend('webgl');
            await tf.ready();
            console.log('[FacePose] ✅ TensorFlow.js ready, backend:', tf.getBackend());
          }

          // Import face landmarks detection
          if (!faceLandmarksDetection) {
            console.log('[FacePose] Loading face-landmarks-detection package...');
            faceLandmarksDetection = await import('@tensorflow-models/face-landmarks-detection');
            console.log('[FacePose] ✅ Package loaded');
          }

          // Create detector with MediaPipe FaceMesh model
          console.log('[FacePose] Creating face detector (this may take 10-20 seconds)...');
          const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
          
          // Try MediaPipe runtime first (more reliable), fallback to tfjs
          let detector = null;
          try {
            console.log('[FacePose] Trying MediaPipe runtime...');
            const mediapipeConfig: any = {
              runtime: 'mediapipe',
              solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
              refineLandmarks: true,
              maxFaces: 1,
            };
            detector = await faceLandmarksDetection.createDetector(model, mediapipeConfig);
            console.log('[FacePose] ✅ MediaPipe runtime loaded');
          } catch (mpError) {
            console.warn('[FacePose] MediaPipe runtime failed, trying tfjs...', mpError);
            const tfjsConfig: any = {
              runtime: 'tfjs',
              refineLandmarks: true,
              maxFaces: 1,
            };
            detector = await faceLandmarksDetection.createDetector(model, tfjsConfig);
            console.log('[FacePose] ✅ TFJS runtime loaded');
          }
          
          detectorRef.current = detector;
          
          console.log('[FacePose] ✅ Face detector created successfully!');
          setIsReady(true);
        })(),
        timeoutPromise
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to initialize face detection';
      setError(message);
      console.error('[FacePose] ❌ Initialization error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [isReady, isLoading]);

  const detectFace = useCallback(async () => {
    const video = videoRef.current;
    const detector = detectorRef.current;

    if (!video) {
      return;
    }
    
    if (!detector) {
      // Log occasionally
      if (Math.random() < 0.01) console.log('[FacePose] Waiting for detector...');
      return;
    }
    
    if (video.readyState < 2) {
      // Log occasionally
      if (Math.random() < 0.01) console.log('[FacePose] Video not ready, readyState:', video.readyState);
      return;
    }
    
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      if (Math.random() < 0.01) console.log('[FacePose] Video dimensions not ready');
      return;
    }

    // Throttle detection
    const now = performance.now();
    if (now - lastDetectionRef.current < detectionInterval) {
      return;
    }
    lastDetectionRef.current = now;
    
    // Log first detection attempt
    if (lastDetectionRef.current === now) {
      console.log('[FacePose] 🔍 Running face detection...');
    }

    try {
      // Create canvas if needed
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
      }
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      // Draw video frame to canvas
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        console.error('[FacePose] Cannot get canvas context');
        return;
      }
      ctx.drawImage(video, 0, 0);
      
      // Try detection on canvas (more reliable than video element)
      let faces;
      try {
        faces = await detector.estimateFaces(canvas);
      } catch (detectError) {
        console.error('[FacePose] estimateFaces error:', detectError);
        return;
      }
      
      // Log detection results occasionally
      if (Math.random() < 0.05) {
        console.log(`[FacePose] Detection result: ${faces?.length || 0} face(s) found`);
      }

      if (faces && faces.length > 0 && faces[0].keypoints) {
        const face = faces[0];
        const landmarks = face.keypoints;
        
        // Calculate head pose
        const { yaw, pitch, roll } = calculateHeadPose(
          landmarks,
          video.videoWidth,
          video.videoHeight
        );

        // Log occasionally (every 30 frames ~ 2 seconds)
        if (Math.random() < 0.033) {
          console.log(`[FacePose] 👤 Face detected - Yaw: ${yaw.toFixed(1)}° Pitch: ${pitch.toFixed(1)}°`);
        }

        // Include landmarks in pose for mesh visualization
        setCurrentPose({
          detected: true,
          yaw,
          pitch,
          roll,
          confidence: face.box ? 1.0 : minConfidence,
          landmarks: landmarks.map((kp: any) => ({
            x: kp.x / video.videoWidth,
            y: kp.y / video.videoHeight,
            z: kp.z || 0,
          })),
        });
      } else {
        setCurrentPose(DEFAULT_POSE);
      }
    } catch (err) {
      console.error('[FacePose] Detection error:', err);
      setCurrentPose(DEFAULT_POSE);
    }
  }, [detectionInterval, minConfidence]);

  const detectionLoop = useCallback(() => {
    detectFace();
    animationFrameRef.current = requestAnimationFrame(detectionLoop);
  }, [detectFace]);

  const startDetection = useCallback((video: HTMLVideoElement) => {
    if (!isReady) {
      console.warn('[FacePose] ⚠️ Cannot start detection - model not ready');
      return;
    }

    console.log('[FacePose] 🎬 Starting face detection loop');
    console.log(`[FacePose] Video dimensions: ${video.videoWidth}x${video.videoHeight}, readyState: ${video.readyState}`);
    videoRef.current = video;
    
    // Wait for video to be ready if needed
    if (video.readyState < 2 || video.videoWidth === 0) {
      console.log('[FacePose] Video not fully ready, waiting...');
      const checkReady = () => {
        if (video.readyState >= 2 && video.videoWidth > 0) {
          console.log(`[FacePose] Video now ready: ${video.videoWidth}x${video.videoHeight}`);
          if (animationFrameRef.current === null) {
            detectionLoop();
          }
        } else {
          setTimeout(checkReady, 100);
        }
      };
      checkReady();
      return;
    }
    
    // Start detection loop
    if (animationFrameRef.current === null) {
      detectionLoop();
    }
  }, [isReady, detectionLoop]);

  const stopDetection = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    videoRef.current = null;
    setCurrentPose(DEFAULT_POSE);
  }, []);

  const checkPoseMatch = useCallback((target: PoseTarget): boolean => {
    if (!currentPose.detected) return false;

    const { yaw, pitch } = currentPose;
    const [yawMin, yawMax] = target.yawRange;
    const [pitchMin, pitchMax] = target.pitchRange;

    const yawOk = yaw >= yawMin && yaw <= yawMax;
    const pitchOk = pitch >= pitchMin && pitch <= pitchMax;

    return yawOk && pitchOk;
  }, [currentPose]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopDetection();
    };
  }, [stopDetection]);

  return {
    isLoading,
    isReady,
    error,
    currentPose,
    initialize,
    startDetection,
    stopDetection,
    checkPoseMatch,
  };
}
