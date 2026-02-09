import React, { useEffect, useRef, useCallback } from 'react';
import { clsx } from 'clsx';
import type { PoseTarget, FacePose, FaceLandmark } from '../types/enrollment';

// ============================================================================
// FACE MESH CONNECTIONS - subset for visualization
// Based on MediaPipe FaceMesh topology
// ============================================================================

// Face oval contour (outer edge of face)
const FACE_OVAL = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109, 10
];

// Left eye contour
const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33];

// Right eye contour
const RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263];

// Left eyebrow
const LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];

// Right eyebrow
const RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];

// Lips outer contour
const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61];

// Lips inner contour
const LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78];

// Nose bridge and tip
const NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2];

// Left iris (if available)
const LEFT_IRIS = [468, 469, 470, 471, 472];

// Right iris (if available)  
const RIGHT_IRIS = [473, 474, 475, 476, 477];

// Additional tessellation connections for mesh effect
// These create the "grid" appearance over the face
const TESSELLATION_CONNECTIONS = [
  // Forehead horizontal lines
  [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251],
  [71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301],
  
  // Cheek connections
  [127, 34, 143, 111, 117, 118, 119, 120, 121, 128],
  [356, 264, 372, 340, 346, 347, 348, 349, 350, 357],
  
  // Nose to eye connections
  [168, 193, 122, 245],
  [168, 417, 351, 465],
  
  // Under eye to cheek
  [226, 31, 228, 229, 230, 231],
  [446, 261, 448, 449, 450, 451],
  
  // Chin connections
  [176, 148, 152, 377, 400, 378, 379],
  [83, 18, 313, 406, 335, 273],
  
  // Vertical face lines
  [10, 151, 9, 8, 168, 6, 197, 195, 5],
  [152, 175, 199, 200, 18, 17],
];

/**
 * Draw connected path from landmark indices
 */
function drawLandmarkPath(
  ctx: CanvasRenderingContext2D,
  landmarks: FaceLandmark[],
  indices: number[],
  w: number,
  h: number,
  color: string,
  lineWidth: number = 1,
  closed: boolean = false
): void {
  if (indices.length < 2) return;
  
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  
  const first = landmarks[indices[0]];
  if (!first) return;
  
  ctx.moveTo(first.x * w, first.y * h);
  
  for (let i = 1; i < indices.length; i++) {
    const idx = indices[i];
    if (idx >= landmarks.length) continue;
    const pt = landmarks[idx];
    ctx.lineTo(pt.x * w, pt.y * h);
  }
  
  if (closed) {
    ctx.closePath();
  }
  
  ctx.stroke();
}

/**
 * Draw face mesh on canvas - mimics Python MediaPipe visualization
 */
function drawFaceMesh(
  ctx: CanvasRenderingContext2D,
  landmarks: FaceLandmark[],
  w: number,
  h: number
): void {
  if (!landmarks || landmarks.length < 468) return;
  
  // Draw tessellation lines first (background mesh) - thin gray
  for (const connection of TESSELLATION_CONNECTIONS) {
    drawLandmarkPath(ctx, landmarks, connection, w, h, 'rgba(128, 128, 128, 0.4)', 1);
  }
  
  // Draw face oval contour (gray)
  drawLandmarkPath(ctx, landmarks, FACE_OVAL, w, h, 'rgba(128, 128, 128, 0.7)', 1, true);
  
  // Draw eyes - contours (cyan/turquoise like Python)
  drawLandmarkPath(ctx, landmarks, LEFT_EYE, w, h, 'rgba(80, 110, 10, 0.9)', 1, true);
  drawLandmarkPath(ctx, landmarks, RIGHT_EYE, w, h, 'rgba(80, 110, 10, 0.9)', 1, true);
  
  // Draw eyebrows (light gray)
  drawLandmarkPath(ctx, landmarks, LEFT_EYEBROW, w, h, 'rgba(160, 160, 160, 0.7)', 1);
  drawLandmarkPath(ctx, landmarks, RIGHT_EYEBROW, w, h, 'rgba(160, 160, 160, 0.7)', 1);
  
  // Draw lips (pink/red tint)
  drawLandmarkPath(ctx, landmarks, LIPS_OUTER, w, h, 'rgba(80, 110, 10, 0.8)', 1, true);
  drawLandmarkPath(ctx, landmarks, LIPS_INNER, w, h, 'rgba(80, 110, 10, 0.6)', 1, true);
  
  // Draw nose (light olive like Python contours)
  drawLandmarkPath(ctx, landmarks, NOSE, w, h, 'rgba(80, 110, 10, 0.7)', 1);
  
  // Draw irises if available (yellow like Python)
  if (landmarks.length > 477) {
    drawLandmarkPath(ctx, landmarks, LEFT_IRIS, w, h, 'rgba(48, 255, 255, 0.9)', 2, true);
    drawLandmarkPath(ctx, landmarks, RIGHT_IRIS, w, h, 'rgba(48, 255, 255, 0.9)', 2, true);
  }
  
  // Draw key landmark points for pose estimation
  const keyPoints = [1, 152, 33, 263, 61, 291]; // Nose tip, chin, eye corners, mouth corners
  ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
  for (const idx of keyPoints) {
    if (idx < landmarks.length) {
      const pt = landmarks[idx];
      ctx.beginPath();
      ctx.arc(pt.x * w, pt.y * h, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

export interface CameraFeedProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isActive: boolean;
  targetPose: PoseTarget | null;
  currentPose: FacePose;
  isCountingDown: boolean;
  countdownSeconds: number;
  isPoseValid: boolean;
  isStabilizing: boolean;
  onCapture: () => void;
  className?: string;
}

/**
 * Camera feed component with pose overlay and countdown UI
 */
export function CameraFeed({
  videoRef,
  canvasRef,
  isActive,
  targetPose,
  currentPose,
  isCountingDown,
  countdownSeconds,
  isPoseValid,
  isStabilizing,
  onCapture,
  className,
}: CameraFeedProps) {
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Draw the overlay (face guide circle, countdown, etc.)
  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Match canvas size to video
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const w = canvas.width;
    const h = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, w, h);

    // Draw face mesh overlay if landmarks are available
    if (currentPose.detected && currentPose.landmarks) {
      drawFaceMesh(ctx, currentPose.landmarks, w, h);
    }

    // Draw face guide oval (dashed circle like in the mockup)
    const centerX = w / 2;
    const centerY = h / 2;
    const radiusX = Math.min(w, h) * 0.15;  // Reduced from 0.35 to ~25% size
    const radiusY = radiusX * 1.3; // Slightly oval for face shape

    // Determine border color based on state
    let borderColor = 'rgba(255, 165, 0, 0.8)'; // Orange - default
    if (!currentPose.detected) {
      borderColor = 'rgba(255, 0, 0, 0.8)'; // Red - no face
    } else if (isCountingDown) {
      borderColor = 'rgba(0, 255, 0, 0.9)'; // Green - counting down
    } else if (isPoseValid) {
      borderColor = 'rgba(0, 255, 0, 0.8)'; // Green - pose valid
    } else if (isStabilizing) {
      borderColor = 'rgba(0, 200, 200, 0.8)'; // Cyan - stabilizing
    }

    // Draw dashed oval guide
    ctx.save();
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3;
    ctx.setLineDash([15, 10]);
    
    ctx.beginPath();
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // Draw countdown if active
    if (isCountingDown && countdownSeconds > 0) {
      // Progress ring
      const progress = (3 - countdownSeconds + 1) / 3; // Assuming 3 second countdown
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + (Math.PI * 2 * progress);

      ctx.save();
      ctx.strokeStyle = 'rgba(0, 255, 0, 0.9)';
      ctx.lineWidth = 8;
      ctx.lineCap = 'round';
      ctx.setLineDash([]);
      
      ctx.beginPath();
      ctx.arc(centerX, centerY, radiusX * 0.8, startAngle, endAngle);
      ctx.stroke();
      ctx.restore();

      // Countdown number
      ctx.save();
      // Counter the CSS scaleX(-1) mirror so numbers read correctly
      ctx.scale(-1, 1);
      ctx.translate(-w, 0);

      ctx.fillStyle = 'rgba(0, 255, 0, 1)';
      ctx.font = 'bold 72px Arial, sans-serif';  // Smaller font to fit
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      // Text shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
      ctx.shadowBlur = 10;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 2;
      
      ctx.fillText(String(countdownSeconds), centerX, centerY);
      ctx.restore();
    }

    // Draw status text at bottom
    let statusText = '';
    let statusColor = 'white';

    if (!currentPose.detected) {
      statusText = 'No face detected';
      statusColor = '#ff4444';
    } else if (isCountingDown) {
      statusText = 'Hold still!';
      statusColor = '#44ff44';
    } else if (isPoseValid && isStabilizing) {
      statusText = 'Hold steady...';
      statusColor = '#44dddd';
    } else if (isPoseValid) {
      statusText = 'HOLD STILL...';
      statusColor = '#44ff44';
    } else if (targetPose) {
      statusText = getGuidanceText(currentPose, targetPose);
      statusColor = '#ffaa44';
    }

    if (statusText) {
      ctx.save();
      // Counter the CSS scaleX(-1) mirror so text reads correctly
      ctx.scale(-1, 1);
      ctx.translate(-w, 0);

      ctx.fillStyle = statusColor;
      ctx.font = 'bold 24px Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      
      // Text shadow
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
      ctx.shadowBlur = 4;
      ctx.shadowOffsetX = 1;
      ctx.shadowOffsetY = 1;
      
      ctx.fillText(statusText, centerX, h - 30);
      ctx.restore();
    }
  }, [videoRef, currentPose, targetPose, isCountingDown, countdownSeconds, isPoseValid, isStabilizing]);

  // Animation loop for overlay
  useEffect(() => {
    if (!isActive) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const animate = () => {
      drawOverlay();
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
    };
  }, [isActive, drawOverlay]);

  return (
    <div className={clsx('camera-feed-container', className)} style={styles.container}>
      {/* Video element (mirrored for selfie view) */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={styles.video}
      />
      
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} style={styles.hiddenCanvas} />
      
      {/* Overlay canvas for UI elements */}
      <canvas ref={overlayCanvasRef} style={styles.overlay} />
      
      {/* Instruction text at top */}
      {targetPose && (
        <div style={styles.instructionBar}>
          <span style={styles.stepIndicator}>
            {targetPose.displayName}
          </span>
          <span style={styles.instruction}>
            {targetPose.instruction}
          </span>
        </div>
      )}
    </div>
  );
}

/**
 * Get guidance text based on pose deviation
 */
function getGuidanceText(pose: FacePose, target: PoseTarget): string {
  const { yaw, pitch } = pose;
  const [yawMin, yawMax] = target.yawRange;
  const [pitchMin, pitchMax] = target.pitchRange;

  const yawLow = yaw < yawMin;
  const yawHigh = yaw > yawMax;
  const pitchLow = pitch < pitchMin;
  const pitchHigh = pitch > pitchMax;

  const hints: string[] = [];

  switch (target.name) {
    case 'left':
      if (yawHigh) hints.push('Too far! Come back RIGHT');
      else if (yawLow) hints.push('Turn more LEFT');
      break;
    case 'right':
      if (yawLow) hints.push('Too far! Come back LEFT');
      else if (yawHigh) hints.push('Turn more RIGHT');
      break;
    case 'up':
      if (pitchLow) hints.push('Too far! Lower chin');
      else if (pitchHigh) hints.push('Tilt chin UP more');
      break;
    case 'down':
      if (pitchHigh) hints.push('Too far! Raise chin');
      else if (pitchLow) hints.push('Tilt chin DOWN more');
      break;
    default:
      if (yawLow) hints.push('Turn LEFT');
      else if (yawHigh) hints.push('Turn RIGHT');
      if (pitchLow) hints.push('Look DOWN');
      else if (pitchHigh) hints.push('Look UP');
  }

  return hints.join(' | ') || target.instruction;
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    position: 'relative',
    width: '100%',
    maxWidth: '640px',
    aspectRatio: '4/3',
    backgroundColor: '#000',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  video: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    objectFit: 'cover',
    transform: 'scaleX(-1)', // Mirror for natural selfie view
  },
  hiddenCanvas: {
    display: 'none',
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    transform: 'scaleX(-1)', // Mirror to match video
  },
  instructionBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    padding: '12px 16px',
    background: 'linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%)',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  stepIndicator: {
    color: '#fff',
    fontSize: '14px',
    fontWeight: 600,
    opacity: 0.9,
  },
  instruction: {
    color: '#00ffff',
    fontSize: '18px',
    fontWeight: 700,
  },
};
