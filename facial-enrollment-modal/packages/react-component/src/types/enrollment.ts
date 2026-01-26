/**
 * Enrollment status from the backend/dashboard
 */
export type EnrollmentStatus = 
  | 'enrolled'      // Completed and published to IoT devices
  | 'pending'       // Embeddings generated but not published
  | 'unenrolled';   // No enrollment data

/**
 * Internal capture state during enrollment flow
 */
export type CaptureState = 
  | 'idle'          // Waiting to start
  | 'initializing'  // Loading camera/models
  | 'capturing'     // Actively capturing poses
  | 'processing'    // Sending to backend for embedding
  | 'complete'      // Successfully enrolled
  | 'error';        // Something went wrong

/**
 * The 5 capture poses
 */
export type PoseName = 'front' | 'left' | 'right' | 'up' | 'down';

/**
 * Target pose configuration
 */
export interface PoseTarget {
  name: PoseName;
  displayName: string;
  instruction: string;
  yawRange: [number, number];   // [min, max] degrees
  pitchRange: [number, number]; // [min, max] degrees
  audioFile: string;
}

/**
 * A single face landmark point
 */
export interface FaceLandmark {
  x: number;  // Normalized 0-1
  y: number;  // Normalized 0-1
  z: number;  // Depth (relative)
}

/**
 * Current face pose from TensorFlow.js detection
 */
export interface FacePose {
  detected: boolean;
  yaw: number;      // Left/right rotation (-90 to 90)
  pitch: number;    // Up/down rotation (-90 to 90)
  roll: number;     // Tilt (-90 to 90)
  confidence: number;
  landmarks?: FaceLandmark[];  // 468 landmarks from MediaPipe FaceMesh
}

/**
 * Captured frame data
 */
export interface CapturedFrame {
  pose: PoseName;
  imageData: string;  // Base64 encoded JPEG
  timestamp: number;
}

/**
 * Enrollment result from backend
 */
export interface EnrollmentResult {
  success: boolean;
  userId: string;
  message: string;
  embeddingCount?: number;
  profileImageUrl?: string;
}

/**
 * Props for the main EnrollmentModal component
 */
export interface EnrollmentModalProps {
  /** Whether the modal is open */
  isOpen: boolean;
  
  /** Callback when modal should close */
  onClose: () => void;
  
  /** User ID to enroll */
  userId: string;
  
  /** Current enrollment status from backend */
  enrollmentStatus: EnrollmentStatus;
  
  /** API endpoint base URL (e.g., "/api/enrollment" or "http://localhost:3001/api/enrollment") */
  apiEndpoint: string;
  
  /** Callback when enrollment completes successfully */
  onEnrollmentComplete?: (result: EnrollmentResult) => void;
  
  /** Optional user display name */
  userName?: string;
  
  /** Enable audio guidance (default: true) */
  enableAudio?: boolean;
  
  /** Custom styles */
  className?: string;
}

/**
 * Props for CameraFeed component
 */
export interface CameraFeedProps {
  /** Whether camera should be active */
  isActive: boolean;
  
  /** Current target pose */
  targetPose: PoseTarget | null;
  
  /** Callback when pose is detected */
  onPoseDetected: (pose: FacePose) => void;
  
  /** Callback when frame should be captured */
  onCapture: (imageData: string) => void;
  
  /** Whether currently in countdown */
  isCountingDown: boolean;
  
  /** Countdown seconds remaining */
  countdownSeconds: number;
  
  /** Whether pose matches target */
  isPoseValid: boolean;
}

/**
 * Props for CaptureProgress component
 */
export interface CaptureProgressProps {
  /** Captured frames so far */
  captures: CapturedFrame[];
  
  /** Current capture step (0-4) */
  currentStep: number;
  
  /** Total steps */
  totalSteps: number;
}

/**
 * API request for submitting captures
 */
export interface SubmitCapturesRequest {
  userId: string;
  captures: Array<{
    pose: PoseName;
    imageData: string;
  }>;
}

/**
 * API response from enrollment endpoint
 */
export interface EnrollmentApiResponse {
  success: boolean;
  message: string;
  data?: {
    userId: string;
    embeddingCount: number;
    profileImagePath: string;
    status: EnrollmentStatus;
  };
  error?: string;
}

/**
 * API response from status endpoint
 */
export interface StatusApiResponse {
  userId: string;
  status: EnrollmentStatus;
  enrolledAt?: string;
  imageCount?: number;
  profileImageUrl?: string;
}
