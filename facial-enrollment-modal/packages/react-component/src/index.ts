// Components
export { EnrollmentModal, CameraFeed, CaptureProgress } from './components';

// Hooks
export { useCamera, useFacePose, useEnrollment } from './hooks';
export type {
  UseCameraOptions,
  UseCameraReturn,
  UseFacePoseOptions,
  UseFacePoseReturn,
  UseEnrollmentOptions,
  UseEnrollmentReturn,
} from './hooks';

// Services
export { getAudioService, AudioService } from './services/audioService';
export { createApiClient, EnrollmentApiClient } from './services/apiClient';

// Types
export type {
  EnrollmentStatus,
  CaptureState,
  PoseName,
  PoseTarget,
  FacePose,
  CapturedFrame,
  EnrollmentResult,
  EnrollmentModalProps,
  CameraFeedProps,
  CaptureProgressProps,
  SubmitCapturesRequest,
  EnrollmentApiResponse,
  StatusApiResponse,
} from './types/enrollment';

// Constants
export {
  CAPTURE_TARGETS,
  SECONDARY_AUDIO,
  COUNTDOWN_SECONDS,
  POSE_HOLD_TIME,
  STABLE_FRAME_THRESHOLD,
  COUNTDOWN_TOLERANCE,
  ADVISORY_AUDIO_INTERVAL,
  ADVISORY_AUDIO_MAX,
  AUTO_CLOSE_DELAY,
  getPoseTarget,
} from './constants';
