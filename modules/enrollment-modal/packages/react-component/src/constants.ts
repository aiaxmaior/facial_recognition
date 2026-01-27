import type { PoseTarget, PoseName } from './types/enrollment';

/**
 * Capture targets - matching the Python implementation
 * NORMALIZED: pitch 0 = straight, positive = up, negative = down
 * yaw 0 = centered, positive = left, negative = right
 */
export const CAPTURE_TARGETS: PoseTarget[] = [
  {
    name: 'front',
    displayName: 'Front View',
    instruction: 'Look straight at the camera',
    yawRange: [-25, 25],
    pitchRange: [-20, 20],
    audioFile: 'look_forward.mp3',
  },
  {
    name: 'left',
    displayName: 'Left View',
    instruction: 'Turn your head LEFT',
    yawRange: [10, 50],
    pitchRange: [-30, 30],
    audioFile: 'turn_left.mp3',
  },
  {
    name: 'right',
    displayName: 'Right View',
    instruction: 'Turn your head RIGHT',
    yawRange: [-50, -10],
    pitchRange: [-30, 30],
    audioFile: 'turn_right.mp3',
  },
  {
    name: 'up',
    displayName: 'Up View',
    instruction: 'Tilt your chin UP slightly',
    yawRange: [-30, 30],
    pitchRange: [15, 50],  // INVERTED: positive pitch = looking up
    audioFile: 'look_up.mp3',
  },
  {
    name: 'down',
    displayName: 'Down View',
    instruction: 'Tilt your chin DOWN slightly',
    yawRange: [-40, 40],
    pitchRange: [-50, -15],  // INVERTED: negative pitch = looking down
    audioFile: 'look_down.mp3',
  },
];

/**
 * Secondary guidance audio for re-orientation
 */
export const SECONDARY_AUDIO: Record<string, string> = {
  left_exceeded: 'guidance_left_exceeded.mp3',
  left_more: 'guidance_left_more.mp3',
  right_exceeded: 'guidance_right_exceeded.mp3',
  right_more: 'guidance_right_more.mp3',
  up_exceeded: 'guidance_up_exceeded.mp3',
  up_more: 'guidance_up_more.mp3',
  down_exceeded: 'guidance_down_exceeded.mp3',
  down_more: 'guidance_down_more.mp3',
  level_chin: 'guidance_level_chin.mp3',
  face_forward: 'guidance_face_forward.mp3',
};

/**
 * Timing constants
 */
export const COUNTDOWN_SECONDS = 3;
export const POSE_HOLD_TIME = 0.5; // seconds to hold before countdown
export const STABLE_FRAME_THRESHOLD = 10; // frames required for stability
export const COUNTDOWN_TOLERANCE = 40; // extra degrees during countdown

/**
 * Audio guidance intervals
 */
export const ADVISORY_AUDIO_INTERVAL = 3.0; // seconds between prompts
export const ADVISORY_AUDIO_MAX = 5; // max prompts per capture step

/**
 * Auto-revert timing
 */
export const AUTO_CLOSE_DELAY = 10; // seconds after completion

/**
 * Get pose target by name
 */
export function getPoseTarget(name: PoseName): PoseTarget | undefined {
  return CAPTURE_TARGETS.find((t) => t.name === name);
}
