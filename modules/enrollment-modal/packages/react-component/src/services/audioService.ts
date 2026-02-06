import { SECONDARY_AUDIO, ADVISORY_AUDIO_INTERVAL, ADVISORY_AUDIO_MAX } from '../constants';
import type { PoseName } from '../types/enrollment';

/**
 * Audio player service for enrollment guidance cues
 */
class AudioService {
  private enabled: boolean = true;
  private audioCache: Map<string, HTMLAudioElement> = new Map();
  private lastPlayedTime: Map<string, number> = new Map();
  private minRepeatInterval: number = 2000; // ms
  private audioBasePath: string;
  
  // Advisory audio tracking
  private lastAdvisoryTime: number = 0;
  private advisoryCount: number = 0;
  
  // Direction audio tracking - delay guidance after direction plays
  private lastDirectionTime: number = 0;
  private directionGracePeriod: number = 3000; // 3 seconds after direction before guidance

  constructor(audioBasePath: string = '/audio') {
    this.audioBasePath = audioBasePath;
  }

  /**
   * Set the base path for audio files
   */
  setAudioBasePath(path: string): void {
    this.audioBasePath = path;
    this.audioCache.clear();
  }

  /**
   * Enable or disable audio
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Check if audio is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Preload audio files for faster playback
   */
  async preload(filenames: string[]): Promise<void> {
    const promises = filenames.map(async (filename) => {
      try {
        const audio = new Audio(`${this.audioBasePath}/${filename}`);
        await audio.load();
        this.audioCache.set(filename, audio);
      } catch (err) {
        console.warn(`Failed to preload audio: ${filename}`, err);
      }
    });

    await Promise.allSettled(promises);
  }

  /**
   * Play an audio file
   */
  play(filename: string, force: boolean = false): void {
    if (!this.enabled || !filename) {
      console.log(`[Audio] Skipped (disabled or no file): ${filename}`);
      return;
    }

    const now = Date.now();
    const lastPlayed = this.lastPlayedTime.get(filename) || 0;

    // Skip if recently played (unless forced)
    if (!force && now - lastPlayed < this.minRepeatInterval) {
      console.log(`[Audio] Skipped (rate limited): ${filename}`);
      return;
    }

    this.lastPlayedTime.set(filename, now);

    // Get from cache or create new
    let audio = this.audioCache.get(filename);
    
    if (!audio) {
      console.log(`[Audio] Loading new audio file: ${filename}`);
      audio = new Audio(`${this.audioBasePath}/${filename}`);
      this.audioCache.set(filename, audio);
    } else {
      // Reset if already played
      audio.currentTime = 0;
    }

    console.log(`[Audio] ▶️ Playing: ${filename}`);
    audio.play().catch((err) => {
      console.warn(`[Audio] ❌ Playback failed: ${filename}`, err);
    });
  }

  /**
   * Play direction audio for a capture step
   */
  playDirection(poseName: PoseName): void {
    const audioMap: Record<PoseName, string> = {
      front: 'look_forward.mp3',
      left: 'turn_left.mp3',
      right: 'turn_right.mp3',
      up: 'look_up.mp3',
      down: 'look_down.mp3',
    };

    const filename = audioMap[poseName];
    if (filename) {
      this.play(filename, true);
      this.lastDirectionTime = Date.now(); // Track when direction audio played
      console.log(`[Audio] Direction played, guidance delayed for ${this.directionGracePeriod}ms`);
    }
  }

  /**
   * Play "hold pose" audio
   */
  playHold(): void {
    this.play('hold_pose.mp3');
  }

  /**
   * Play countdown beep
   */
  playBeep(): void {
    this.play('beep.mp3', true);
  }

  /**
   * Play capture complete audio
   */
  playComplete(): void {
    this.play('capture_complete.mp3', true);
  }

  /**
   * Reset advisory audio state (call when moving to new step)
   */
  resetAdvisoryState(): void {
    this.lastAdvisoryTime = 0;
    this.advisoryCount = 0;
  }

  /**
   * Play advisory/guidance audio (rate-limited)
   */
  playAdvisory(audioKey: string): void {
    if (!this.enabled) {
      console.log(`[Audio] Advisory skipped - audio disabled`);
      return;
    }

    const now = Date.now();
    
    // Check if we're still in the grace period after direction audio
    const timeSinceDirection = now - this.lastDirectionTime;
    if (timeSinceDirection < this.directionGracePeriod) {
      // Only log occasionally to avoid spam
      if (Math.random() < 0.02) {
        console.log(`[Audio] Advisory skipped - grace period (${(timeSinceDirection/1000).toFixed(1)}s < ${this.directionGracePeriod/1000}s)`);
      }
      return;
    }

    // Check rate limiting
    if (this.advisoryCount >= ADVISORY_AUDIO_MAX) {
      console.log(`[Audio] Advisory skipped - max count reached (${this.advisoryCount}/${ADVISORY_AUDIO_MAX})`);
      return;
    }

    const timeSinceLast = now - this.lastAdvisoryTime;

    if (timeSinceLast < ADVISORY_AUDIO_INTERVAL * 1000) {
      console.log(`[Audio] Advisory skipped - rate limited (${(timeSinceLast/1000).toFixed(1)}s < ${ADVISORY_AUDIO_INTERVAL}s)`);
      return;
    }

    const filename = SECONDARY_AUDIO[audioKey];
    if (filename) {
      console.log(`[Audio] 🔊 Playing advisory: ${audioKey} -> ${filename} (count: ${this.advisoryCount + 1})`);
      this.play(filename);
      this.lastAdvisoryTime = now;
      this.advisoryCount++;
    } else {
      console.warn(`[Audio] Unknown advisory key: ${audioKey}`);
    }
  }

  /**
   * Get guidance audio key based on current pose and target
   * 
   * Pitch convention (physically correct):
   * - Positive pitch = looking UP
   * - Negative pitch = looking DOWN
   * 
   * Up target: pitchRange [15, 50] - need positive pitch
   * Down target: pitchRange [-50, -15] - need negative pitch
   */
  getGuidanceAudioKey(
    yaw: number,
    pitch: number,
    targetName: PoseName,
    yawRange: [number, number],
    pitchRange: [number, number]
  ): string | null {
    const [yawMin, yawMax] = yawRange;
    const [pitchMin, pitchMax] = pitchRange;

    const yawLow = yaw < yawMin;
    const yawHigh = yaw > yawMax;
    const pitchLow = pitch < pitchMin;
    const pitchHigh = pitch > pitchMax;

    const yawOk = !yawLow && !yawHigh;
    const pitchOk = !pitchLow && !pitchHigh;

    switch (targetName) {
      case 'left':
        if (yawOk) {
          return pitchOk ? null : 'level_chin';
        }
        return yawHigh ? 'left_exceeded' : 'left_more';

      case 'right':
        if (yawOk) {
          return pitchOk ? null : 'level_chin';
        }
        return yawLow ? 'right_exceeded' : 'right_more';

      case 'up':
        // Up target needs positive pitch (15 to 50)
        // pitchLow means pitch < 15 (not looking up enough)
        // pitchHigh means pitch > 50 (looking up too much)
        if (pitchOk) {
          return yawOk ? null : 'face_forward';
        }
        return pitchHigh ? 'up_exceeded' : 'up_more';

      case 'down':
        // Down target needs negative pitch (-50 to -15)
        // pitchLow means pitch < -50 (looking down too much)
        // pitchHigh means pitch > -15 (not looking down enough)
        if (pitchOk) {
          return yawOk ? null : 'face_forward';
        }
        return pitchLow ? 'down_exceeded' : 'down_more';

      case 'front':
      default:
        // Front: pitch should be near 0 (-20 to 20)
        // pitchHigh (> 20) means looking up too much
        // pitchLow (< -20) means looking down too much
        if (yawLow) return 'right_exceeded';  // Need to turn right (toward center)
        if (yawHigh) return 'left_exceeded';  // Need to turn left (toward center)
        if (pitchHigh) return 'up_exceeded';  // Looking up, need to look down
        if (pitchLow) return 'down_exceeded'; // Looking down, need to look up
        return null;
    }
  }

  /**
   * Stop all audio
   */
  stopAll(): void {
    this.audioCache.forEach((audio) => {
      audio.pause();
      audio.currentTime = 0;
    });
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.stopAll();
    this.audioCache.clear();
    this.lastPlayedTime.clear();
  }
}

// Singleton instance
let audioServiceInstance: AudioService | null = null;

export function getAudioService(basePath?: string): AudioService {
  if (!audioServiceInstance) {
    audioServiceInstance = new AudioService(basePath);
  } else if (basePath) {
    audioServiceInstance.setAudioBasePath(basePath);
  }
  return audioServiceInstance;
}

export { AudioService };
