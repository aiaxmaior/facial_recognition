import { useRef, useState, useCallback, useEffect } from 'react';

export interface UseCameraOptions {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  frameRate?: number;
}

export interface UseCameraReturn {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  isReady: boolean;
  error: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  captureFrame: () => string | null;
  getVideoElement: () => HTMLVideoElement | null;
}

const DEFAULT_OPTIONS: UseCameraOptions = {
  width: 640,
  height: 480,
  facingMode: 'user',
  frameRate: 30,
};

/**
 * Hook for managing webcam access via WebRTC getUserMedia
 */
export function useCamera(options: UseCameraOptions = {}): UseCameraReturn {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const startCamera = useCallback(async () => {
    console.log('[Camera] 📷 Starting camera...');
    setError(null);
    setIsReady(false);

    try {
      // Check for browser support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera not supported in this browser');
      }

      console.log('[Camera] Requesting camera access...');
      // Request camera access
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: opts.width },
          height: { ideal: opts.height },
          facingMode: opts.facingMode,
          frameRate: { ideal: opts.frameRate },
        },
        audio: false,
      });

      console.log('[Camera] ✅ Camera stream obtained');
      streamRef.current = stream;

      // Attach stream to video element
      if (videoRef.current) {
        const video = videoRef.current;
        video.srcObject = stream;
        
        // Wait for video to be ready
        await new Promise<void>((resolve, reject) => {
          let resolved = false;
          
          const onReady = async () => {
            if (resolved) return;
            resolved = true;
            
            console.log(`[Camera] Video ready - dimensions: ${video.videoWidth}x${video.videoHeight}`);
            
            try {
              // Try to play the video
              await video.play();
              console.log('[Camera] ✅ Camera fully initialized and playing');
              setIsReady(true);
              resolve();
            } catch (playError) {
              // Autoplay might be blocked, but video is still usable
              console.warn('[Camera] ⚠️ Autoplay blocked, but camera is ready:', playError);
              setIsReady(true);
              resolve();
            }
          };
          
          // Check if already ready (metadata already loaded)
          if (video.readyState >= 1) {
            console.log('[Camera] Video already has metadata, readyState:', video.readyState);
            onReady();
            return;
          }
          
          console.log('[Camera] Waiting for video metadata...');
          
          // Listen for metadata loaded
          video.onloadedmetadata = () => {
            console.log('[Camera] onloadedmetadata fired');
            onReady();
          };
          
          // Also listen for canplay as a backup
          video.oncanplay = () => {
            console.log('[Camera] oncanplay fired');
            onReady();
          };
          
          video.onerror = () => {
            if (!resolved) {
              resolved = true;
              console.error('[Camera] ❌ Video error event');
              reject(new Error('Failed to load video'));
            }
          };
          
          // Timeout after 10 seconds
          setTimeout(() => {
            if (!resolved) {
              resolved = true;
              console.log('[Camera] Timeout reached, readyState:', video.readyState);
              // If video has some data, consider it ready anyway
              if (video.readyState >= 1) {
                console.log('[Camera] ⚠️ Timeout but video has data - continuing');
                setIsReady(true);
                resolve();
              } else {
                console.error('[Camera] ❌ Camera initialization timeout');
                reject(new Error('Camera initialization timeout'));
              }
            }
          }, 10000);
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to access camera';
      
      // Provide user-friendly error messages
      if (message.includes('NotAllowedError') || message.includes('Permission denied')) {
        setError('Camera access denied. Please allow camera permissions.');
      } else if (message.includes('NotFoundError') || message.includes('DevicesNotFoundError')) {
        setError('No camera found. Please connect a webcam.');
      } else if (message.includes('NotReadableError') || message.includes('TrackStartError')) {
        setError('Camera is in use by another application.');
      } else {
        setError(message);
      }
      
      setIsReady(false);
    }
  }, [opts.width, opts.height, opts.facingMode, opts.frameRate]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    
    setIsReady(false);
  }, []);

  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas || !isReady) {
      console.log('[Camera] captureFrame: not ready', { video: !!video, canvas: !!canvas, isReady });
      return null;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.log('[Camera] captureFrame: no canvas context');
      return null;
    }

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame to canvas (mirrored for selfie view)
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
    ctx.restore();

    // Return as base64 JPEG
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    console.log(`[Camera] 📸 Frame captured: ${canvas.width}x${canvas.height}, ${(imageData.length / 1024).toFixed(1)}KB`);
    return imageData;
  }, [isReady]);

  const getVideoElement = useCallback((): HTMLVideoElement | null => {
    return videoRef.current;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  return {
    videoRef,
    canvasRef,
    isReady,
    error,
    startCamera,
    stopCamera,
    captureFrame,
    getVideoElement,
  };
}
