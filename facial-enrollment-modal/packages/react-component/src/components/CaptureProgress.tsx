import React from 'react';
import { clsx } from 'clsx';
import { CAPTURE_TARGETS } from '../constants';
import type { CapturedFrame } from '../types/enrollment';

export interface CaptureProgressProps {
  captures: CapturedFrame[];
  currentStep: number;
  className?: string;
}

/**
 * Capture progress component showing thumbnail previews
 */
export function CaptureProgress({ captures, currentStep, className }: CaptureProgressProps) {
  return (
    <div className={clsx('capture-progress', className)} style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>Captured Progress</span>
        <span style={styles.counter}>{captures.length}/{CAPTURE_TARGETS.length}</span>
      </div>
      
      <div style={styles.thumbnailList}>
        {CAPTURE_TARGETS.map((target, index) => {
          const capture = captures.find((c) => c.pose === target.name);
          const isCompleted = !!capture;
          const isCurrent = index === currentStep;
          const isPending = index > currentStep;

          return (
            <div
              key={target.name}
              style={{
                ...styles.thumbnailItem,
                ...(isCurrent ? styles.thumbnailCurrent : {}),
              }}
            >
              <div
                style={{
                  ...styles.thumbnail,
                  ...(isCompleted ? styles.thumbnailCompleted : {}),
                  ...(isCurrent ? styles.thumbnailActive : {}),
                }}
              >
                {capture ? (
                  <>
                    <img
                      src={capture.imageData}
                      alt={target.displayName}
                      style={styles.thumbnailImage}
                    />
                    <div style={styles.checkmark}>
                      <CheckIcon />
                    </div>
                  </>
                ) : (
                  <div style={styles.placeholder}>
                    <CameraIcon />
                  </div>
                )}
              </div>
              
              <div style={styles.labelContainer}>
                <span
                  style={{
                    ...styles.label,
                    ...(isCompleted ? styles.labelCompleted : {}),
                    ...(isCurrent ? styles.labelActive : {}),
                  }}
                >
                  {target.displayName}
                </span>
                
                {isCompleted && (
                  <span style={styles.statusCompleted}>Completed</span>
                )}
                {isCurrent && !isCompleted && (
                  <span style={styles.statusInstruction}>
                    {target.instruction.replace('your head ', '').replace('your chin ', '')}
                  </span>
                )}
                {isPending && (
                  <span style={styles.statusPending}>Pending</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CheckIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function CameraIcon() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="2" y="6" width="20" height="14" rx="2" />
      <circle cx="12" cy="13" r="4" />
      <path d="M5 6V4a1 1 0 011-1h3" />
    </svg>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    padding: '16px',
    backgroundColor: '#f8f9fa',
    borderRadius: '8px',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '8px',
  },
  title: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#1a1a1a',
  },
  counter: {
    fontSize: '14px',
    fontWeight: 500,
    color: '#666',
  },
  thumbnailList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  thumbnailItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '8px',
    borderRadius: '8px',
    backgroundColor: '#fff',
    border: '1px solid #e0e0e0',
    transition: 'all 0.2s ease',
  },
  thumbnailCurrent: {
    borderColor: '#3b82f6',
    boxShadow: '0 0 0 2px rgba(59, 130, 246, 0.2)',
  },
  thumbnail: {
    position: 'relative',
    width: '64px',
    height: '64px',
    borderRadius: '8px',
    backgroundColor: '#f0f0f0',
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
  },
  thumbnailCompleted: {
    border: '2px solid #22c55e',
  },
  thumbnailActive: {
    border: '2px solid #3b82f6',
  },
  thumbnailImage: {
    width: '100%',
    height: '100%',
    objectFit: 'cover',
  },
  placeholder: {
    color: '#999',
  },
  checkmark: {
    position: 'absolute',
    bottom: '4px',
    right: '4px',
    width: '20px',
    height: '20px',
    borderRadius: '50%',
    backgroundColor: '#22c55e',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  labelContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
    flex: 1,
    minWidth: 0,
  },
  label: {
    fontSize: '14px',
    fontWeight: 600,
    color: '#333',
  },
  labelCompleted: {
    color: '#22c55e',
  },
  labelActive: {
    color: '#3b82f6',
  },
  statusCompleted: {
    fontSize: '12px',
    color: '#22c55e',
    fontWeight: 500,
  },
  statusInstruction: {
    fontSize: '12px',
    color: '#666',
  },
  statusPending: {
    fontSize: '12px',
    color: '#999',
  },
};
