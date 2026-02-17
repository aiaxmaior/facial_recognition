// Pipeline data types based on analysis JSONs

export interface DetectionConfig {
  frame_sample_rate: number
  min_confidence: number
  min_coverage: number
  model_path: string
}

export interface DetectionSummary {
  total_videos: number
  total_scenes: number
  scenes_with_person: number
  scenes_without_person: number
  vlm_copies_created: number
}

export interface FrameDetection {
  frame_idx: number
  num_persons: number
  max_confidence: number
  total_bbox_area_ratio: number
}

export interface SceneDetection {
  scene_path: string
  duration_seconds: number
  total_frames: number
  sampled_frames: number
  person_present: boolean
  detection_coverage: number
  avg_confidence: number
  avg_bbox_area_ratio: number
  max_persons_detected: number
  frame_detections: FrameDetection[]
}

export interface EmotionConfig {
  sample_rate: number
  valence_mapping: Record<string, number>
  arousal_mapping: Record<string, number>
}

export interface EmotionSummary {
  total_scenes: number
  scenes_with_faces: number
  successful: number
  failed: number
  dominant_emotions: Record<string, number>
}

export interface FrameEmotion {
  frame_idx: number
  num_faces: number
  dominant_emotion: string
  emotion_scores: Record<string, number>
  valence: number
  arousal: number
}

export interface SceneEmotion {
  scene_path: string
  total_frames_analyzed: number
  frames_with_faces: number
  dominant_emotion: string
  emotion_distribution: Record<string, number>
  mean_valence: number
  mean_arousal: number
  valence_range: [number, number]
  arousal_range: [number, number]
  pain_pleasure_score: number
  frame_emotions: FrameEmotion[]
}

export interface CaptionContext {
  scene_name: string
  vlm_video_path: string
  person_coverage: number
  person_confidence: number
  max_persons: number
  dominant_emotion: string
  emotion_confidence: number
  valence: number
  arousal: number
  age_estimates: string[]
  gender_estimates: string[]
  motion_intensity: number
  nudenet_labels: Record<string, number>
  nudenet_detection_rate: number
}

export interface SceneCaption {
  scene_name: string
  vlm_path: string
  context: CaptionContext
  caption: string | null
  error: string | null
}

export interface CaptionSummary {
  total_clips: number
  successful: number
  failed: number
  model: string
  api_url: string
}

export interface ProcessingSummary {
  total_videos: number
  total_frames_processed: number
  total_detections: number
  videos_with_detections: number
  videos_with_exposed: number
  avg_detection_rate: number
  avg_exposed_rate: number
  total_processing_time: number
  output_directory: string
  processed_at: string
}

// Combined clip data for the table
export interface ClipData {
  id: string
  sceneName: string
  videoName: string
  scenePath: string
  vlmPath: string | null
  
  // Detection data
  durationSeconds: number
  totalFrames: number
  sampledFrames: number
  personPresent: boolean
  detectionCoverage: number
  avgConfidence: number
  maxPersons: number
  
  // Emotion data
  dominantEmotion: string
  emotionDistribution: Record<string, number>
  meanValence: number
  meanArousal: number
  painPleasureScore: number
  
  // Caption data
  caption: string | null
  captionError: string | null
  
  // Context data
  ageEstimates: string[]
  genderEstimates: string[]
  motionIntensity: number
  nudenetLabels: Record<string, number>
  nudenetDetectionRate: number
}

// Video summary data
export interface VideoData {
  name: string
  totalScenes: number
  scenesWithPerson: number
  avgDetectionCoverage: number
  avgConfidence: number
  emotionBreakdown: Record<string, number>
  avgValence: number
  avgArousal: number
  totalDuration: number
  captionsGenerated: number
  captionsFailed: number
}

// Pipeline status/insights
export interface PipelineInsights {
  totalClips: number
  clipsWithPerson: number
  clipsWithCaption: number
  clipsFailed: number
  emotionDistribution: Record<string, number>
  averageValence: number
  averageArousal: number
  averageDetectionCoverage: number
  totalVideos: number
  processingTime: number
  processedAt: string
}

// API response types
export interface ApiResponse<T> {
  data: T
  status: 'success' | 'error'
  message?: string
}

// Demographics data types
export interface DemographicsSceneStat {
  sceneName: string
  scenePath: string
  framesAnalyzed: number
  framesWithPersons: number
  meanAge: number | null
  minAge: number | null
  maxAge: number | null
  meanMotion: number
  erraticMotionRatio: number
  ageDistribution: Record<string, number>
  genderDistribution: Record<string, number>
  raceDistribution: Record<string, number>
}

export interface AgeHistogramBin {
  range: string
  min: number
  max: number
  count: number
}

export interface DemographicsSummary {
  totalScenes: number
  scenesWithPersons: number
  totalPersonsDetected: number
  meanAgeOverall: number | null
}

export interface DemographicsData {
  summary: DemographicsSummary
  ageCategories: Record<string, [number, number]>
  ageDistribution: Record<string, number>
  genderDistribution: Record<string, number>
  raceDistribution: Record<string, number>
  ageHistogram: AgeHistogramBin[]
  sceneStats: DemographicsSceneStat[]
}

// NudeNet data types
export interface NudenetSceneStat {
  sceneName: string
  scenePath: string
  framesSampled: number
  framesWithDetections: number
  detectionRate: number
  labelCounts: Record<string, number>
  totalDetections: number
  hasExposed: boolean
}

export interface NudenetSummary {
  totalScenes: number
  scenesWithDetections: number
  scenesWithExposed: number
  avgDetectionRate: number
  totalDetections: number
}

export interface NudenetData {
  summary: NudenetSummary
  labelDistribution: Record<string, number>
  exposedDistribution: Record<string, number>
  coveredDistribution: Record<string, number>
  detectionRateHistogram: AgeHistogramBin[]
  sceneStats: NudenetSceneStat[]
}

// Unified Explorer Data types
export interface FeatureDefinition {
  key: string
  label: string
  category: string
}

export interface UnifiedSceneData {
  sceneName: string
  scenePath: string
  // Demographics
  meanAge?: number | null
  minAge?: number | null
  maxAge?: number | null
  meanMotion?: number
  erraticMotionRatio?: number
  framesWithPersons?: number
  maleCount?: number
  femaleCount?: number
  ageDistribution?: Record<string, number>
  genderDistribution?: Record<string, number>
  raceDistribution?: Record<string, number>
  // Emotions
  dominantEmotion?: string
  meanValence?: number
  meanArousal?: number
  painPleasureScore?: number
  emotionDistribution?: Record<string, number>
  // NudeNet
  nudenetDetectionRate?: number
  nudenetFramesSampled?: number
  nudenetFramesWithDetections?: number
  nudenetTotalDetections?: number
  nudenetExposedCount?: number
  nudenetCoveredCount?: number
  nudenetLabelCounts?: Record<string, number>
  // Detection
  personPresent?: boolean
  detectionCoverage?: number
  avgConfidence?: number
  maxPersons?: number
  durationSeconds?: number
}

export interface ExplorerData {
  scenes: UnifiedSceneData[]
  numericFeatures: FeatureDefinition[]
  categoricalFeatures: FeatureDefinition[]
  totalScenes: number
}

// HAR (Human Action Recognition) data types
export interface HARActionPrediction {
  label: string
  score: number
}

export interface HARClipData {
  original_video: string
  scene_number: number
  clip_path: string
  start_time: number
  end_time: number
  duration: number
  predictions: {
    top_k: HARActionPrediction[]
    rgb: Record<string, number>
    pose: Record<string, number>
    audio: Record<string, number>
  }
  models_used: number[]
}

export interface HARVideoSummary {
  video_name: string
  total_clips: number
  total_duration: number
  action_distribution: Record<string, number>
  dominant_action: string
}

export interface HARSummary {
  total_videos: number
  total_clips: number
  total_duration: number
  unique_actions: number
  action_distribution: Record<string, number>
}

export interface HARActionsData {
  summary: HARSummary
  video_summaries: HARVideoSummary[]
  clips: HARClipData[]
}

// HAR Action with rank for multi-action scenes
export interface RankedAction {
  label: string
  score: number
  rank: number
}

// Extended ClipData with HAR fields
export interface ClipDataWithHAR extends ClipData {
  harAction?: string        // Top-1 action (backwards compat)
  harScore?: number         // Top-1 score (backwards compat)
  harModelsUsed?: number[]
  // NEW: Top-3 ranked actions for complex multi-action scenes
  dominantActions?: RankedAction[]
}

// Per-Person Data Types (for multi-person tracking)
export interface PersonSummary {
  person_idx: number
  appearances: number
  mean_age: number | null
  age_range: [number, number] | null
  dominant_gender: string | null
  gender_distribution: Record<string, number>
  dominant_emotion: string | null
  emotion_distribution: Record<string, number>
  dominant_race: string | null
  needs_review_ratio: number
}

export interface PersonTimelineEntry {
  frame_idx: number
  person_idx: number
  bbox: [number, number, number, number] | null
  age: number | null
  age_source: string | null
  gender: string | null
  gender_confidence: number | null
  race: string | null
  race_confidence: number | null
  emotion: string | null
  emotion_scores: Record<string, number> | null
  valence: number | null
  arousal: number | null
  needs_review: boolean
  age_disagreement: boolean
  gender_disagreement: boolean
}

export interface ScenePersonsData {
  sceneName: string
  totalFramesAnalyzed: number
  maxPersonsInFrame: number
  uniquePersonSlots: number
  personSummaries: PersonSummary[]
  timeline: PersonTimelineEntry[]
}

// =============================================================================
// AUDIO ANALYSIS TYPES
// =============================================================================

export interface AudioSegmentation {
  speechRatio: number
  nonVerbalRatio: number
  silenceRatio: number
  segmentCount?: number
}

export interface AudioAcousticProfile {
  meanPitchHz?: number
  pitchTrend?: string
  energyTrend?: string
  intensityRms?: number
}

export interface AudioClassification {
  dominantCue: string
  confidence: number
  valenceHint: string
}

export interface AudioSceneStat {
  sceneName: string
  scenePath: string
  audioPresent: boolean
  durationSeconds: number
  processingStatus: string
  segmentation?: AudioSegmentation
  acousticProfile?: AudioAcousticProfile
  classification?: AudioClassification
  cotContext?: string
}

export interface AudioSummary {
  totalScenes: number
  scenesWithAudio: number
  scenesWithSpeech: number
  avgDurationSeconds: number
  avgSpeechRatio: number
  avgSilenceRatio: number
}

export interface AudioAnalysisData {
  summary: AudioSummary
  cueDistribution: Record<string, number>
  valenceHintDistribution: Record<string, number>
  pitchTrendDistribution: Record<string, number>
  energyTrendDistribution: Record<string, number>
  durationHistogram: AgeHistogramBin[]  // Reusing histogram bin type
  speechRatioHistogram: AgeHistogramBin[]
  sceneStats: AudioSceneStat[]
}

// Audio-visual fusion types for quadrant visualization
export interface AudioVisualFusionPoint {
  sceneName: string
  audioValence: string  // 'positive', 'negative', 'neutral', 'ambiguous'
  visualValence: number
  audioArousal?: number  // Derived from energy trend
  visualArousal: number
  audioPresent: boolean
  dominantCue?: string
  dominantEmotion?: string
}

// Embedding projection for 3D visualization
export interface EmbeddingProjection {
  sceneName: string
  x: number
  y: number
  z: number
  dominantCue?: string
  valenceHint?: string
  visualEmotion?: string
}
