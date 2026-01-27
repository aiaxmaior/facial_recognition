import { createClient, RedisClientType } from 'redis';

/**
 * Enrollment status values
 */
export type EnrollmentStatus = 'unenrolled' | 'captured' | 'published';

/**
 * Detailed internal status for logging/tracking
 */
export type EnrollmentStatusDetailed =
  | 'unenrolled'
  | 'captured'
  | 'published'
  | 'pending:vectorizer'
  | 'pending:iot_confirmation'
  | 'pending:bridge_confirmation'
  | 'error:vectorizer'
  | 'error:iot'
  | 'error:bridge';

/**
 * Status log entry for tracking enrollment progress
 */
export interface StatusLogEntry {
  status: EnrollmentStatusDetailed;
  timestamp: string;
  message?: string;
  details?: Record<string, unknown>;
}

/**
 * Enrollment record stored in Redis
 */
export interface EnrollmentRecord {
  employee_id: string;
  
  // Status
  enrollmentStatus: EnrollmentStatus;
  detailedStatus: EnrollmentStatusDetailed;
  statusLog: StatusLogEntry[];
  
  // Enrollment data
  enrollmentProcessedFile?: string;  // Base64 Float32Array (512 dims)
  enrollmentPictureThumbnail?: string;  // Base64 JPEG 128x128
  
  // Metadata
  embedding_dim?: number;
  embedding_model?: string;
  image_count?: number;
  
  // Timestamps
  created_at: string;
  updated_at: string;
  captured_at?: string;
  published_at?: string;
  
  // For cleanup
  ttl_days?: number;
}

/**
 * Configuration for Redis transitory store
 */
export interface TransitoryStoreConfig {
  /** Redis connection URL (e.g., redis://localhost:6379) */
  redisUrl: string;
  /** Key prefix for enrollment records */
  keyPrefix?: string;
  /** Default TTL in days for enrollment records (0 = no expiry) */
  defaultTtlDays?: number;
}

/**
 * Redis-based transitory store for enrollment data
 * 
 * Data persists until:
 * 1. Enrollment is successfully published and confirmed
 * 2. User restarts enrollment (overwrites existing)
 * 3. TTL expires (configurable, default 30 days)
 */
export class TransitoryStore {
  private client: RedisClientType;
  private keyPrefix: string;
  private defaultTtlDays: number;
  private connected: boolean = false;

  constructor(config: TransitoryStoreConfig) {
    this.client = createClient({ url: config.redisUrl });
    this.keyPrefix = config.keyPrefix || 'enrollment:';
    this.defaultTtlDays = config.defaultTtlDays || 30;

    // Handle connection events
    this.client.on('error', (err) => {
      console.error('Redis connection error:', err);
      this.connected = false;
    });

    this.client.on('connect', () => {
      console.log('Redis connected');
      this.connected = true;
    });

    this.client.on('disconnect', () => {
      console.log('Redis disconnected');
      this.connected = false;
    });
  }

  /**
   * Connect to Redis
   */
  async connect(): Promise<void> {
    if (!this.connected) {
      await this.client.connect();
      this.connected = true;
    }
  }

  /**
   * Disconnect from Redis
   */
  async disconnect(): Promise<void> {
    if (this.connected) {
      await this.client.quit();
      this.connected = false;
    }
  }

  /**
   * Check if connected to Redis
   */
  isConnected(): boolean {
    return this.connected;
  }

  /**
   * Generate Redis key for an employee
   */
  private getKey(employeeId: string): string {
    return `${this.keyPrefix}${employeeId}`;
  }

  /**
   * Create a new enrollment record
   */
  async createEnrollment(employeeId: string): Promise<EnrollmentRecord> {
    const now = new Date().toISOString();
    
    const record: EnrollmentRecord = {
      employee_id: employeeId,
      enrollmentStatus: 'unenrolled',
      detailedStatus: 'unenrolled',
      statusLog: [{
        status: 'unenrolled',
        timestamp: now,
        message: 'Enrollment initiated',
      }],
      created_at: now,
      updated_at: now,
      ttl_days: this.defaultTtlDays,
    };

    await this.saveRecord(employeeId, record);
    return record;
  }

  /**
   * Get enrollment record by employee ID
   */
  async getEnrollment(employeeId: string): Promise<EnrollmentRecord | null> {
    const key = this.getKey(employeeId);
    const data = await this.client.get(key);
    
    if (!data) {
      return null;
    }

    return JSON.parse(data) as EnrollmentRecord;
  }

  /**
   * Update enrollment with captured data (after vectorizer processing)
   */
  async setCaptured(
    employeeId: string,
    data: {
      enrollmentProcessedFile: string;
      enrollmentPictureThumbnail: string;
      embedding_dim?: number;
      embedding_model?: string;
      image_count?: number;
    }
  ): Promise<EnrollmentRecord> {
    let record = await this.getEnrollment(employeeId);
    
    if (!record) {
      record = await this.createEnrollment(employeeId);
    }

    const now = new Date().toISOString();

    record.enrollmentStatus = 'captured';
    record.detailedStatus = 'captured';
    record.enrollmentProcessedFile = data.enrollmentProcessedFile;
    record.enrollmentPictureThumbnail = data.enrollmentPictureThumbnail;
    record.embedding_dim = data.embedding_dim || 512;
    record.embedding_model = data.embedding_model || 'ArcFace';
    record.image_count = data.image_count || 5;
    record.captured_at = now;
    record.updated_at = now;

    record.statusLog.push({
      status: 'captured',
      timestamp: now,
      message: 'Images processed, embedding generated',
      details: {
        embedding_dim: record.embedding_dim,
        embedding_model: record.embedding_model,
        image_count: record.image_count,
      },
    });

    await this.saveRecord(employeeId, record);
    return record;
  }

  /**
   * Update status to pending vectorizer
   */
  async setPendingVectorizer(employeeId: string): Promise<EnrollmentRecord> {
    const record = await this.getOrCreate(employeeId);
    const now = new Date().toISOString();

    record.detailedStatus = 'pending:vectorizer';
    record.updated_at = now;

    record.statusLog.push({
      status: 'pending:vectorizer',
      timestamp: now,
      message: 'Sending images to GPU vectorizer',
    });

    await this.saveRecord(employeeId, record);
    return record;
  }

  /**
   * Update status to pending IoT confirmation
   */
  async setPendingIoT(employeeId: string): Promise<EnrollmentRecord> {
    const record = await this.getOrCreate(employeeId);
    const now = new Date().toISOString();

    record.detailedStatus = 'pending:iot_confirmation';
    record.updated_at = now;

    record.statusLog.push({
      status: 'pending:iot_confirmation',
      timestamp: now,
      message: 'Publishing to IoT broker, awaiting confirmation',
    });

    await this.saveRecord(employeeId, record);
    return record;
  }

  /**
   * Update status to pending Bridge confirmation
   */
  async setPendingBridge(employeeId: string): Promise<EnrollmentRecord> {
    const record = await this.getOrCreate(employeeId);
    const now = new Date().toISOString();

    record.detailedStatus = 'pending:bridge_confirmation';
    record.updated_at = now;

    record.statusLog.push({
      status: 'pending:bridge_confirmation',
      timestamp: now,
      message: 'Sending to Bridge, awaiting confirmation',
    });

    await this.saveRecord(employeeId, record);
    return record;
  }

  /**
   * Mark enrollment as published (complete)
   */
  async setPublished(
    employeeId: string,
    details?: { devices_notified?: number }
  ): Promise<EnrollmentRecord> {
    const record = await this.getOrCreate(employeeId);
    const now = new Date().toISOString();

    record.enrollmentStatus = 'published';
    record.detailedStatus = 'published';
    record.published_at = now;
    record.updated_at = now;

    record.statusLog.push({
      status: 'published',
      timestamp: now,
      message: 'Enrollment published successfully',
      details,
    });

    await this.saveRecord(employeeId, record);
    return record;
  }

  /**
   * Set error status
   */
  async setError(
    employeeId: string,
    errorType: 'vectorizer' | 'iot' | 'bridge',
    errorMessage: string
  ): Promise<EnrollmentRecord> {
    const record = await this.getOrCreate(employeeId);
    const now = new Date().toISOString();

    const errorStatus = `error:${errorType}` as EnrollmentStatusDetailed;
    record.detailedStatus = errorStatus;
    record.updated_at = now;

    record.statusLog.push({
      status: errorStatus,
      timestamp: now,
      message: errorMessage,
    });

    await this.saveRecord(employeeId, record);
    return record;
  }

  /**
   * Delete enrollment record (after successful completion or manual cleanup)
   */
  async deleteEnrollment(employeeId: string): Promise<boolean> {
    const key = this.getKey(employeeId);
    const result = await this.client.del(key);
    return result > 0;
  }

  /**
   * Get all pending enrollments (for monitoring/admin)
   */
  async getPendingEnrollments(): Promise<EnrollmentRecord[]> {
    const keys = await this.client.keys(`${this.keyPrefix}*`);
    const records: EnrollmentRecord[] = [];

    for (const key of keys) {
      const data = await this.client.get(key);
      if (data) {
        const record = JSON.parse(data) as EnrollmentRecord;
        if (record.enrollmentStatus !== 'published') {
          records.push(record);
        }
      }
    }

    return records;
  }

  /**
   * Get status log for an enrollment
   */
  async getStatusLog(employeeId: string): Promise<StatusLogEntry[]> {
    const record = await this.getEnrollment(employeeId);
    return record?.statusLog || [];
  }

  /**
   * Helper: Get or create enrollment record
   */
  private async getOrCreate(employeeId: string): Promise<EnrollmentRecord> {
    let record = await this.getEnrollment(employeeId);
    if (!record) {
      record = await this.createEnrollment(employeeId);
    }
    return record;
  }

  /**
   * Helper: Save record to Redis with TTL
   */
  private async saveRecord(employeeId: string, record: EnrollmentRecord): Promise<void> {
    const key = this.getKey(employeeId);
    const data = JSON.stringify(record);

    if (record.ttl_days && record.ttl_days > 0) {
      const ttlSeconds = record.ttl_days * 24 * 60 * 60;
      await this.client.setEx(key, ttlSeconds, data);
    } else {
      await this.client.set(key, data);
    }
  }
}

/**
 * Create transitory store from environment variables
 */
export function createTransitoryStore(): TransitoryStore | null {
  const redisUrl = process.env.REDIS_URL || process.env.TRANSITORY_REDIS_URL;

  if (!redisUrl) {
    console.warn('REDIS_URL not configured, transitory store will not be available');
    return null;
  }

  const ttlDays = process.env.ENROLLMENT_TTL_DAYS 
    ? parseInt(process.env.ENROLLMENT_TTL_DAYS, 10) 
    : 30;

  return new TransitoryStore({
    redisUrl,
    keyPrefix: 'enrollment:',
    defaultTtlDays: ttlDays,
  });
}

/**
 * In-memory fallback store for development/testing
 * (when Redis is not available)
 */
export class InMemoryTransitoryStore {
  private store = new Map<string, EnrollmentRecord>();

  async connect(): Promise<void> {
    console.log('Using in-memory transitory store (data will not persist)');
  }

  async disconnect(): Promise<void> {}

  isConnected(): boolean {
    return true;
  }

  async createEnrollment(employeeId: string): Promise<EnrollmentRecord> {
    const now = new Date().toISOString();
    const record: EnrollmentRecord = {
      employee_id: employeeId,
      enrollmentStatus: 'unenrolled',
      detailedStatus: 'unenrolled',
      statusLog: [{ status: 'unenrolled', timestamp: now, message: 'Enrollment initiated' }],
      created_at: now,
      updated_at: now,
    };
    this.store.set(employeeId, record);
    return record;
  }

  async getEnrollment(employeeId: string): Promise<EnrollmentRecord | null> {
    return this.store.get(employeeId) || null;
  }

  async setCaptured(
    employeeId: string,
    data: {
      enrollmentProcessedFile: string;
      enrollmentPictureThumbnail: string;
      embedding_dim?: number;
      embedding_model?: string;
      image_count?: number;
    }
  ): Promise<EnrollmentRecord> {
    let record = this.store.get(employeeId);
    if (!record) {
      record = await this.createEnrollment(employeeId);
    }

    const now = new Date().toISOString();
    record.enrollmentStatus = 'captured';
    record.detailedStatus = 'captured';
    record.enrollmentProcessedFile = data.enrollmentProcessedFile;
    record.enrollmentPictureThumbnail = data.enrollmentPictureThumbnail;
    record.embedding_dim = data.embedding_dim || 512;
    record.embedding_model = data.embedding_model || 'ArcFace';
    record.image_count = data.image_count || 5;
    record.captured_at = now;
    record.updated_at = now;
    record.statusLog.push({ status: 'captured', timestamp: now, message: 'Images processed' });

    this.store.set(employeeId, record);
    return record;
  }

  async setPendingVectorizer(employeeId: string): Promise<EnrollmentRecord> {
    const record = this.store.get(employeeId) || await this.createEnrollment(employeeId);
    record.detailedStatus = 'pending:vectorizer';
    record.updated_at = new Date().toISOString();
    this.store.set(employeeId, record);
    return record;
  }

  async setPendingIoT(employeeId: string): Promise<EnrollmentRecord> {
    const record = this.store.get(employeeId) || await this.createEnrollment(employeeId);
    record.detailedStatus = 'pending:iot_confirmation';
    record.updated_at = new Date().toISOString();
    this.store.set(employeeId, record);
    return record;
  }

  async setPendingBridge(employeeId: string): Promise<EnrollmentRecord> {
    const record = this.store.get(employeeId) || await this.createEnrollment(employeeId);
    record.detailedStatus = 'pending:bridge_confirmation';
    record.updated_at = new Date().toISOString();
    this.store.set(employeeId, record);
    return record;
  }

  async setPublished(employeeId: string, details?: { devices_notified?: number }): Promise<EnrollmentRecord> {
    const record = this.store.get(employeeId) || await this.createEnrollment(employeeId);
    const now = new Date().toISOString();
    record.enrollmentStatus = 'published';
    record.detailedStatus = 'published';
    record.published_at = now;
    record.updated_at = now;
    record.statusLog.push({ status: 'published', timestamp: now, message: 'Published', details });
    this.store.set(employeeId, record);
    return record;
  }

  async setError(employeeId: string, errorType: 'vectorizer' | 'iot' | 'bridge', errorMessage: string): Promise<EnrollmentRecord> {
    const record = this.store.get(employeeId) || await this.createEnrollment(employeeId);
    record.detailedStatus = `error:${errorType}` as EnrollmentStatusDetailed;
    record.updated_at = new Date().toISOString();
    record.statusLog.push({ status: record.detailedStatus, timestamp: record.updated_at, message: errorMessage });
    this.store.set(employeeId, record);
    return record;
  }

  async deleteEnrollment(employeeId: string): Promise<boolean> {
    return this.store.delete(employeeId);
  }

  async getPendingEnrollments(): Promise<EnrollmentRecord[]> {
    return Array.from(this.store.values()).filter(r => r.enrollmentStatus !== 'published');
  }

  async getStatusLog(employeeId: string): Promise<StatusLogEntry[]> {
    return this.store.get(employeeId)?.statusLog || [];
  }
}
