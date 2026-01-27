import axios, { AxiosInstance } from 'axios';

/**
 * Configuration for IoT Broker Service
 */
export interface IoTBrokerConfig {
  /** Base URL for IoT Broker Data Service (e.g., "https://hbss-bridgestg.qraie.ai") */
  dataServiceUrl: string;
  /** Tenant ID for multi-tenant support */
  tenantId?: string;
  /** Request timeout in milliseconds */
  timeout?: number;
}

/**
 * Device information from IoT Broker
 */
export interface EdgeDevice {
  device_id: string;
  display_name: string;
  device_category: string;
  capability: 'face_recognition' | 'emotion_monitoring';
  location_label?: string;
  status: 'online' | 'offline' | 'provisioning';
  last_heartbeat_at: string | null;
}

export interface DeviceListResponse {
  devices: EdgeDevice[];
  total: number;
  online: number;
  offline: number;
}

export interface PublishEnrollmentRequest {
  employee_id: string;
}

export interface PublishEnrollmentResponse {
  success: boolean;
  message: string;
  data?: {
    employee_id: string;
    devices_notified: number;
  };
  error?: string;
}

/**
 * Service for communicating with the IoT Broker Data Service
 * 
 * The IoT Broker uses a multi-tenant architecture where tenant ID
 * is extracted from the Host header or X-Tenant-ID header.
 */
export class IoTBrokerService {
  private client: AxiosInstance;
  private tenantId?: string;

  constructor(config: IoTBrokerConfig) {
    this.tenantId = config.tenantId;
    
    this.client = axios.create({
      baseURL: config.dataServiceUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Content-Type': 'application/json',
        // Add tenant header if configured
        ...(config.tenantId && { 'X-Tenant-ID': config.tenantId }),
      },
    });
  }

  /**
   * Set or update the tenant ID for multi-tenant support
   */
  setTenantId(tenantId: string): void {
    this.tenantId = tenantId;
    this.client.defaults.headers.common['X-Tenant-ID'] = tenantId;
  }

  /**
   * Get list of devices from IoT Broker
   * 
   * @param deviceType - Optional filter: "face_recognition" or "emotion_monitoring"
   */
  async getDevices(deviceType?: 'face_recognition' | 'emotion_monitoring'): Promise<DeviceListResponse> {
    try {
      const params = deviceType ? { 'device-type': deviceType } : {};
      const response = await this.client.get('/api/data/devices', { params });
      return response.data;
    } catch (err) {
      if (axios.isAxiosError(err)) {
        throw new Error(`IoT Broker error: ${err.response?.data?.error || err.message}`);
      }
      throw err;
    }
  }

  /**
   * Get list of face recognition devices
   */
  async getFaceRecognitionDevices(): Promise<DeviceListResponse> {
    return this.getDevices('face_recognition');
  }

  /**
   * Publish enrollment to all face recognition devices
   * 
   * This is the simplified API that only requires employee_id.
   * The IoT Broker automatically retrieves the embedded_file from
   * the database and broadcasts to all face recognition devices.
   * 
   * @param employeeId - Employee/Person ID to publish enrollment for
   */
  async publishEnrollment(employeeId: string): Promise<PublishEnrollmentResponse> {
    try {
      const response = await this.client.post<PublishEnrollmentResponse>(
        '/api/data/enrollment/publish',
        { employee_id: employeeId }
      );
      return response.data;
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const errorMessage = err.response?.data?.error || err.message;
        return {
          success: false,
          message: 'Failed to publish enrollment',
          error: errorMessage,
        };
      }
      throw err;
    }
  }

  /**
   * Register a new device
   */
  async registerDevice(device: {
    device_id: string;
    display_name: string;
    capability: 'face_recognition' | 'emotion_monitoring';
    location_label?: string;
    device_category?: string;
  }): Promise<EdgeDevice> {
    try {
      const response = await this.client.post('/api/data/devices', device);
      return response.data;
    } catch (err) {
      if (axios.isAxiosError(err)) {
        throw new Error(`Failed to register device: ${err.response?.data?.error || err.message}`);
      }
      throw err;
    }
  }

  /**
   * Update device heartbeat
   */
  async updateHeartbeat(deviceId: string): Promise<void> {
    try {
      await this.client.post(`/api/data/devices/${deviceId}/heartbeat`);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        throw new Error(`Heartbeat failed: ${err.response?.data?.error || err.message}`);
      }
      throw err;
    }
  }

  /**
   * Health check for IoT Broker Data Service
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health', { timeout: 5000 });
      return response.status === 200;
    } catch {
      return false;
    }
  }
}

/**
 * Create IoT Broker service from environment variables
 */
export function createIoTBrokerService(): IoTBrokerService | null {
  const dataServiceUrl = process.env.IOT_BROKER_URL;
  const tenantId = process.env.IOT_TENANT_ID;

  if (!dataServiceUrl) {
    console.warn('IOT_BROKER_URL not configured, IoT publishing will be simulated');
    return null;
  }

  return new IoTBrokerService({
    dataServiceUrl,
    tenantId,
  });
}
