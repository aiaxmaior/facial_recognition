import type {
  EnrollmentApiResponse,
  StatusApiResponse,
  SubmitCapturesRequest,
  EnrollmentStatus,
} from '../types/enrollment';

export interface ApiClientConfig {
  baseUrl: string;
  timeout?: number;
  /** Tenant host for IoT Broker API calls (e.g., "hbss-bridgestg.qraie.ai") */
  tenantHost?: string;
  /** IoT Broker Data Service base URL (e.g., "https://hbss-bridgestg.qraie.ai") */
  iotBrokerUrl?: string;
}

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
 * API client for enrollment endpoints
 */
export class EnrollmentApiClient {
  private baseUrl: string;
  private timeout: number;
  private tenantHost?: string;
  private iotBrokerUrl?: string;

  constructor(config: ApiClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = config.timeout || 30000;
    this.tenantHost = config.tenantHost;
    this.iotBrokerUrl = config.iotBrokerUrl?.replace(/\/$/, '');
  }

  /**
   * Configure tenant for IoT Broker API calls
   */
  setTenant(tenantHost: string, iotBrokerUrl?: string): void {
    this.tenantHost = tenantHost;
    if (iotBrokerUrl) {
      this.iotBrokerUrl = iotBrokerUrl.replace(/\/$/, '');
    }
  }

  /**
   * Get enrollment status for a user
   */
  async getStatus(userId: string): Promise<StatusApiResponse> {
    const response = await this.fetch(`/status/${userId}`);
    return response.json();
  }

  /**
   * Submit captured images for enrollment
   */
  async submitCaptures(request: SubmitCapturesRequest): Promise<EnrollmentApiResponse> {
    const response = await this.fetch('/capture', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    return response.json();
  }

  /**
   * Publish enrollment to IoT devices via the IoT Broker Data Service
   * 
   * This is the simplified API that only requires employee_id.
   * The system automatically retrieves the embedded_file from the database
   * and broadcasts to all face recognition devices.
   * 
   * @param employeeId - Employee/Person ID to publish enrollment for
   */
  async publishToDevices(employeeId: string): Promise<PublishEnrollmentResponse> {
    // If IoT Broker URL is configured, use the new API directly
    if (this.iotBrokerUrl) {
      const response = await this.fetchIoT('/api/data/enrollment/publish', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ employee_id: employeeId }),
      });
      return response.json();
    }
    
    // Fallback to local API server proxy
    const response = await this.fetch(`/publish/${employeeId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ employee_id: employeeId }),
    });
    return response.json();
  }

  /**
   * Get list of devices from IoT Broker Data Service
   * 
   * @param deviceType - Optional filter: "face_recognition" or "emotion_monitoring"
   */
  async getDevices(deviceType?: 'face_recognition' | 'emotion_monitoring'): Promise<DeviceListResponse> {
    const queryParams = deviceType ? `?device-type=${deviceType}` : '';
    
    // If IoT Broker URL is configured, use it directly
    if (this.iotBrokerUrl) {
      const response = await this.fetchIoT(`/api/data/devices${queryParams}`);
      return response.json();
    }
    
    // Fallback to local API server proxy
    const response = await this.fetch(`/devices${queryParams}`);
    return response.json();
  }

  /**
   * Get list of face recognition devices
   */
  async getFaceRecognitionDevices(): Promise<DeviceListResponse> {
    return this.getDevices('face_recognition');
  }

  /**
   * Delete enrollment data
   */
  async deleteEnrollment(userId: string): Promise<EnrollmentApiResponse> {
    const response = await this.fetch(`/${userId}`, {
      method: 'DELETE',
    });
    return response.json();
  }

  /**
   * Internal fetch wrapper with timeout and error handling
   */
  private async fetch(path: string, options: RequestInit = {}): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...options,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`API error ${response.status}: ${errorBody}`);
      }

      return response;
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Fetch wrapper for IoT Broker API calls with tenant header support
   */
  private async fetchIoT(path: string, options: RequestInit = {}): Promise<Response> {
    if (!this.iotBrokerUrl) {
      throw new Error('IoT Broker URL not configured');
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    // Build headers with tenant support
    const headers: HeadersInit = {
      ...(options.headers as Record<string, string>),
    };

    // Add tenant header if configured
    if (this.tenantHost) {
      // For browser requests, we can't set Host header directly,
      // so use X-Tenant-ID for service identification
      const tenantId = this.extractTenantId(this.tenantHost);
      if (tenantId) {
        headers['X-Tenant-ID'] = tenantId;
      }
    }

    try {
      const response = await fetch(`${this.iotBrokerUrl}${path}`, {
        ...options,
        headers,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`IoT API error ${response.status}: ${errorBody}`);
      }

      return response;
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        throw new Error('IoT API request timeout');
      }
      throw err;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Extract tenant ID from host string
   * Pattern: {tenant-id}-bridge{env}.{domain}
   */
  private extractTenantId(host: string): string | null {
    const match = host.match(/^([a-z0-9]+)-bridge/i);
    return match ? match[1] : null;
  }
}

/**
 * Create a new API client instance
 */
export function createApiClient(baseUrl: string, options?: Partial<ApiClientConfig>): EnrollmentApiClient {
  return new EnrollmentApiClient({ 
    baseUrl,
    ...options
  });
}

/**
 * Create an API client configured for multi-tenant IoT Broker
 */
export function createTenantApiClient(
  baseUrl: string,
  tenantHost: string,
  iotBrokerUrl: string
): EnrollmentApiClient {
  return new EnrollmentApiClient({
    baseUrl,
    tenantHost,
    iotBrokerUrl,
  });
}
