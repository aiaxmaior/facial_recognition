import type {
  EnrollmentApiResponse,
  StatusApiResponse,
  SubmitCapturesRequest,
  EnrollmentStatus,
} from '../types/enrollment';

export interface ApiClientConfig {
  baseUrl: string;
  timeout?: number;
}

/**
 * API client for enrollment endpoints
 */
export class EnrollmentApiClient {
  private baseUrl: string;
  private timeout: number;

  constructor(config: ApiClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, ''); // Remove trailing slash
    this.timeout = config.timeout || 30000;
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
   * Publish enrollment to IoT devices
   */
  async publishToDevices(userId: string): Promise<EnrollmentApiResponse> {
    const response = await this.fetch(`/publish/${userId}`, {
      method: 'POST',
    });
    return response.json();
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
}

/**
 * Create a new API client instance
 */
export function createApiClient(baseUrl: string): EnrollmentApiClient {
  return new EnrollmentApiClient({ baseUrl });
}
