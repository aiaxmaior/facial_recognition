import axios, { AxiosInstance } from 'axios';

interface CaptureData {
  pose: string;
  imageData: string; // Base64 encoded
}

interface EnrollmentResponse {
  success: boolean;
  embeddingCount: number;
  profileImagePath?: string;
  error?: string;
}

/**
 * Service for communicating with the Python DeepFace backend
 */
export class PythonEnrollmentService {
  private client: AxiosInstance;

  constructor(baseUrl: string) {
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 60000, // 60 seconds for processing
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  /**
   * Process enrollment images via Python backend
   */
  async processEnrollment(userId: string, captures: CaptureData[]): Promise<EnrollmentResponse> {
    try {
      const response = await this.client.post('/api/process', {
        user_id: userId,
        captures: captures.map((c) => ({
          pose: c.pose,
          image_data: c.imageData,
        })),
      });

      return {
        success: response.data.success,
        embeddingCount: response.data.embedding_count || captures.length,
        profileImagePath: response.data.profile_image_path,
      };
    } catch (err) {
      if (axios.isAxiosError(err)) {
        if (err.code === 'ECONNREFUSED') {
          throw new Error('Python backend not available');
        }
        throw new Error(err.response?.data?.error || err.message);
      }
      throw err;
    }
  }

  /**
   * Get embedding for a user
   */
  async getEmbedding(userId: string): Promise<{ embedding: number[] } | null> {
    try {
      const response = await this.client.get(`/api/embedding/${userId}`);
      return response.data;
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 404) {
        return null;
      }
      throw err;
    }
  }

  /**
   * Delete enrollment from Python backend
   */
  async deleteEnrollment(userId: string): Promise<boolean> {
    try {
      await this.client.delete(`/api/enrollment/${userId}`);
      return true;
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 404) {
        return false;
      }
      throw err;
    }
  }

  /**
   * Health check for Python backend
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
