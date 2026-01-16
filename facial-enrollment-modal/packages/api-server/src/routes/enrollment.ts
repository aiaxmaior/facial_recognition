import { Router, Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { PythonEnrollmentService } from '../services/pythonProxy';

const router = Router();

// Initialize Python service
const pythonService = new PythonEnrollmentService(
  process.env.PYTHON_API_URL || 'http://localhost:5000'
);

// In-memory store for demo (replace with database in production)
interface EnrollmentRecord {
  userId: string;
  status: 'enrolled' | 'pending' | 'unenrolled';
  enrolledAt?: string;
  imageCount?: number;
  profileImagePath?: string;
  embeddingId?: string;
}

const enrollmentStore = new Map<string, EnrollmentRecord>();

/**
 * GET /api/enrollment/status/:userId
 * Get enrollment status for a user
 */
router.get('/status/:userId', async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const record = enrollmentStore.get(userId);
    
    if (!record) {
      return res.json({
        userId,
        status: 'unenrolled',
      });
    }
    
    res.json({
      userId: record.userId,
      status: record.status,
      enrolledAt: record.enrolledAt,
      imageCount: record.imageCount,
      profileImageUrl: record.profileImagePath ? `/api/enrollment/profile/${userId}` : undefined,
    });
  } catch (err) {
    console.error('Status error:', err);
    res.status(500).json({
      success: false,
      error: 'Failed to get enrollment status',
    });
  }
});

/**
 * POST /api/enrollment/capture
 * Submit captured images for enrollment processing
 */
router.post('/capture', async (req: Request, res: Response) => {
  try {
    const { userId, captures } = req.body;
    
    if (!userId) {
      return res.status(400).json({
        success: false,
        error: 'userId is required',
      });
    }
    
    if (!captures || !Array.isArray(captures) || captures.length < 5) {
      return res.status(400).json({
        success: false,
        error: 'At least 5 captured images are required',
      });
    }
    
    console.log(`Processing enrollment for user: ${userId}`);
    console.log(`Received ${captures.length} captures`);
    
    // Try to process with Python backend
    let embeddingResult;
    try {
      embeddingResult = await pythonService.processEnrollment(userId, captures);
      console.log('Python processing successful:', embeddingResult);
    } catch (pythonErr) {
      console.warn('Python backend unavailable, using mock processing:', pythonErr);
      
      // Mock processing for demo when Python backend is not available
      embeddingResult = {
        success: true,
        embeddingCount: captures.length,
        profileImagePath: `/images/${userId}/profile_128.jpg`,
      };
    }
    
    // Update enrollment store
    const record: EnrollmentRecord = {
      userId,
      status: 'pending', // Pending until published to IoT
      enrolledAt: new Date().toISOString(),
      imageCount: embeddingResult.embeddingCount,
      profileImagePath: embeddingResult.profileImagePath,
      embeddingId: uuidv4(),
    };
    
    enrollmentStore.set(userId, record);
    
    res.json({
      success: true,
      message: `Successfully processed ${embeddingResult.embeddingCount} images for enrollment`,
      data: {
        userId,
        embeddingCount: embeddingResult.embeddingCount,
        profileImagePath: embeddingResult.profileImagePath,
        status: record.status,
      },
    });
  } catch (err) {
    console.error('Capture processing error:', err);
    res.status(500).json({
      success: false,
      error: err instanceof Error ? err.message : 'Failed to process captures',
    });
  }
});

/**
 * POST /api/enrollment/publish/:userId
 * Publish enrollment to IoT edge devices
 */
router.post('/publish/:userId', async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const record = enrollmentStore.get(userId);
    
    if (!record) {
      return res.status(404).json({
        success: false,
        error: 'Enrollment not found',
      });
    }
    
    if (record.status === 'unenrolled') {
      return res.status(400).json({
        success: false,
        error: 'No embedding to publish',
      });
    }
    
    // TODO: Implement actual IoT publishing
    // This would call the IoT broker service to sync embeddings to edge devices
    console.log(`Publishing enrollment to IoT devices for user: ${userId}`);
    
    // Simulate IoT publish delay
    await new Promise((resolve) => setTimeout(resolve, 500));
    
    // Update status to enrolled
    record.status = 'enrolled';
    enrollmentStore.set(userId, record);
    
    res.json({
      success: true,
      message: 'Enrollment published to IoT devices',
      data: {
        userId,
        status: 'enrolled',
      },
    });
  } catch (err) {
    console.error('Publish error:', err);
    res.status(500).json({
      success: false,
      error: 'Failed to publish enrollment',
    });
  }
});

/**
 * DELETE /api/enrollment/:userId
 * Delete enrollment data
 */
router.delete('/:userId', async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const record = enrollmentStore.get(userId);
    
    if (!record) {
      return res.status(404).json({
        success: false,
        error: 'Enrollment not found',
      });
    }
    
    // TODO: Also delete from Python backend and IoT devices
    enrollmentStore.delete(userId);
    
    res.json({
      success: true,
      message: 'Enrollment deleted',
    });
  } catch (err) {
    console.error('Delete error:', err);
    res.status(500).json({
      success: false,
      error: 'Failed to delete enrollment',
    });
  }
});

export { router as enrollmentRouter };
