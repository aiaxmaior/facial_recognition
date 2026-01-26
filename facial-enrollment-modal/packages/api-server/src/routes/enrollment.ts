import { Router, Request, Response } from 'express';
import { PythonEnrollmentService } from '../services/pythonProxy';
import { IoTBrokerService, createIoTBrokerService, DeviceListResponse } from '../services/iotBrokerService';
import { 
  TransitoryStore, 
  InMemoryTransitoryStore, 
  createTransitoryStore,
  EnrollmentRecord,
  EnrollmentStatus
} from '../services/transitoryStore';
import { logger } from '../services/logger';

const router = Router();

// Initialize services
const pythonService = new PythonEnrollmentService(
  process.env.PYTHON_API_URL || 'http://localhost:5000'
);

const iotBrokerService = createIoTBrokerService();

// Initialize transitory store (Redis or in-memory fallback)
let transitoryStore: TransitoryStore | InMemoryTransitoryStore;

const redisStore = createTransitoryStore();
if (redisStore) {
  transitoryStore = redisStore;
  redisStore.connect().catch(err => {
    console.error('Failed to connect to Redis, using in-memory fallback:', err);
    transitoryStore = new InMemoryTransitoryStore();
    transitoryStore.connect();
  });
} else {
  transitoryStore = new InMemoryTransitoryStore();
  transitoryStore.connect();
}

/**
 * Helper: Map internal status to simplified status for API responses
 */
function toSimplifiedStatus(record: EnrollmentRecord): EnrollmentStatus {
  return record.enrollmentStatus;
}

/**
 * GET /api/enrollment/status/:employeeId
 * Get enrollment status for an employee
 */
router.get('/status/:employeeId', async (req: Request, res: Response) => {
  try {
    const { employeeId } = req.params;
    
    const record = await transitoryStore.getEnrollment(employeeId);
    
    if (!record) {
      return res.json({
        employee_id: employeeId,
        enrollmentStatus: 'unenrolled',
      });
    }
    
    res.json({
      employee_id: record.employee_id,
      enrollmentStatus: toSimplifiedStatus(record),
      enrolled_at: record.published_at || record.captured_at,
      image_count: record.image_count,
      enrollmentPictureThumbnail: record.enrollmentPictureThumbnail || null,
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
 * GET /api/enrollment/status/:employeeId/detailed
 * Get detailed enrollment status with full status log
 */
router.get('/status/:employeeId/detailed', async (req: Request, res: Response) => {
  try {
    const { employeeId } = req.params;
    
    const record = await transitoryStore.getEnrollment(employeeId);
    
    if (!record) {
      return res.json({
        employee_id: employeeId,
        enrollmentStatus: 'unenrolled',
        detailedStatus: 'unenrolled',
        statusLog: [],
      });
    }
    
    res.json({
      employee_id: record.employee_id,
      enrollmentStatus: toSimplifiedStatus(record),
      detailedStatus: record.detailedStatus,
      statusLog: record.statusLog,
      created_at: record.created_at,
      updated_at: record.updated_at,
      captured_at: record.captured_at,
      published_at: record.published_at,
    });
  } catch (err) {
    console.error('Detailed status error:', err);
    res.status(500).json({
      success: false,
      error: 'Failed to get detailed enrollment status',
    });
  }
});

/**
 * GET /api/enrollment/devices
 * Get list of devices from IoT Broker
 * Query param: device-type (optional) - "face_recognition" or "emotion_monitoring"
 */
router.get('/devices', async (req: Request, res: Response) => {
  try {
    const deviceType = req.query['device-type'] as 'face_recognition' | 'emotion_monitoring' | undefined;
    
    if (iotBrokerService) {
      const devices = await iotBrokerService.getDevices(deviceType);
      return res.json(devices);
    }
    
    // Mock response when IoT Broker is not configured
    const mockDevices: DeviceListResponse = {
      devices: [
        {
          device_id: 'cam-001',
          display_name: 'Lobby Camera (Demo)',
          device_category: 'camera',
          capability: 'face_recognition',
          location_label: 'Main Entrance',
          status: 'online',
          last_heartbeat_at: new Date().toISOString(),
        },
        {
          device_id: 'cam-002',
          display_name: 'Break Room Camera (Demo)',
          device_category: 'camera',
          capability: 'face_recognition',
          location_label: 'Break Room',
          status: 'online',
          last_heartbeat_at: new Date().toISOString(),
        },
      ],
      total: 2,
      online: 2,
      offline: 0,
    };
    
    res.json(mockDevices);
  } catch (err) {
    console.error('Device list error:', err);
    res.status(500).json({
      success: false,
      error: 'Failed to get device list',
    });
  }
});

/**
 * GET /api/enrollment/pending
 * Get all pending enrollments (admin endpoint)
 */
router.get('/pending', async (req: Request, res: Response) => {
  try {
    const pending = await transitoryStore.getPendingEnrollments();
    
    res.json({
      success: true,
      count: pending.length,
      enrollments: pending.map(record => ({
        employee_id: record.employee_id,
        enrollmentStatus: toSimplifiedStatus(record),
        detailedStatus: record.detailedStatus,
        created_at: record.created_at,
        updated_at: record.updated_at,
      })),
    });
  } catch (err) {
    console.error('Pending enrollments error:', err);
    res.status(500).json({
      success: false,
      error: 'Failed to get pending enrollments',
    });
  }
});

/**
 * POST /api/enrollment/capture
 * Submit captured images for enrollment processing
 */
router.post('/capture', async (req: Request, res: Response) => {
  try {
    const { employee_id, captures } = req.body;
    // Support both employee_id and userId for backward compatibility
    const employeeId = employee_id || req.body.userId;
    
    if (!employeeId) {
      return res.status(400).json({
        success: false,
        error: 'employee_id is required',
      });
    }
    
    if (!captures || !Array.isArray(captures) || captures.length < 5) {
      return res.status(400).json({
        success: false,
        error: 'At least 5 captured images are required',
      });
    }
    
    logger.enrollmentEvent('capture_started', employeeId, 'unenrolled', {
      action: 'capture_start',
    });
    
    // Create/update enrollment record - mark as pending vectorizer
    await transitoryStore.setPendingVectorizer(employeeId);
    logger.statusChange(employeeId, 'unenrolled', 'pending:vectorizer');
    
    // Try to process with Python backend (GPU vectorizer)
    let enrollmentProcessedFile: string | undefined;
    let enrollmentPictureThumbnail: string | undefined;
    let embeddingCount = captures.length;
    
    const vectorizerStart = Date.now();
    try {
      logger.vectorizerEvent(employeeId, 'start');
      const embeddingResult = await pythonService.processEnrollment(employeeId, captures);
      logger.vectorizerEvent(employeeId, 'complete', Date.now() - vectorizerStart);
      
      // Get the embedding data
      const embeddingData = await pythonService.getEmbedding(employeeId);
      if (embeddingData?.embedding) {
        // Convert embedding array to base64 Float32Array
        enrollmentProcessedFile = Buffer.from(
          new Float32Array(embeddingData.embedding).buffer
        ).toString('base64');
      }
      
      embeddingCount = embeddingResult.embeddingCount;
      enrollmentPictureThumbnail = embeddingResult.profileImagePath;
      
    } catch (pythonErr) {
      logger.warning('Python backend unavailable, using mock processing', {
        employee_id: employeeId,
        action: 'vectorizer_fallback',
        error_type: 'vectorizer_unavailable',
      });
      
      // Mock processing for demo when Python backend is not available
      // In production, this should be an error
      enrollmentProcessedFile = Buffer.from(new Float32Array(512).buffer).toString('base64');
      enrollmentPictureThumbnail = `/images/${employeeId}/profile_128.jpg`;
    }
    
    // Update transitory store with captured data
    const record = await transitoryStore.setCaptured(employeeId, {
      enrollmentProcessedFile: enrollmentProcessedFile || '',
      enrollmentPictureThumbnail: enrollmentPictureThumbnail || '',
      embedding_dim: 512,
      embedding_model: 'ArcFace',
      image_count: embeddingCount,
    });
    
    logger.statusChange(employeeId, 'pending:vectorizer', 'captured');
    logger.enrollmentEvent('capture_complete', employeeId, 'captured', {
      action: 'capture_complete',
    });
    
    res.json({
      success: true,
      message: `Successfully processed ${embeddingCount} images for enrollment`,
      data: {
        employee_id: employeeId,
        embedding_count: embeddingCount,
        enrollmentPictureThumbnail: record.enrollmentPictureThumbnail,
        enrollmentStatus: toSimplifiedStatus(record),
      },
    });
  } catch (err) {
    // Log error in transitory store if we have an employee ID
    const employeeId = req.body.employee_id || req.body.userId;
    
    logger.error('Capture processing failed', {
      employee_id: employeeId,
      action: 'capture_error',
      error_type: 'capture_processing_error',
    }, err instanceof Error ? err : undefined);
    
    if (employeeId) {
      await transitoryStore.setError(
        employeeId, 
        'vectorizer', 
        err instanceof Error ? err.message : 'Unknown error'
      );
    }
    
    res.status(500).json({
      success: false,
      error: err instanceof Error ? err.message : 'Failed to process captures',
    });
  }
});

/**
 * POST /api/enrollment/publish/:employeeId
 * Publish enrollment to IoT edge devices
 * 
 * This endpoint:
 * 1. Gets enrollment data from transitory store
 * 2. Calls IoT Broker to publish to devices
 * 3. On success, sends data to Bridge
 * 4. On full completion, optionally removes from transitory store
 */
router.post('/publish/:employeeId', async (req: Request, res: Response) => {
  try {
    const { employeeId } = req.params;
    // Also support employee_id in body
    const targetEmployeeId = req.body?.employee_id || employeeId;
    
    // Get enrollment from transitory store
    const record = await transitoryStore.getEnrollment(targetEmployeeId);
    
    if (!record) {
      return res.status(404).json({
        success: false,
        error: 'Enrollment not found',
      });
    }
    
    if (record.enrollmentStatus === 'unenrolled' || !record.enrollmentProcessedFile) {
      return res.status(400).json({
        success: false,
        error: 'No enrollmentProcessedFile to publish. Complete capture first.',
      });
    }
    
    logger.enrollmentEvent('publish_started', targetEmployeeId, 'captured', {
      action: 'publish_start',
    });
    
    // Mark as pending IoT confirmation
    await transitoryStore.setPendingIoT(targetEmployeeId);
    logger.statusChange(targetEmployeeId, 'captured', 'pending:iot_confirmation');
    
    let devicesNotified = 0;
    let publishSuccess = false;
    
    // Publish to IoT Broker if configured
    if (iotBrokerService) {
      try {
        const result = await iotBrokerService.publishEnrollment(targetEmployeeId);
        publishSuccess = result.success;
        devicesNotified = result.data?.devices_notified || 0;
        
        logger.iotPublish(targetEmployeeId, publishSuccess, devicesNotified, result.error);
        
        if (!result.success) {
          await transitoryStore.setError(targetEmployeeId, 'iot', result.error || 'IoT publish failed');
        }
      } catch (iotErr) {
        logger.error('IoT Broker error', {
          employee_id: targetEmployeeId,
          action: 'iot_publish',
          error_type: 'iot_broker_error',
        }, iotErr instanceof Error ? iotErr : undefined);
        
        await transitoryStore.setError(
          targetEmployeeId, 
          'iot', 
          iotErr instanceof Error ? iotErr.message : 'IoT publish failed'
        );
      }
    } else {
      // Simulate IoT publish when broker is not configured
      logger.warning('IoT Broker not configured, simulating publish', {
        employee_id: targetEmployeeId,
        action: 'iot_publish_simulated',
      });
      await new Promise((resolve) => setTimeout(resolve, 500));
      publishSuccess = true;
      devicesNotified = 2; // Mock value
      logger.iotPublish(targetEmployeeId, true, devicesNotified);
    }
    
    if (publishSuccess) {
      // Mark as published
      await transitoryStore.setPublished(targetEmployeeId, { devices_notified: devicesNotified });
      
      logger.statusChange(targetEmployeeId, 'pending:iot_confirmation', 'published');
      logger.enrollmentEvent('publish_complete', targetEmployeeId, 'published', {
        action: 'publish_complete',
        devices_notified: devicesNotified,
      });
      
      // Note: In production, you would also call the Bridge API here:
      // await bridgeService.completeEnrollment({
      //   employee_id: targetEmployeeId,
      //   enrollmentProcessedFile: record.enrollmentProcessedFile,
      //   enrollmentPictureThumbnail: record.enrollmentPictureThumbnail,
      //   enrollmentStatus: 'published',
      // });
    }
    
    const updatedRecord = await transitoryStore.getEnrollment(targetEmployeeId);
    
    res.json({
      success: publishSuccess,
      message: publishSuccess 
        ? 'Enrollment published to IoT devices' 
        : 'Failed to publish enrollment',
      data: {
        employee_id: targetEmployeeId,
        devices_notified: devicesNotified,
        enrollmentStatus: updatedRecord ? toSimplifiedStatus(updatedRecord) : 'captured',
      },
    });
  } catch (err) {
    logger.error('Publish failed', {
      employee_id: req.params.employeeId,
      action: 'publish_error',
      error_type: 'publish_error',
    }, err instanceof Error ? err : undefined);
    
    res.status(500).json({
      success: false,
      error: 'Failed to publish enrollment',
    });
  }
});

/**
 * POST /api/enrollment/complete
 * Complete enrollment and send to Bridge (final step)
 */
router.post('/complete', async (req: Request, res: Response) => {
  try {
    const { employee_id } = req.body;
    
    if (!employee_id) {
      return res.status(400).json({
        success: false,
        error: 'employee_id is required',
      });
    }
    
    const record = await transitoryStore.getEnrollment(employee_id);
    
    if (!record) {
      return res.status(404).json({
        success: false,
        error: 'Enrollment not found',
      });
    }
    
    if (record.enrollmentStatus !== 'published') {
      return res.status(400).json({
        success: false,
        error: 'Enrollment must be published before completing',
      });
    }
    
    // Here you would call the Bridge API to persist the data
    // For now, we just return the data that would be sent
    
    const bridgePayload = {
      employee_id: record.employee_id,
      enrollmentProcessedFile: record.enrollmentProcessedFile,
      embedding_dim: record.embedding_dim,
      model: record.embedding_model,
      enrollmentPictureThumbnail: record.enrollmentPictureThumbnail,
      enrollmentStatus: 'published' as EnrollmentStatus,
    };
    
    // Optionally remove from transitory store after successful Bridge confirmation
    // await transitoryStore.deleteEnrollment(employee_id);
    
    res.json({
      success: true,
      message: 'Enrollment complete',
      data: {
        employee_id: record.employee_id,
        enrollmentStatus: 'published',
        enrolled_at: record.published_at,
      },
      bridgePayload, // For debugging - remove in production
    });
  } catch (err) {
    console.error('Complete error:', err);
    res.status(500).json({
      success: false,
      error: 'Failed to complete enrollment',
    });
  }
});

/**
 * DELETE /api/enrollment/:employeeId
 * Delete enrollment data
 */
router.delete('/:employeeId', async (req: Request, res: Response) => {
  try {
    const { employeeId } = req.params;
    
    const record = await transitoryStore.getEnrollment(employeeId);
    
    if (!record) {
      return res.status(404).json({
        success: false,
        error: 'Enrollment not found',
      });
    }
    
    // Delete from Python backend
    try {
      await pythonService.deleteEnrollment(employeeId);
    } catch (err) {
      console.warn('Failed to delete from Python backend:', err);
    }
    
    // Remove from transitory store
    await transitoryStore.deleteEnrollment(employeeId);
    
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
