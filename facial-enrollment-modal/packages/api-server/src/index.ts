import express from 'express';
import cors from 'cors';
import path from 'path';
import { enrollmentRouter } from './routes/enrollment';
import { requestLogger, errorLogger } from './middleware/requestLogger';
import { logger } from './services/logger';

const app = express();
const PORT = process.env.PORT || 3003;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Large limit for base64 images
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Request logging middleware (before routes)
app.use(requestLogger);

// Serve static audio files from the parent facial_recognition directory
const audioPath = process.env.AUDIO_PATH || path.join(__dirname, '../../../../audio');
app.use('/audio', express.static(audioPath));

// API routes
app.use('/api/enrollment', enrollmentRouter);

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date().toISOString(),
    services: {
      redis: process.env.REDIS_URL ? 'configured' : 'in-memory-fallback',
      iotBroker: process.env.IOT_BROKER_URL ? 'configured' : 'simulated',
      graylog: process.env.GRAYLOG_HOST ? 'configured' : 'console-only',
    }
  });
});

// Error logging middleware (after routes)
app.use(errorLogger);

// Error handler
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  res.status(500).json({
    success: false,
    error: err.message || 'Internal server error',
  });
});

app.listen(PORT, () => {
  logger.info('Enrollment API server started', {
    action: 'server_start',
  });
  
  console.log(`\n🚀 Enrollment API server running on http://localhost:${PORT}`);
  console.log(`📁 Audio files served from: ${audioPath}`);
  
  console.log(`\n📊 Service Status:`);
  console.log(`   Redis:     ${process.env.REDIS_URL ? '✓ Configured' : '⚠ In-memory fallback'}`);
  console.log(`   IoT Broker: ${process.env.IOT_BROKER_URL ? '✓ Configured' : '⚠ Simulated'}`);
  console.log(`   Graylog:   ${process.env.GRAYLOG_HOST ? '✓ Configured' : '⚠ Console only'}`);
  
  console.log(`\n📡 Endpoints:`);
  console.log(`   GET  /health - Health check`);
  console.log(`   GET  /api/enrollment/status/:employeeId - Get enrollment status`);
  console.log(`   GET  /api/enrollment/status/:employeeId/detailed - Get detailed status with log`);
  console.log(`   GET  /api/enrollment/devices - Get device list`);
  console.log(`   GET  /api/enrollment/pending - Get pending enrollments (admin)`);
  console.log(`   POST /api/enrollment/capture - Submit captured images`);
  console.log(`   POST /api/enrollment/publish/:employeeId - Publish to IoT devices`);
  console.log(`   POST /api/enrollment/complete - Complete enrollment (send to Bridge)`);
  console.log(`   DELETE /api/enrollment/:employeeId - Delete enrollment`);
});

export { app };
