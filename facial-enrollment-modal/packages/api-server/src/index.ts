import express from 'express';
import cors from 'cors';
import { enrollmentRouter } from './routes/enrollment';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Large limit for base64 images
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Serve static audio files from the parent facial_recognition directory
const audioPath = process.env.AUDIO_PATH || path.join(__dirname, '../../../../audio');
app.use('/audio', express.static(audioPath));

// API routes
app.use('/api/enrollment', enrollmentRouter);

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Error handler
app.use((err: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('API Error:', err);
  res.status(500).json({
    success: false,
    error: err.message || 'Internal server error',
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Enrollment API server running on http://localhost:${PORT}`);
  console.log(`📁 Audio files served from: ${audioPath}`);
  console.log(`\nEndpoints:`);
  console.log(`  GET  /health - Health check`);
  console.log(`  GET  /api/enrollment/status/:userId - Get enrollment status`);
  console.log(`  POST /api/enrollment/capture - Submit captured images`);
  console.log(`  POST /api/enrollment/publish/:userId - Publish to IoT devices`);
  console.log(`  DELETE /api/enrollment/:userId - Delete enrollment`);
});

export { app };
