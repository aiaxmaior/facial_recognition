import { Request, Response, NextFunction } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '../services/logger';

/**
 * Extend Express Request to include request ID
 */
declare global {
  namespace Express {
    interface Request {
      requestId?: string;
      startTime?: number;
    }
  }
}

/**
 * Request logging middleware
 * - Assigns unique request ID
 * - Logs request start and completion
 * - Tracks response time
 */
export function requestLogger(req: Request, res: Response, next: NextFunction): void {
  // Assign request ID
  req.requestId = req.headers['x-request-id'] as string || uuidv4();
  req.startTime = Date.now();

  // Add request ID to response headers
  res.setHeader('X-Request-ID', req.requestId);

  // Log on response finish
  res.on('finish', () => {
    const duration = Date.now() - (req.startTime || Date.now());
    
    // Extract employee ID from params or body if available
    const employeeId = req.params.employeeId || 
                       req.params.userId || 
                       req.body?.employee_id || 
                       req.body?.userId;

    // Extract tenant ID from headers
    const tenantId = req.headers['x-tenant-id'] as string || undefined;

    logger.apiRequest(
      req.method,
      req.originalUrl || req.url,
      res.statusCode,
      duration,
      {
        employee_id: employeeId,
        request_id: req.requestId,
        tenant_id: tenantId,
      }
    );
  });

  next();
}

/**
 * Error logging middleware
 * Must be registered after routes
 */
export function errorLogger(
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const employeeId = req.params.employeeId || 
                     req.params.userId || 
                     req.body?.employee_id;

  logger.error(`Unhandled error: ${err.message}`, {
    employee_id: employeeId,
    request_id: req.requestId,
    action: 'unhandled_error',
    error_type: err.name,
  }, err);

  next(err);
}
