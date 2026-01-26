import dgram from 'dgram';
import os from 'os';

/**
 * Graylog Logger Service
 * 
 * Uses GELF (Graylog Extended Log Format) over UDP.
 * 
 * NOTE: Graylog host/port configuration is placeholder until actual
 * Graylog instance details are confirmed. The GELF protocol is standard
 * and should work with any Graylog setup.
 * 
 * Set GRAYLOG_HOST environment variable to enable. Without it, logs
 * are output to console only.
 */

/**
 * Log levels matching Graylog/syslog levels
 */
export enum LogLevel {
  EMERGENCY = 0,
  ALERT = 1,
  CRITICAL = 2,
  ERROR = 3,
  WARNING = 4,
  NOTICE = 5,
  INFO = 6,
  DEBUG = 7,
}

/**
 * GELF (Graylog Extended Log Format) message structure
 */
interface GelfMessage {
  version: '1.1';
  host: string;
  short_message: string;
  full_message?: string;
  timestamp: number;
  level: LogLevel;
  facility?: string;
  // Custom fields (must be prefixed with _)
  [key: `_${string}`]: string | number | boolean | undefined;
}

/**
 * Logger configuration
 */
export interface LoggerConfig {
  /** Graylog server host */
  graylogHost?: string;
  /** Graylog GELF UDP port (default: 12201) */
  graylogPort?: number;
  /** Application/service name */
  facility?: string;
  /** Minimum log level to send */
  minLevel?: LogLevel;
  /** Enable console output */
  consoleOutput?: boolean;
  /** Environment (dev, staging, prod) */
  environment?: string;
}

/**
 * Enrollment-specific log context
 */
export interface EnrollmentLogContext {
  employee_id?: string;
  enrollmentStatus?: string;
  detailedStatus?: string;
  action?: string;
  duration_ms?: number;
  devices_notified?: number;
  error_type?: string;
  request_id?: string;
  tenant_id?: string;
}

/**
 * Logger service with Graylog integration
 */
export class Logger {
  private socket: dgram.Socket | null = null;
  private config: Required<LoggerConfig>;
  private hostname: string;

  constructor(config: LoggerConfig = {}) {
    this.config = {
      graylogHost: config.graylogHost || '',
      graylogPort: config.graylogPort || 12201,
      facility: config.facility || 'enrollment-api',
      minLevel: config.minLevel ?? LogLevel.INFO,
      consoleOutput: config.consoleOutput ?? true,
      environment: config.environment || process.env.NODE_ENV || 'development',
    };

    this.hostname = os.hostname();

    // Initialize UDP socket for Graylog if configured
    if (this.config.graylogHost) {
      this.socket = dgram.createSocket('udp4');
      this.socket.on('error', (err) => {
        console.error('Graylog socket error:', err);
      });
    }
  }

  /**
   * Close the UDP socket
   */
  close(): void {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }

  /**
   * Send log to Graylog via GELF UDP
   */
  private sendToGraylog(message: GelfMessage): void {
    if (!this.socket || !this.config.graylogHost) {
      return;
    }

    const payload = Buffer.from(JSON.stringify(message));

    this.socket.send(
      payload,
      0,
      payload.length,
      this.config.graylogPort,
      this.config.graylogHost,
      (err) => {
        if (err) {
          console.error('Failed to send to Graylog:', err);
        }
      }
    );
  }

  /**
   * Format console output with color
   */
  private formatConsole(level: LogLevel, message: string, context?: EnrollmentLogContext): string {
    const timestamp = new Date().toISOString();
    const levelNames = ['EMERG', 'ALERT', 'CRIT', 'ERROR', 'WARN', 'NOTICE', 'INFO', 'DEBUG'];
    const levelName = levelNames[level] || 'UNKNOWN';
    
    let output = `[${timestamp}] [${levelName}] ${message}`;
    
    if (context && Object.keys(context).length > 0) {
      output += ` | ${JSON.stringify(context)}`;
    }
    
    return output;
  }

  /**
   * Core logging method
   */
  private log(
    level: LogLevel,
    message: string,
    context?: EnrollmentLogContext,
    fullMessage?: string
  ): void {
    // Check minimum level
    if (level > this.config.minLevel) {
      return;
    }

    // Console output
    if (this.config.consoleOutput) {
      const formatted = this.formatConsole(level, message, context);
      
      switch (level) {
        case LogLevel.EMERGENCY:
        case LogLevel.ALERT:
        case LogLevel.CRITICAL:
        case LogLevel.ERROR:
          console.error(formatted);
          break;
        case LogLevel.WARNING:
          console.warn(formatted);
          break;
        case LogLevel.DEBUG:
          console.debug(formatted);
          break;
        default:
          console.log(formatted);
      }
    }

    // Build GELF message
    const gelfMessage: GelfMessage = {
      version: '1.1',
      host: this.hostname,
      short_message: message,
      full_message: fullMessage,
      timestamp: Date.now() / 1000, // Unix timestamp with ms precision
      level,
      facility: this.config.facility,
      _environment: this.config.environment,
    };

    // Add context fields with _ prefix
    if (context) {
      if (context.employee_id) gelfMessage._employee_id = context.employee_id;
      if (context.enrollmentStatus) gelfMessage._enrollment_status = context.enrollmentStatus;
      if (context.detailedStatus) gelfMessage._detailed_status = context.detailedStatus;
      if (context.action) gelfMessage._action = context.action;
      if (context.duration_ms !== undefined) gelfMessage._duration_ms = context.duration_ms;
      if (context.devices_notified !== undefined) gelfMessage._devices_notified = context.devices_notified;
      if (context.error_type) gelfMessage._error_type = context.error_type;
      if (context.request_id) gelfMessage._request_id = context.request_id;
      if (context.tenant_id) gelfMessage._tenant_id = context.tenant_id;
    }

    // Send to Graylog
    this.sendToGraylog(gelfMessage);
  }

  // Convenience methods

  emergency(message: string, context?: EnrollmentLogContext): void {
    this.log(LogLevel.EMERGENCY, message, context);
  }

  alert(message: string, context?: EnrollmentLogContext): void {
    this.log(LogLevel.ALERT, message, context);
  }

  critical(message: string, context?: EnrollmentLogContext): void {
    this.log(LogLevel.CRITICAL, message, context);
  }

  error(message: string, context?: EnrollmentLogContext, error?: Error): void {
    this.log(LogLevel.ERROR, message, context, error?.stack);
  }

  warning(message: string, context?: EnrollmentLogContext): void {
    this.log(LogLevel.WARNING, message, context);
  }

  warn(message: string, context?: EnrollmentLogContext): void {
    this.warning(message, context);
  }

  notice(message: string, context?: EnrollmentLogContext): void {
    this.log(LogLevel.NOTICE, message, context);
  }

  info(message: string, context?: EnrollmentLogContext): void {
    this.log(LogLevel.INFO, message, context);
  }

  debug(message: string, context?: EnrollmentLogContext): void {
    this.log(LogLevel.DEBUG, message, context);
  }

  // Enrollment-specific logging methods

  /**
   * Log enrollment workflow events
   */
  enrollmentEvent(
    action: string,
    employeeId: string,
    status: string,
    details?: Partial<EnrollmentLogContext>
  ): void {
    this.info(`Enrollment ${action}`, {
      employee_id: employeeId,
      enrollmentStatus: status,
      action,
      ...details,
    });
  }

  /**
   * Log enrollment status change
   */
  statusChange(
    employeeId: string,
    fromStatus: string,
    toStatus: string,
    details?: Partial<EnrollmentLogContext>
  ): void {
    this.info(`Status changed: ${fromStatus} -> ${toStatus}`, {
      employee_id: employeeId,
      enrollmentStatus: toStatus,
      action: 'status_change',
      ...details,
    });
  }

  /**
   * Log IoT publish event
   */
  iotPublish(
    employeeId: string,
    success: boolean,
    devicesNotified: number,
    error?: string
  ): void {
    if (success) {
      this.info('IoT publish successful', {
        employee_id: employeeId,
        action: 'iot_publish',
        devices_notified: devicesNotified,
      });
    } else {
      this.error('IoT publish failed', {
        employee_id: employeeId,
        action: 'iot_publish',
        error_type: 'iot_publish_error',
      });
    }
  }

  /**
   * Log vectorizer processing
   */
  vectorizerEvent(
    employeeId: string,
    action: 'start' | 'complete' | 'error',
    durationMs?: number,
    error?: string
  ): void {
    const context: EnrollmentLogContext = {
      employee_id: employeeId,
      action: `vectorizer_${action}`,
    };

    if (durationMs !== undefined) {
      context.duration_ms = durationMs;
    }

    if (action === 'error') {
      this.error(`Vectorizer ${action}: ${error}`, {
        ...context,
        error_type: 'vectorizer_error',
      });
    } else {
      this.info(`Vectorizer ${action}`, context);
    }
  }

  /**
   * Log API request
   */
  apiRequest(
    method: string,
    path: string,
    statusCode: number,
    durationMs: number,
    context?: Partial<EnrollmentLogContext>
  ): void {
    const level = statusCode >= 500 ? LogLevel.ERROR : 
                  statusCode >= 400 ? LogLevel.WARNING : 
                  LogLevel.INFO;

    this.log(level, `${method} ${path} ${statusCode}`, {
      action: 'api_request',
      duration_ms: durationMs,
      ...context,
    });
  }
}

/**
 * Create logger from environment variables
 */
export function createLogger(): Logger {
  return new Logger({
    graylogHost: process.env.GRAYLOG_HOST,
    graylogPort: process.env.GRAYLOG_PORT ? parseInt(process.env.GRAYLOG_PORT, 10) : 12201,
    facility: process.env.LOG_FACILITY || 'enrollment-api',
    minLevel: process.env.LOG_LEVEL ? parseInt(process.env.LOG_LEVEL, 10) : LogLevel.INFO,
    consoleOutput: process.env.LOG_CONSOLE !== 'false',
    environment: process.env.NODE_ENV || 'development',
  });
}

// Default logger instance
export const logger = createLogger();
