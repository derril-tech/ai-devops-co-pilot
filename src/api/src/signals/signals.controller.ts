import {
  Controller,
  Post,
  Body,
  Headers,
  HttpStatus,
  HttpException,
  UseGuards,
  Logger,
} from '@nestjs/common';
import { SignalsService } from './signals.service';
import { SignalCreateDto, SignalBatchCreateDto } from './dto/signal.dto';
import { ApiTags, ApiOperation, ApiResponse, ApiHeader } from '@nestjs/swagger';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';
import { RequestIdMiddleware } from '../middleware/request-id.middleware';
import { IdempotencyMiddleware } from '../middleware/idempotency.middleware';

@ApiTags('Signals')
@Controller('v1/signals')
@UseGuards(JwtAuthGuard)
export class SignalsController {
  private readonly logger = new Logger(SignalsController.name);

  constructor(
    private readonly signalsService: SignalsService,
    private readonly requestIdMiddleware: RequestIdMiddleware,
    private readonly idempotencyMiddleware: IdempotencyMiddleware,
  ) {}

  @Post('ingest')
  @ApiOperation({ summary: 'Ingest telemetry signals' })
  @ApiResponse({ status: 201, description: 'Signals ingested successfully' })
  @ApiResponse({ status: 400, description: 'Invalid signal data' })
  @ApiResponse({ status: 401, description: 'Unauthorized' })
  @ApiResponse({ status: 409, description: 'Duplicate request (idempotency)' })
  @ApiHeader({
    name: 'Idempotency-Key',
    description: 'Unique key for request idempotency',
    required: false,
  })
  @ApiHeader({
    name: 'X-Request-ID',
    description: 'Request ID for tracing',
    required: false,
  })
  async ingestSignals(
    @Body() signalBatchDto: SignalBatchCreateDto,
    @Headers() headers: Record<string, string>,
  ) {
    const requestId = this.requestIdMiddleware.getRequestId();

    try {
      this.logger.log(`[${requestId}] Processing signal batch with ${signalBatchDto.signals.length} signals`);

      // Check for cached response from idempotency middleware
      const cachedResult = await this.idempotencyMiddleware.processRequest(
        signalBatchDto,
        headers['idempotency-key'],
        headers['x-org-id'],
        headers['x-user-id'],
      );

      if (cachedResult && cachedResult.cached) {
        this.logger.log(`[${requestId}] Returning cached response for idempotency key`);
        return cachedResult.data;
      }

      // Process the signals
      const result = await this.signalsService.ingestSignals(
        signalBatchDto.signals,
        requestId,
      );

      // Store response in idempotency cache if key was provided
      if (headers['idempotency-key']) {
        await this.idempotencyMiddleware.storeResponse(
          headers['idempotency-key'],
          result,
          headers['x-org-id'],
        );
      }

      this.logger.log(`[${requestId}] Successfully ingested ${result.processed} signals`);
      return result;

    } catch (error) {
      this.logger.error(`[${requestId}] Failed to ingest signals: ${error.message}`, error.stack);

      if (error instanceof HttpException) {
        throw error;
      }

      throw new HttpException(
        {
          statusCode: HttpStatus.INTERNAL_SERVER_ERROR,
          message: 'Failed to ingest signals',
          error: 'Internal Server Error',
          requestId,
        },
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }

  @Post('ingest/single')
  @ApiOperation({ summary: 'Ingest a single telemetry signal' })
  @ApiResponse({ status: 201, description: 'Signal ingested successfully' })
  @ApiResponse({ status: 400, description: 'Invalid signal data' })
  @ApiResponse({ status: 401, description: 'Unauthorized' })
  @ApiHeader({
    name: 'Idempotency-Key',
    description: 'Unique key for request idempotency',
    required: false,
  })
  async ingestSingleSignal(
    @Body() signalDto: SignalCreateDto,
    @Headers() headers: Record<string, string>,
  ) {
    const requestId = this.requestIdMiddleware.getRequestId();

    try {
      this.logger.log(`[${requestId}] Processing single signal`);

      // Convert single signal to batch
      const batchDto: SignalBatchCreateDto = {
        signals: [signalDto],
        idempotency_key: headers['idempotency-key'],
      };

      // Reuse batch processing logic
      return await this.ingestSignals(batchDto, headers);

    } catch (error) {
      this.logger.error(`[${requestId}] Failed to ingest single signal: ${error.message}`, error.stack);
      throw error;
    }
  }
}
