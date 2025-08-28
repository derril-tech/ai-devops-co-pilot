import { Injectable, Logger, BadRequestException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Signal } from '../entities/signal.entity';
import { SignalCreateDto } from './dto/signal.dto';
import { NatsService } from '../nats/nats.service';
import { validate } from 'class-validator';

@Injectable()
export class SignalsService {
  private readonly logger = new Logger(SignalsService.name);
  private readonly batchSize = 1000;

  constructor(
    @InjectRepository(Signal)
    private readonly signalRepository: Repository<Signal>,
    private readonly natsService: NatsService,
  ) {}

  async ingestSignals(
    signals: SignalCreateDto[],
    requestId: string,
  ): Promise<{
    processed: number;
    failed: number;
    errors: string[];
    requestId: string;
  }> {
    const result = {
      processed: 0,
      failed: 0,
      errors: [],
      requestId,
    };

    if (!signals || signals.length === 0) {
      throw new BadRequestException('No signals provided');
    }

    if (signals.length > 10000) {
      throw new BadRequestException('Too many signals in batch (max 10000)');
    }

    this.logger.log(`[${requestId}] Starting batch processing of ${signals.length} signals`);

    // Validate all signals first
    const validSignals: SignalCreateDto[] = [];
    const validationErrors: string[] = [];

    for (let i = 0; i < signals.length; i++) {
      const signal = signals[i];

      try {
        // Validate signal data
        const errors = await validate(signal);
        if (errors.length > 0) {
          const errorMessages = errors.map(err => `${err.property}: ${Object.values(err.constraints || {}).join(', ')}`);
          validationErrors.push(`Signal ${i}: ${errorMessages.join('; ')}`);
          result.failed++;
          continue;
        }

        // Additional business logic validation
        if (!this.isValidSignal(signal)) {
          validationErrors.push(`Signal ${i}: Invalid signal data`);
          result.failed++;
          continue;
        }

        validSignals.push(signal);

      } catch (error) {
        validationErrors.push(`Signal ${i}: Validation error - ${error.message}`);
        result.failed++;
      }
    }

    // Store validation errors
    result.errors = validationErrors;

    if (validSignals.length === 0) {
      this.logger.warn(`[${requestId}] No valid signals to process`);
      return result;
    }

    // Process signals in batches
    const batches = this.chunkArray(validSignals, this.batchSize);

    for (const batch of batches) {
      try {
        await this.processBatch(batch, requestId);
        result.processed += batch.length;
      } catch (error) {
        this.logger.error(`[${requestId}] Failed to process batch: ${error.message}`, error.stack);
        result.failed += batch.length;
        result.errors.push(`Batch processing failed: ${error.message}`);
      }
    }

    // Publish to NATS for further processing
    await this.publishSignalsToNats(validSignals, requestId);

    this.logger.log(`[${requestId}] Completed processing: ${result.processed} processed, ${result.failed} failed`);

    return result;
  }

  private async processBatch(signals: SignalCreateDto[], requestId: string): Promise<void> {
    // Convert DTOs to entities
    const signalEntities = signals.map(signal => {
      const entity = new Signal();
      entity.orgId = signal.org_id;
      entity.source = signal.source;
      entity.kind = signal.kind;
      entity.ts = signal.ts;
      entity.key = signal.key;
      entity.value = signal.value ? signal.value.toString() : null;
      entity.text = signal.text;
      entity.labels = JSON.stringify(signal.labels || {});
      entity.meta = JSON.stringify(signal.meta || {});
      return entity;
    });

    // Bulk insert with conflict resolution
    await this.signalRepository
      .createQueryBuilder()
      .insert()
      .into(Signal)
      .values(signalEntities)
      .orUpdate(['updated_at'], ['org_id', 'source', 'key', 'ts'])
      .execute();
  }

  private async publishSignalsToNats(signals: SignalCreateDto[], requestId: string): Promise<void> {
    try {
      // Group signals by type for efficient processing
      const signalsByType = this.groupSignalsByType(signals);

      // Publish to appropriate NATS subjects
      for (const [signalType, typeSignals] of Object.entries(signalsByType)) {
        await this.natsService.publish(`signals.${signalType}`, {
          signals: typeSignals,
          requestId,
          timestamp: new Date().toISOString(),
        });
      }

      this.logger.debug(`[${requestId}] Published ${signals.length} signals to NATS`);

    } catch (error) {
      this.logger.error(`[${requestId}] Failed to publish to NATS: ${error.message}`, error.stack);
      // Don't fail the entire operation if NATS publishing fails
    }
  }

  private groupSignalsByType(signals: SignalCreateDto[]): Record<string, SignalCreateDto[]> {
    const groups: Record<string, SignalCreateDto[]> = {};

    for (const signal of signals) {
      const type = signal.kind;
      if (!groups[type]) {
        groups[type] = [];
      }
      groups[type].push(signal);
    }

    return groups;
  }

  private isValidSignal(signal: SignalCreateDto): boolean {
    // Business logic validation
    if (!signal.org_id || !signal.source || !signal.kind || !signal.key) {
      return false;
    }

    if (!['metric', 'log', 'event', 'trace'].includes(signal.kind)) {
      return false;
    }

    // Validate timestamp
    if (!(signal.ts instanceof Date) && isNaN(Date.parse(signal.ts))) {
      return false;
    }

    // Validate value if present
    if (signal.value !== undefined && signal.value !== null) {
      if (typeof signal.value !== 'number') {
        return false;
      }
    }

    return true;
  }

  private chunkArray<T>(array: T[], chunkSize: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += chunkSize) {
      chunks.push(array.slice(i, i + chunkSize));
    }
    return chunks;
  }

  async getSignalStats(orgId: string, hours: number = 24): Promise<{
    total: number;
    byType: Record<string, number>;
    bySource: Record<string, number>;
  }> {
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);

    const [totalResult, typeResult, sourceResult] = await Promise.all([
      this.signalRepository.count({
        where: { orgId, createdAt: { $gte: since } },
      }),
      this.signalRepository
        .createQueryBuilder('signal')
        .select('kind', 'kind')
        .addSelect('COUNT(*)', 'count')
        .where('org_id = :orgId', { orgId })
        .andWhere('created_at >= :since', { since })
        .groupBy('kind')
        .getRawMany(),
      this.signalRepository
        .createQueryBuilder('signal')
        .select('source', 'source')
        .addSelect('COUNT(*)', 'count')
        .where('org_id = :orgId', { orgId })
        .andWhere('created_at >= :since', { since })
        .groupBy('source')
        .getRawMany(),
    ]);

    return {
      total: totalResult,
      byType: Object.fromEntries(typeResult.map(row => [row.kind, parseInt(row.count)])),
      bySource: Object.fromEntries(sourceResult.map(row => [row.source, parseInt(row.count)])),
    };
  }
}
