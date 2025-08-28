import { ApiProperty, ApiPropertyOptional } from '@nestjs/swagger';
import { Type } from 'class-transformer';
import {
  IsUUID,
  IsString,
  IsEnum,
  IsOptional,
  IsNumber,
  IsObject,
  IsDate,
  IsArray,
  ArrayMaxSize,
  ValidateNested,
} from 'class-validator';

export class SignalCreateDto {
  @ApiProperty({
    description: 'Organization ID',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  @IsUUID()
  org_id: string;

  @ApiProperty({
    description: 'Signal source identifier',
    example: 'prometheus:connector-123',
  })
  @IsString()
  source: string;

  @ApiProperty({
    description: 'Signal type',
    enum: ['metric', 'log', 'event', 'trace'],
    example: 'metric',
  })
  @IsEnum(['metric', 'log', 'event', 'trace'])
  kind: string;

  @ApiProperty({
    description: 'Signal timestamp',
    example: '2023-12-01T10:30:00Z',
  })
  @Type(() => Date)
  @IsDate()
  ts: Date;

  @ApiProperty({
    description: 'Signal key/name',
    example: 'cpu.usage',
  })
  @IsString()
  key: string;

  @ApiPropertyOptional({
    description: 'Numeric value (for metrics)',
    example: 85.5,
  })
  @IsOptional()
  @IsNumber()
  value?: number;

  @ApiPropertyOptional({
    description: 'Text content (for logs/events)',
    example: 'User login successful',
  })
  @IsOptional()
  @IsString()
  text?: string;

  @ApiPropertyOptional({
    description: 'Key-value labels',
    example: { instance: 'web-01', region: 'us-west' },
  })
  @IsOptional()
  @IsObject()
  labels?: Record<string, any>;

  @ApiPropertyOptional({
    description: 'Additional metadata',
    example: { severity: 'info', component: 'auth' },
  })
  @IsOptional()
  @IsObject()
  meta?: Record<string, any>;
}

export class SignalBatchCreateDto {
  @ApiProperty({
    description: 'Array of signals to ingest',
    type: [SignalCreateDto],
  })
  @IsArray()
  @ArrayMaxSize(10000)
  @ValidateNested({ each: true })
  @Type(() => SignalCreateDto)
  signals: SignalCreateDto[];

  @ApiPropertyOptional({
    description: 'Idempotency key for the entire batch',
    example: 'batch-12345-abcdef',
  })
  @IsOptional()
  @IsString()
  idempotency_key?: string;
}

export class SignalResponseDto {
  @ApiProperty({
    description: 'Signal ID',
    example: '550e8400-e29b-41d4-a716-446655440001',
  })
  id: string;

  @ApiProperty({
    description: 'Organization ID',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  org_id: string;

  @ApiProperty({
    description: 'Signal source',
    example: 'prometheus:connector-123',
  })
  source: string;

  @ApiProperty({
    description: 'Signal type',
    example: 'metric',
  })
  kind: string;

  @ApiProperty({
    description: 'Signal timestamp',
    example: '2023-12-01T10:30:00Z',
  })
  ts: Date;

  @ApiProperty({
    description: 'Signal key',
    example: 'cpu.usage',
  })
  key: string;

  @ApiPropertyOptional({
    description: 'Numeric value',
    example: 85.5,
  })
  value?: number;

  @ApiPropertyOptional({
    description: 'Text content',
    example: 'User login successful',
  })
  text?: string;

  @ApiProperty({
    description: 'Key-value labels',
    example: { instance: 'web-01', region: 'us-west' },
  })
  labels: Record<string, any>;

  @ApiProperty({
    description: 'Additional metadata',
    example: { severity: 'info', component: 'auth' },
  })
  meta: Record<string, any>;

  @ApiProperty({
    description: 'Creation timestamp',
    example: '2023-12-01T10:30:00Z',
  })
  created_at: Date;
}

export class SignalBatchResponseDto {
  @ApiProperty({
    description: 'Number of signals processed successfully',
    example: 950,
  })
  processed: number;

  @ApiProperty({
    description: 'Number of signals that failed processing',
    example: 50,
  })
  failed: number;

  @ApiProperty({
    description: 'List of processing errors',
    example: ['Signal 5: Invalid timestamp', 'Signal 10: Missing required field'],
  })
  errors: string[];

  @ApiProperty({
    description: 'Request ID for tracing',
    example: 'req-12345-abcdef',
  })
  requestId: string;
}

export class SignalStatsDto {
  @ApiProperty({
    description: 'Total number of signals',
    example: 15420,
  })
  total: number;

  @ApiProperty({
    description: 'Signals grouped by type',
    example: { metric: 12000, log: 2500, event: 920 },
  })
  byType: Record<string, number>;

  @ApiProperty({
    description: 'Signals grouped by source',
    example: { 'prometheus:prod': 8000, 'loki:prod': 4500, 'github:main': 2920 },
  })
  bySource: Record<string, number>;
}
