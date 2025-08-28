import {
  Entity,
  Column,
  PrimaryGeneratedColumn,
  CreateDateColumn,
  UpdateDateColumn,
  Index,
} from 'typeorm';
import { ApiProperty } from '@nestjs/swagger';

@Entity('signals')
@Index(['orgId', 'ts'])
@Index(['orgId', 'source'])
export class Signal {
  @ApiProperty({
    description: 'Signal ID',
    example: '550e8400-e29b-41d4-a716-446655440001',
  })
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @ApiProperty({
    description: 'Organization ID',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  @Column('uuid')
  @Index()
  orgId: string;

  @ApiProperty({
    description: 'Signal source',
    example: 'prometheus:connector-123',
  })
  @Column()
  @Index()
  source: string;

  @ApiProperty({
    description: 'Signal type',
    example: 'metric',
  })
  @Column()
  @Index()
  kind: string;

  @ApiProperty({
    description: 'Signal timestamp',
    example: '2023-12-01T10:30:00Z',
  })
  @Column('timestamptz')
  @Index()
  ts: Date;

  @ApiProperty({
    description: 'Signal key',
    example: 'cpu.usage',
  })
  @Column()
  key: string;

  @ApiProperty({
    description: 'Numeric value (stored as string for precision)',
    example: '85.5',
  })
  @Column({ nullable: true })
  value: string;

  @ApiProperty({
    description: 'Text content',
    example: 'User login successful',
  })
  @Column({ type: 'text', nullable: true })
  text: string;

  @ApiProperty({
    description: 'Key-value labels (JSON)',
    example: '{"instance": "web-01", "region": "us-west"}',
  })
  @Column({ type: 'jsonb', default: '{}' })
  labels: Record<string, any>;

  @ApiProperty({
    description: 'Additional metadata (JSON)',
    example: '{"severity": "info", "component": "auth"}',
  })
  @Column({ type: 'jsonb', default: '{}' })
  meta: Record<string, any>;

  @ApiProperty({
    description: 'Creation timestamp',
    example: '2023-12-01T10:30:00Z',
  })
  @CreateDateColumn({ type: 'timestamptz' })
  createdAt: Date;

  @ApiProperty({
    description: 'Last update timestamp',
    example: '2023-12-01T10:30:00Z',
  })
  @UpdateDateColumn({ type: 'timestamptz' })
  updatedAt: Date;
}
