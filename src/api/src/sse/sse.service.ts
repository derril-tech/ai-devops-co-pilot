import { Injectable, Logger, OnModuleDestroy } from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { Observable, Subject, filter, map } from 'rxjs';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Incident } from '../entities/incident.entity';
import { Connector } from '../entities/connector.entity';
import { Signal } from '../entities/signal.entity';
import { NatsService } from '../nats/nats.service';

export interface SseEvent {
  id: string;
  type: string;
  data: any;
  timestamp: Date;
}

export interface IncidentEvent extends SseEvent {
  incidentId: string;
}

export interface ConnectorEvent extends SseEvent {
  connectorId: string;
  status: string;
  message: string;
}

@Injectable()
export class SseService implements OnModuleDestroy {
  private readonly logger = new Logger(SseService.name);
  private readonly subjects = new Map<string, Subject<SseEvent>>();
  private readonly activeStreams = new Set<string>();

  constructor(
    private readonly eventEmitter: EventEmitter2,
    private readonly natsService: NatsService,
    @InjectRepository(Incident)
    private readonly incidentRepository: Repository<Incident>,
    @InjectRepository(Connector)
    private readonly connectorRepository: Repository<Connector>,
    @InjectRepository(Signal)
    private readonly signalRepository: Repository<Signal>,
  ) {
    this.setupNatsSubscriptions();
    this.setupEventListeners();
  }

  onModuleDestroy() {
    // Clean up all subjects
    for (const subject of this.subjects.values()) {
      subject.complete();
    }
    this.subjects.clear();
  }

  async validateIncidentAccess(incidentId: string, orgId: string): Promise<boolean> {
    try {
      const incident = await this.incidentRepository.findOne({
        where: { id: incidentId, orgId },
      });
      return !!incident;
    } catch (error) {
      this.logger.error(`Failed to validate incident access: ${error.message}`);
      return false;
    }
  }

  async validateConnectorAccess(connectorId: string, orgId: string): Promise<boolean> {
    try {
      const connector = await this.connectorRepository.findOne({
        where: { id: connectorId, orgId },
      });
      return !!connector;
    } catch (error) {
      this.logger.error(`Failed to validate connector access: ${error.message}`);
      return false;
    }
  }

  createIncidentStream(incidentId: string, eventTypes: string[] = []): Observable<SseEvent> {
    const streamKey = `incident:${incidentId}`;

    if (!this.subjects.has(streamKey)) {
      this.subjects.set(streamKey, new Subject<SseEvent>());
      this.activeStreams.add(streamKey);
    }

    const subject = this.subjects.get(streamKey)!;

    return subject.pipe(
      filter((event) => {
        if (eventTypes.length === 0) return true;
        return eventTypes.includes(event.type);
      }),
    );
  }

  createSignalStream(orgId: string, sourceFilter: string[] = [], kindFilter: string[] = []): Observable<Signal> {
    const streamKey = `signals:${orgId}`;

    if (!this.subjects.has(streamKey)) {
      this.subjects.set(streamKey, new Subject<Signal>());
      this.activeStreams.add(streamKey);
    }

    const subject = this.subjects.get(streamKey) as Subject<Signal>;

    return subject.pipe(
      filter((signal) => {
        // Apply source filter
        if (sourceFilter.length > 0 && !sourceFilter.some(source => signal.source.includes(source))) {
          return false;
        }

        // Apply kind filter
        if (kindFilter.length > 0 && !kindFilter.includes(signal.kind)) {
          return false;
        }

        return true;
      }),
    );
  }

  createConnectorStream(connectorId: string): Observable<ConnectorEvent> {
    const streamKey = `connector:${connectorId}`;

    if (!this.subjects.has(streamKey)) {
      this.subjects.set(streamKey, new Subject<ConnectorEvent>());
      this.activeStreams.add(streamKey);
    }

    return this.subjects.get(streamKey) as Observable<ConnectorEvent>;
  }

  private setupNatsSubscriptions(): void {
    // Subscribe to incident events
    this.natsService.subscribe('incidents.*', (msg) => {
      try {
        const event = JSON.parse(msg.data.toString());
        this.handleIncidentEvent(event);
      } catch (error) {
        this.logger.error(`Failed to process incident event: ${error.message}`);
      }
    });

    // Subscribe to signal events
    this.natsService.subscribe('signals.*', (msg) => {
      try {
        const event = JSON.parse(msg.data.toString());
        this.handleSignalEvent(event);
      } catch (error) {
        this.logger.error(`Failed to process signal event: ${error.message}`);
      }
    });

    // Subscribe to connector events
    this.natsService.subscribe('connectors.*', (msg) => {
      try {
        const event = JSON.parse(msg.data.toString());
        this.handleConnectorEvent(event);
      } catch (error) {
        this.logger.error(`Failed to process connector event: ${error.message}`);
      }
    });
  }

  private setupEventListeners(): void {
    // Listen for internal events
    this.eventEmitter.on('incident.updated', (event) => {
      this.handleIncidentEvent(event);
    });

    this.eventEmitter.on('signal.ingested', (event) => {
      this.handleSignalEvent(event);
    });

    this.eventEmitter.on('connector.status', (event) => {
      this.handleConnectorEvent(event);
    });
  }

  private handleIncidentEvent(event: any): void {
    const incidentId = event.incidentId || event.id;
    const streamKey = `incident:${incidentId}`;

    if (this.subjects.has(streamKey)) {
      const sseEvent: IncidentEvent = {
        id: event.id || `${incidentId}-${Date.now()}`,
        type: event.type || 'updated',
        data: event.data || event,
        timestamp: new Date(),
        incidentId,
      };

      this.subjects.get(streamKey)!.next(sseEvent);
    }
  }

  private handleSignalEvent(event: any): void {
    if (!event.signals || !Array.isArray(event.signals)) {
      return;
    }

    // Group signals by org
    const signalsByOrg = new Map<string, Signal[]>();

    for (const signalData of event.signals) {
      const orgId = signalData.org_id;
      if (!signalsByOrg.has(orgId)) {
        signalsByOrg.set(orgId, []);
      }

      // Convert to Signal entity format
      const signal: Signal = {
        id: signalData.id || `signal-${Date.now()}-${Math.random()}`,
        orgId,
        source: signalData.source,
        kind: signalData.kind,
        ts: new Date(signalData.ts),
        key: signalData.key,
        value: signalData.value?.toString(),
        text: signalData.text,
        labels: signalData.labels || {},
        meta: signalData.meta || {},
        createdAt: new Date(),
        updatedAt: new Date(),
      };

      signalsByOrg.get(orgId)!.push(signal);
    }

    // Emit to each org's stream
    for (const [orgId, signals] of signalsByOrg) {
      const streamKey = `signals:${orgId}`;

      if (this.subjects.has(streamKey)) {
        const subject = this.subjects.get(streamKey) as Subject<Signal>;

        for (const signal of signals) {
          subject.next(signal);
        }
      }
    }
  }

  private handleConnectorEvent(event: any): void {
    const connectorId = event.connectorId || event.id;
    const streamKey = `connector:${connectorId}`;

    if (this.subjects.has(streamKey)) {
      const sseEvent: ConnectorEvent = {
        id: event.id || `${connectorId}-${Date.now()}`,
        type: event.type || 'status',
        data: event.data || event,
        timestamp: new Date(),
        connectorId,
        status: event.status || 'unknown',
        message: event.message || '',
      };

      this.subjects.get(streamKey)!.next(sseEvent);
    }
  }

  // Public methods for emitting events
  emitIncidentEvent(incidentId: string, eventType: string, data: any): void {
    const event = {
      id: `${incidentId}-${Date.now()}`,
      type: eventType,
      data,
      timestamp: new Date(),
      incidentId,
    };

    this.handleIncidentEvent(event);
  }

  emitConnectorEvent(connectorId: string, eventType: string, status: string, message: string, data?: any): void {
    const event = {
      id: `${connectorId}-${Date.now()}`,
      type: eventType,
      status,
      message,
      data: data || {},
      timestamp: new Date(),
      connectorId,
    };

    this.handleConnectorEvent(event);
  }

  // Cleanup methods
  removeStream(streamKey: string): void {
    if (this.subjects.has(streamKey)) {
      const subject = this.subjects.get(streamKey)!;
      subject.complete();
      this.subjects.delete(streamKey);
      this.activeStreams.delete(streamKey);
    }
  }

  getActiveStreamsCount(): number {
    return this.activeStreams.size;
  }
}
