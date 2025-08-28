import {
  Controller,
  Get,
  Param,
  Query,
  Headers,
  Sse,
  MessageEvent,
  Logger,
  UseGuards,
  HttpException,
  HttpStatus,
} from '@nestjs/common';
import { Observable, fromEvent, map, filter } from 'rxjs';
import { ApiTags, ApiOperation, ApiResponse, ApiParam, ApiQuery } from '@nestjs/swagger';
import { SseService } from './sse.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';

@ApiTags('Server-Sent Events')
@Controller('v1/sse')
@UseGuards(JwtAuthGuard)
export class SseController {
  private readonly logger = new Logger(SseController.name);

  constructor(private readonly sseService: SseService) {}

  @Get('incidents/:incidentId')
  @Sse()
  @ApiOperation({ summary: 'Stream real-time updates for an incident' })
  @ApiResponse({ status: 200, description: 'SSE stream established' })
  @ApiResponse({ status: 403, description: 'Access denied' })
  @ApiResponse({ status: 404, description: 'Incident not found' })
  @ApiParam({
    name: 'incidentId',
    description: 'Incident ID',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  @ApiQuery({
    name: 'types',
    description: 'Event types to subscribe to',
    required: false,
    example: 'signal,evidence,hypothesis',
  })
  async incidentUpdates(
    @Param('incidentId') incidentId: string,
    @Query('types') eventTypes: string = 'all',
    @Headers('x-org-id') orgId: string,
    @Headers('x-user-id') userId: string,
  ): Promise<Observable<MessageEvent>> {
    try {
      this.logger.log(`Establishing SSE stream for incident ${incidentId}, user ${userId}`);

      // Validate access to incident
      const hasAccess = await this.sseService.validateIncidentAccess(incidentId, orgId);
      if (!hasAccess) {
        throw new HttpException('Access denied to incident', HttpStatus.FORBIDDEN);
      }

      // Parse event types
      const types = eventTypes === 'all' ? [] : eventTypes.split(',');

      // Create SSE stream
      return this.sseService.createIncidentStream(incidentId, types).pipe(
        map((event) => ({
          data: JSON.stringify({
            id: event.id,
            type: event.type,
            data: event.data,
            timestamp: event.timestamp,
            incidentId,
          }),
          event: event.type,
          id: event.id,
        })),
        filter(() => true), // Keep all events (additional filtering can be added here)
      );

    } catch (error) {
      this.logger.error(`Failed to establish SSE stream: ${error.message}`, error.stack);
      throw error;
    }
  }

  @Get('signals')
  @Sse()
  @ApiOperation({ summary: 'Stream real-time signal updates' })
  @ApiResponse({ status: 200, description: 'SSE stream established' })
  @ApiResponse({ status: 403, description: 'Access denied' })
  @ApiQuery({
    name: 'sources',
    description: 'Signal sources to subscribe to',
    required: false,
    example: 'prometheus,loki',
  })
  @ApiQuery({
    name: 'kinds',
    description: 'Signal kinds to subscribe to',
    required: false,
    example: 'metric,log,event',
  })
  async signalUpdates(
    @Query('sources') sources: string = 'all',
    @Query('kinds') kinds: string = 'all',
    @Headers('x-org-id') orgId: string,
    @Headers('x-user-id') userId: string,
  ): Promise<Observable<MessageEvent>> {
    try {
      this.logger.log(`Establishing signal SSE stream for org ${orgId}, user ${userId}`);

      // Parse filters
      const sourceFilter = sources === 'all' ? [] : sources.split(',');
      const kindFilter = kinds === 'all' ? [] : kinds.split(',');

      // Create SSE stream
      return this.sseService.createSignalStream(orgId, sourceFilter, kindFilter).pipe(
        map((signal) => ({
          data: JSON.stringify({
            id: signal.id,
            source: signal.source,
            kind: signal.kind,
            key: signal.key,
            value: signal.value,
            text: signal.text,
            labels: signal.labels,
            timestamp: signal.ts,
            orgId,
          }),
          event: 'signal',
          id: signal.id,
        })),
      );

    } catch (error) {
      this.logger.error(`Failed to establish signal SSE stream: ${error.message}`, error.stack);
      throw error;
    }
  }

  @Get('connectors/:connectorId')
  @Sse()
  @ApiOperation({ summary: 'Stream real-time connector status updates' })
  @ApiResponse({ status: 200, description: 'SSE stream established' })
  @ApiResponse({ status: 403, description: 'Access denied' })
  @ApiParam({
    name: 'connectorId',
    description: 'Connector ID',
    example: '550e8400-e29b-41d4-a716-446655440000',
  })
  async connectorUpdates(
    @Param('connectorId') connectorId: string,
    @Headers('x-org-id') orgId: string,
    @Headers('x-user-id') userId: string,
  ): Promise<Observable<MessageEvent>> {
    try {
      this.logger.log(`Establishing connector SSE stream for connector ${connectorId}`);

      // Validate access to connector
      const hasAccess = await this.sseService.validateConnectorAccess(connectorId, orgId);
      if (!hasAccess) {
        throw new HttpException('Access denied to connector', HttpStatus.FORBIDDEN);
      }

      // Create SSE stream
      return this.sseService.createConnectorStream(connectorId).pipe(
        map((event) => ({
          data: JSON.stringify({
            id: event.id,
            type: event.type,
            status: event.status,
            message: event.message,
            timestamp: event.timestamp,
            connectorId,
          }),
          event: 'connector',
          id: event.id,
        })),
      );

    } catch (error) {
      this.logger.error(`Failed to establish connector SSE stream: ${error.message}`, error.stack);
      throw error;
    }
  }
}
