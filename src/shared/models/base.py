"""
Base models and utilities for the AI DevOps Copilot
"""
from datetime import datetime
from typing import Optional, Any, Dict
from uuid import uuid4
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, String, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID

Base = declarative_base()


class BaseDBModel(Base):
    """Base SQLAlchemy model with common fields"""
    __abstract__ = True

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BasePydanticModel(BaseModel):
    """Base Pydantic model with common configuration"""
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Organization(BaseDBModel):
    """Organization model"""
    __tablename__ = "orgs"

    name = Column(String, nullable=False)
    plan = Column(String, default="pro")


class User(BaseDBModel):
    """User model"""
    __tablename__ = "users"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    email = Column(String, nullable=False, unique=True)
    password_hash = Column(String)
    role = Column(String, default="member")
    tz = Column(String, default="UTC")


class Connector(BaseDBModel):
    """External system connector model"""
    __tablename__ = "connectors"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    kind = Column(String, nullable=False)
    name = Column(String, nullable=False)
    config = Column(String, default="{}")  # JSON string
    scopes = Column(String, default="[]")  # JSON array string
    status = Column(String, default="pending")


class Signal(BaseDBModel):
    """Telemetry signal model (metrics, logs, events)"""
    __tablename__ = "signals"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    source = Column(String, nullable=False)
    kind = Column(String, nullable=False)  # metric, log, event, trace
    ts = Column(DateTime(timezone=True), nullable=False)
    key = Column(String, nullable=False)
    value = Column(String)  # Float stored as string for precision
    text = Column(String)  # Log message, event description
    labels = Column(String, default="{}")  # JSON string
    meta = Column(String, default="{}")  # JSON string


class Topology(BaseDBModel):
    """Service topology node model"""
    __tablename__ = "topologies"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    node = Column(String, nullable=False)
    type = Column(String, nullable=False)
    attrs = Column(String, default="{}")  # JSON string


class TopologyEdge(BaseDBModel):
    """Service topology edge model"""
    __tablename__ = "topology_edges"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    source_node = Column(String, nullable=False)
    target_node = Column(String, nullable=False)
    relationship = Column(String, nullable=False)
    attrs = Column(String, default="{}")  # JSON string


class Incident(BaseDBModel):
    """Incident model"""
    __tablename__ = "incidents"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    title = Column(String, nullable=False)
    description = Column(String)
    started_at = Column(DateTime(timezone=True), nullable=False)
    ended_at = Column(DateTime(timezone=True))
    severity = Column(String, default="medium")
    status = Column(String, default="open")
    service = Column(String)
    slo_impact = Column(String, default="{}")  # JSON string
    confidence = Column(String, default="0")  # Decimal stored as string
    meta = Column(String, default="{}")  # JSON string


class Evidence(BaseDBModel):
    """Evidence model for incidents"""
    __tablename__ = "evidence"

    incident_id = Column(UUID(as_uuid=True), nullable=False)
    kind = Column(String, nullable=False)
    ref = Column(String, nullable=False)  # JSON string
    excerpt = Column(String)
    score = Column(String, default="0")  # Decimal stored as string


class Hypothesis(BaseDBModel):
    """Hypothesis model for incidents"""
    __tablename__ = "hypotheses"

    incident_id = Column(UUID(as_uuid=True), nullable=False)
    statement = Column(String, nullable=False)
    confidence = Column(String, default="0")  # Decimal stored as string
    support = Column(String, default="[]")  # JSON array string
    counter = Column(String, default="[]")  # JSON array string


class Document(BaseDBModel):
    """Knowledge base document model"""
    __tablename__ = "docs"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    title = Column(String, nullable=False)
    source = Column(String, nullable=False)
    content_type = Column(String, default="text")
    s3_key = Column(String)
    meta = Column(String, default="{}")  # JSON string


class Chunk(BaseDBModel):
    """Document chunk model for RAG"""
    __tablename__ = "chunks"

    doc_id = Column(UUID(as_uuid=True), nullable=False)
    text = Column(String, nullable=False)
    embedding = Column(String)  # Vector stored as string
    meta = Column(String, default="{}")  # JSON string


class FixPlan(BaseDBModel):
    """Fix plan model"""
    __tablename__ = "fix_plans"

    incident_id = Column(UUID(as_uuid=True), nullable=False)
    summary = Column(String, nullable=False)
    risk_score = Column(String, default="0")  # Decimal stored as string
    preflight = Column(String, default="{}")  # JSON string
    steps = Column(String, nullable=False)  # JSON string
    rollback = Column(String, default="{}")  # JSON string
    status = Column(String, default="draft")


class Command(BaseDBModel):
    """Command execution model"""
    __tablename__ = "commands"

    plan_id = Column(UUID(as_uuid=True), nullable=False)
    tool = Column(String, nullable=False)
    script = Column(String, nullable=False)
    dryrun_output = Column(String)
    result = Column(String)
    exit_code = Column(String)  # Integer stored as string
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))


class Approval(BaseDBModel):
    """Approval model for fix plans"""
    __tablename__ = "approvals"

    plan_id = Column(UUID(as_uuid=True), nullable=False)
    approver = Column(UUID(as_uuid=True), nullable=False)
    status = Column(String, default="pending")
    notes = Column(String)


class AuditLog(BaseDBModel):
    """Audit log model"""
    __tablename__ = "audit_log"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True))
    action = Column(String, nullable=False)
    target = Column(String, nullable=False)
    target_id = Column(UUID(as_uuid=True))
    meta = Column(String, default="{}")  # JSON string


class IdempotencyKey(BaseDBModel):
    """Idempotency key model"""
    __tablename__ = "idempotency_keys"

    key = Column(String, nullable=False, unique=True)
    org_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True))
    request_hash = Column(String, nullable=False)
    request_data = Column(String)  # JSON string
    response_data = Column(String)  # JSON string


class Secret(BaseDBModel):
    """Secret storage model"""
    __tablename__ = "secrets"

    org_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String, nullable=False)
    encrypted_data = Column(String, nullable=False)
    description = Column(String)
    created_by = Column(UUID(as_uuid=True))
    updated_by = Column(UUID(as_uuid=True))
