"""
Pydantic models for API requests and responses
"""
from datetime import datetime
from typing import Optional, Any, Dict, List
from uuid import UUID
from pydantic import BaseModel, Field, validator
from .base import BasePydanticModel


# Organization models
class OrganizationCreate(BasePydanticModel):
    name: str = Field(..., min_length=1, max_length=255)
    plan: str = Field(default="pro", regex="^(free|pro|enterprise)$")


class OrganizationResponse(BasePydanticModel):
    id: UUID
    name: str
    plan: str
    created_at: datetime


# User models
class UserCreate(BasePydanticModel):
    org_id: UUID
    email: str = Field(..., regex=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    password: str = Field(..., min_length=8)
    role: str = Field(default="member", regex="^(admin|member|viewer)$")
    tz: str = Field(default="UTC")


class UserResponse(BasePydanticModel):
    id: UUID
    org_id: UUID
    email: str
    role: str
    tz: str
    created_at: datetime


# Connector models
class ConnectorCreate(BasePydanticModel):
    org_id: UUID
    kind: str = Field(..., regex="^(prometheus|loki|elasticsearch|github|argocd|pagerduty)$")
    name: str = Field(..., min_length=1, max_length=255)
    config: Dict[str, Any] = Field(default_factory=dict)
    scopes: List[str] = Field(default_factory=list)


class ConnectorResponse(BasePydanticModel):
    id: UUID
    org_id: UUID
    kind: str
    name: str
    config: Dict[str, Any]
    scopes: List[str]
    status: str
    created_at: datetime


# Signal models
class SignalCreate(BasePydanticModel):
    org_id: UUID
    source: str
    kind: str = Field(..., regex="^(metric|log|event|trace)$")
    ts: datetime
    key: str
    value: Optional[float] = None
    text: Optional[str] = None
    labels: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SignalBatchCreate(BasePydanticModel):
    signals: List[SignalCreate] = Field(..., max_items=1000)


class SignalResponse(BasePydanticModel):
    id: UUID
    org_id: UUID
    source: str
    kind: str
    ts: datetime
    key: str
    value: Optional[float]
    text: Optional[str]
    labels: Dict[str, Any]
    meta: Dict[str, Any]


# Incident models
class IncidentCreate(BasePydanticModel):
    org_id: UUID
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = None
    service: Optional[str] = None
    severity: str = Field(default="medium", regex="^(low|medium|high|critical)$")


class IncidentResponse(BasePydanticModel):
    id: UUID
    org_id: UUID
    title: str
    description: Optional[str]
    started_at: datetime
    ended_at: Optional[datetime]
    severity: str
    status: str
    service: Optional[str]
    slo_impact: Dict[str, Any]
    confidence: float
    meta: Dict[str, Any]


class IncidentAnalyze(BasePydanticModel):
    window_minutes: int = Field(default=60, ge=1, le=1440)


# Evidence models
class EvidenceResponse(BasePydanticModel):
    id: UUID
    incident_id: UUID
    kind: str
    ref: Dict[str, Any]
    excerpt: Optional[str]
    score: float


# Hypothesis models
class HypothesisResponse(BasePydanticModel):
    id: UUID
    incident_id: UUID
    statement: str
    confidence: float
    support: List[Dict[str, Any]]
    counter: List[Dict[str, Any]]


# Fix Plan models
class FixPlanCreate(BasePydanticModel):
    incident_id: UUID
    mode: str = Field(default="auto", regex="^(auto|manual)$")
    strategy: Optional[str] = None


class FixStep(BasePydanticModel):
    tool: str = Field(..., regex="^(kubectl|helm|terraform|ansible|sql|shell)$")
    script: str
    description: str
    risk_level: str = Field(default="medium", regex="^(low|medium|high)$")


class FixPlanResponse(BasePydanticModel):
    id: UUID
    incident_id: UUID
    summary: str
    risk_score: float
    preflight: Dict[str, Any]
    steps: List[FixStep]
    rollback: Dict[str, Any]
    status: str


class FixPlanApproval(BasePydanticModel):
    plan_id: UUID
    approved: bool
    notes: Optional[str] = None


class FixPlanExecute(BasePydanticModel):
    plan_id: UUID
    runner: str = Field(default="agent", regex="^(agent|gitops|dryrun)$")


# Knowledge Base models
class DocumentUpload(BasePydanticModel):
    org_id: UUID
    title: str
    source: str
    content_type: str = Field(default="text", regex="^(text|markdown|pdf|docx)$")


class DocumentResponse(BasePydanticModel):
    id: UUID
    org_id: UUID
    title: str
    source: str
    content_type: str
    s3_key: Optional[str]
    meta: Dict[str, Any]


class SearchQuery(BasePydanticModel):
    q: str = Field(..., min_length=1, max_length=1000)
    incident_id: Optional[UUID] = None
    k: int = Field(default=8, ge=1, le=20)


class SearchResult(BasePydanticModel):
    document_id: UUID
    title: str
    source: str
    text: str
    score: float
    meta: Dict[str, Any]


# API Response wrappers
class APIResponse(BasePydanticModel):
    success: bool = True
    data: Optional[Any] = None
    message: Optional[str] = None
    errors: Optional[List[str]] = None


class PaginatedResponse(APIResponse):
    data: List[Any]
    total: int
    page: int
    page_size: int
    has_more: bool


# Error models
class APIError(BasePydanticModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


# Health check
class HealthResponse(BasePydanticModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, str]
