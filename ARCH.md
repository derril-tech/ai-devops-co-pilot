# AI DevOps Copilot — ARCH.md

## 1) Topology
- **Frontend/BFF**: Next.js 14 (App Router) on Vercel (SSR for timelines, Server Actions for webhooks/exports).
- **API Gateway**: NestJS (Node 20) — REST `/v1`, OpenAPI 3.1, Zod, Problem+JSON, RBAC (Casbin), RLS, Idempotency‑Key, Request‑ID (ULID).
- **Workers** (Python 3.11 + FastAPI control): `ingest-worker`, `detect-worker`, `correlate-worker`, `rca-worker`, `rag-worker`, `remediate-worker`, `verify-worker`, `export-worker`.
- **Event Bus**: NATS topics — `signal.ingest`, `signal.detect`, `signal.correlate`, `rca.make`, `rag.answer`, `fix.plan`, `fix.preflight`, `fix.execute`, `verify.run`, `export.make`; Redis Streams for progress/rehydration.
- **Data**: Postgres 16 (+ pgvector) for entities & embeddings; ClickHouse for time‑series aggregates; S3/R2 for attachments/exports.
- **Runners**: Isolated command runners (K8s Job agent, self‑hosted runner, or SSH bastion). GitOps PR service (GitHub/GitLab/Bitbucket).

## 2) Data Model (DDL excerpts)
```sql
-- Tenancy
CREATE TABLE orgs (id UUID PRIMARY KEY, name TEXT, plan TEXT DEFAULT 'pro', created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE users (id UUID PRIMARY KEY, org_id UUID, email CITEXT UNIQUE, role TEXT DEFAULT 'member', tz TEXT);

-- Integrations
CREATE TABLE connectors (id UUID PRIMARY KEY, org_id UUID, kind TEXT, config JSONB, scopes TEXT[], status TEXT, created_at TIMESTAMPTZ DEFAULT now());

-- Signals (logs/metrics/events/deploys)
CREATE TABLE signals (
  id UUID PRIMARY KEY, org_id UUID, source TEXT, kind TEXT, ts TIMESTAMPTZ,
  key TEXT, value DOUBLE PRECISION, text TEXT, labels JSONB, meta JSONB
);
CREATE INDEX ON signals(org_id, ts);

-- Topology graph
CREATE TABLE topologies (id UUID PRIMARY KEY, org_id UUID, node TEXT, type TEXT, attrs JSONB);

-- Incidents
CREATE TABLE incidents (
  id UUID PRIMARY KEY, org_id UUID, title TEXT, started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ,
  severity TEXT, status TEXT, service TEXT, slo_impact JSONB, confidence NUMERIC, meta JSONB
);

-- Evidence & Hypotheses
CREATE TABLE evidence (
  id UUID PRIMARY KEY, incident_id UUID, kind TEXT, ref JSONB, excerpt TEXT,
  score NUMERIC, created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE hypotheses (
  id UUID PRIMARY KEY, incident_id UUID, statement TEXT, confidence NUMERIC, support JSONB, counter JSONB
);

-- Knowledge base (RAG)
CREATE TABLE docs (id UUID PRIMARY KEY, org_id UUID, title TEXT, source TEXT, s3_key TEXT, meta JSONB);
CREATE TABLE chunks (id UUID PRIMARY KEY, doc_id UUID, text TEXT, embedding VECTOR(1536), meta JSONB);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- Fix plans & execution
CREATE TABLE fix_plans (
  id UUID PRIMARY KEY, incident_id UUID, summary TEXT, risk_score NUMERIC,
  preflight JSONB, steps JSONB, rollback JSONB, status TEXT
);
CREATE TABLE commands (
  id UUID PRIMARY KEY, plan_id UUID, tool TEXT, script TEXT,
  dryrun_output TEXT, result TEXT, exit_code INT, started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ
);

-- Approvals & Audit
CREATE TABLE approvals (id UUID PRIMARY KEY, plan_id UUID, approver UUID, status TEXT, notes TEXT, created_at TIMESTAMPTZ DEFAULT now());
CREATE TABLE audit_log (id BIGSERIAL PRIMARY KEY, org_id UUID, user_id UUID, action TEXT, target TEXT, meta JSONB, created_at TIMESTAMPTZ DEFAULT now());
```

**Invariants**
- RLS on `org_id` everywhere.
- Every hypothesis cites ≥ 1 `evidence` row.
- Any command requires an **approved** `fix_plan` unless runner is non‑prod.
- Immutable signed runner outputs; all changes audited.

## 3) APIs (selected)
**Connect & Ingest**
- `POST /v1/connectors` `{kind, config}`
- `POST /v1/signals/ingest` `[{...}]` (batch)

**Incidents & RCA**
- `POST /v1/incidents/open` `{title, service}`
- `POST /v1/incidents/{id}/analyze` `{window}` → clusters, anomalies, correlation, hypotheses
- `GET  /v1/incidents/{id}/evidence`

**Knowledge (RAG)**
- `POST /v1/docs/upload`
- `GET  /v1/search?q=...&incident_id=...&k=8` → cited snippets

**Fix & Execute**
- `POST /v1/fix/plan` `{incident_id, mode, strategy?}`
- `POST /v1/fix/preflight` `{plan_id}` (dry‑run, OPA, budget, drift)
- `POST /v1/fix/approve` `{plan_id}`
- `POST /v1/fix/execute` `{plan_id, runner:"gitops|agent|dryrun"}`
- `POST /v1/gitops/pr` `{plan_id, repo, branch, title}`

**Reports & Metrics**
- `POST /v1/exports/incident` `{incident_id, format:"pdf|json"}`
- `GET  /v1/metrics/dora?window=30d`

## 4) Pipelines & Algorithms
### 4.1 Detect
- **Seasonality-aware baselines**: STL decomposition; ESD/KDE spike detection.
- **Windows**: 5m/30m/6h; multi‑resolution to avoid noisy spikes.
- **RED/USE** method views** for svc health (rate, error, duration / utilization, saturation, errors).

### 4.2 Log Clustering
- Parse JSON; grok regex for text logs; extract `err`, `trace`, `stack` fields.
- Build templates via **MinHash** shingling + UMAP for density; semantic clustering (embedding) for variants.
- Output: cluster exemplars with counts, first/last seen, novelty score.

### 4.3 Correlation
- Join anomalies with **deploys/feature flags** ±30m; compute change‑impact likelihood
  `L = f(Δmetric, co‑occurrence, service centrality, prior CFR for service)`.
- Walk **topology graph** to compute blast radius; impacted SLOs & customers.

### 4.4 RCA Hypotheses
- Generate top‑N hypotheses with: statement, confidence, **support** (evidence refs), **counterexamples** (healthy shards, unaffected paths).
- Show “why not X?” to build trust.

### 4.5 Remediation
- **Fix catalog**: pattern → action (restart, rollout pause, config revert, quota raise, JVM flags, DB index hint).
- **Script generator** (templated): kubectl (patch/deploy/rollout), Helm values diff, Terraform plan/apply, Ansible plays, SQL migrations.
- **Preflight**: `--dry-run=server`, OPA policy (allowed namespaces/resources), **SLO budget** gate, **drift** check vs live state.
- **Rollback**: canary/blue‑green; traffic shift; automated verification gates.

### 4.6 Verify
- Canary judge: compare pre/post SLOs, 5xx, P95 latency, error budget burn; halt/rollback on breach with reasoned message.

## 5) Security & Compliance
- SSO (SAML/OIDC), least‑privilege scopes, short‑lived tokens, KMS‑wrapped secrets.
- Policy Engine (OPA/Gatekeeper) on plans/commands; environment segregation (dev/stage/prod).
- Immutable audit; retention & DSR tooling.

## 6) Observability
- **Tracing**: OpenTelemetry spans: `signal.ingest`, `anomaly.detect`, `correlate`, `rca.hypothesize`, `rag.search`, `fix.plan`, `preflight`, `execute`, `verify`, `export`.
- **Metrics (Prometheus)**: pipeline latencies, queue depth, plan success rate, citation coverage, preflight false‑pass/false‑block, runner success/rollback rate.
- **Logging**: structured JSON; redaction for secrets; per‑incident trace IDs.

## 7) Deployment & Scaling
- APIs/Workers: Render/Fly/GKE; autoscale by NATS queue depth. GPU not required.
- ClickHouse: partition by `org_id` + day; materialized views for SLO/DORA.
- Backpressure: DLQs with jittered retries; circuit breakers on upstreams.

## 8) SLOs & Capacity (GA)
- Ingest→anomaly < **5 s p95** at 200k log lines/min burst.
- RCA draft < **30 s p95** on 1h window with 20 services.
- Preflight < **10 s p95**; Verify decision < **60 s p95**.
- Pipeline success ≥ **99%** excl. upstream outages.

## 9) Frontend (key views)
- **Incident Room**: timeline (logs/metrics/deploys), hypothesis panel, topology graph, evidence cards.
- **Fix Wizard**: plan → preflight → approval → execute/PR; script pane with diffs; canary monitor.
- **Reports**: incident PDF; DORA dashboard; SLO burn charts; audit packs.

## 10) Rollout
- **Alpha**: single cluster, PR‑only changes, 2 design partners.
- **Beta**: add runners (non‑prod), OPA policies v1, incident PDFs, DORA.
- **GA**: prod runners gated, canary judge, SSO/SAML, pen test.
```
