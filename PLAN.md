# AI DevOps Copilot — PLAN.md

## 0) One‑liner
Correlate logs, metrics, and deploy history to pinpoint issues—then generate safe, auditable fixes and deployment scripts you can run or PR.

## 1) Vision & Why Now
On‑call teams drown in signals while change velocity rises. Copilot unifies **evidence-first RCA** with **guardrailed automation**, shortening MTTR without risking §prod via preflight and approvals. Built to slot into K8s + GitOps stacks and hybrid clouds.

## 2) Product Pillars
- **Evidence-first**: Every callout cites logs/metrics/config lines and time ranges.
- **Human-in-the-loop**: Read-only discovery by default, approvals required for change.
- **Safe automation**: Dry-runs, OPA policy checks, SLO budget gates, rollback.
- **Git-native**: Prefer PRs to GitOps repos; runners are scoped and auditable.
- **Learning KB**: RAG over runbooks, PMs, ADRs, vendor docs; improves with each incident.

## 3) Outcomes & KPIs
**Customer KPIs**
- ↓ **MTTR** ≥ 35% in 60 days (median).
- ↓ **Change Failure Rate (CFR)** ≥ 20% with preflight gating.
- ↓ On-call triage time ≥ 30% (alert → first hypothesis).
- **Adoption**: ≥ 70% incidents resolved via Copilot-assisted plans.

**Reliability KPIs (DORA)**
- Deploy freq ↑ (no hard target; track trend)
- Lead time stable/↑
- CFR ↓
- MTTR ↓

**Eng SLOs**
- Signal ingest → anomaly < **5 s p95**
- RCA hypothesis draft after alert < **30 s p95**
- Preflight plan < **10 s p95**
- Post-change verify decision < **60 s p95**
- Pipeline success ≥ **99%** (excl. upstream outages)
- RAG citation coverage ≥ **90%** of recommendations

## 4) Target Users
- SRE/Platform running K8s/microservices
- Dev squads on-call
- Infra/Ops teams managing IaC (Terraform/Helm/Ansible)
- Eng leadership tracking reliability KPIs & change risk

## 5) Scope (V1 → V2)
**V1 (GA)**: K8s + Prometheus + Loki/Elastic + GitHub PR (GitOps), ArgoCD/Flux events, PagerDuty import. Anomaly detection, log clustering, change correlation, hypothesis panel, Fix Wizard (kubectl/Helm/Terraform/Ansible), preflight gates (dry-run+OPA+SLO), verify (canary judge), incident PDF, DORA dashboard.
**V2**: Cloud metrics (CloudWatch/GCM), feature flags (LaunchDarkly), SQL fix path, cost/risk optimizer, change risk scoring per commit, auto postmortem draft.

## 6) Roadmap & Milestones
- **M0 — Foundations (3w)**: Connectors (Prom, Loki/Elastic, GitOps), schema, RBAC/RLS, ingest pipeline, SSE streams.
- **M1 — Detect & Correlate (4w)**: Anomaly models (STL/ESD), log clustering (embeddings+MinHash), deploy correlation, blast radius graph.
- **M2 — RCA & Knowledge (3w)**: Hypothesis builder w/ confidence; RAG over runbooks/PMs; cited answers.
- **M3 — Remediation & Preflight (4w)**: Script generator; preflight (dry-run, OPA, budget); approvals & runners; GitOps PRs.
- **M4 — Verify & GA (3w)**: Canary judge, SLO burn charts, DORA widgets, audit/export hardening. **GA**.

## 7) Risks & Mitigations
- **Unsafe automation** → _Mitigate_: read-only by default, approvals, OPA policies, scoped tokens, dry-run & drift checks.
- **Low trust in RCA** → _Mitigate_: show counterevidence, confidence, raw snippets, side-by-side timelines.
- **Connector sprawl/rate limits** → _Mitigate_: backpressure queues, circuit breakers, source-specific quotas.
- **Model hallucination** → _Mitigate_: “No evidence → no recommendation”; force citations; block export if missing.
- **Security concerns** → _Mitigate_: RLS, KMS, short‑lived tokens, immutable audit, env segregation.

## 8) Pricing/Packaging (draft)
- **Team**: core RCA + Fix Wizard (PR-only execution), 50 nodes.
- **Pro**: + runners, OPA policies, SLO gates, DORA analytics, 200 nodes.
- **Enterprise**: SSO/SAML, on‑prem runners, VPC peering, custom retention.

## 9) Launch Checklist
- [ ] Security review (threat model, OPA baseline policies)
- [ ] P0 connectors validated on staging (Prom, Loki/Elastic, GitHub, ArgoCD/Flux)
- [ ] SLO monitors for pipeline latencies
- [ ] DR/backup tested; seed synthetic incidents for demos
- [ ] Docs: setup, runner scopes, runbook authoring
- [ ] GA announcement + 2 design partners case studies

## 10) Definition of Done (GA)
- 4 design partners running in prod, >30 incidents triaged, MTTR ↓ >25% verified.
- RAG recommendation citations ≥ 90%; preflight false‑pass < 1%, false‑block < 5%.
- All execute paths gated by approval + audit logged; pen test passed.
