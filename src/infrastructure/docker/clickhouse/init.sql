-- Create database
CREATE DATABASE IF NOT EXISTS devops_copilot;

-- Use the database
USE devops_copilot;

-- Create signals rollup table for time-series analytics
CREATE TABLE IF NOT EXISTS signals_rollup (
    org_id String,
    source String,
    kind String,
    key String,
    ts DateTime,
    count UInt64,
    sum_value Float64,
    min_value Float64,
    max_value Float64,
    avg_value Float64,
    p50_value Float64,
    p95_value Float64,
    p99_value Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(ts)
ORDER BY (org_id, source, key, ts)
TTL ts + INTERVAL 1 YEAR;

-- Create incidents analytics table
CREATE TABLE IF NOT EXISTS incidents_analytics (
    org_id String,
    incident_id String,
    started_at DateTime,
    ended_at Nullable(DateTime),
    duration_seconds UInt32,
    severity String,
    service String,
    resolution_time_minutes UInt32,
    evidence_count UInt32,
    hypotheses_count UInt32,
    fix_plans_count UInt32
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(started_at)
ORDER BY (org_id, started_at, incident_id);

-- Create DORA metrics table
CREATE TABLE IF NOT EXISTS dora_metrics (
    org_id String,
    date Date,
    deployment_frequency Float64,
    lead_time_for_changes_minutes Float64,
    change_failure_rate Float64,
    mean_time_to_recovery_minutes Float64,
    deployment_count UInt32,
    successful_deployments UInt32,
    failed_deployments UInt32,
    incident_count UInt32,
    resolved_incidents UInt32
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (org_id, date);

-- Create SLO tracking table
CREATE TABLE IF NOT EXISTS slo_tracking (
    org_id String,
    service String,
    slo_name String,
    period_start DateTime,
    period_end DateTime,
    target_percentage Float64,
    actual_percentage Float64,
    is_met UInt8,
    error_budget_used Float64,
    error_budget_remaining Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(period_start)
ORDER BY (org_id, service, slo_name, period_start);

-- Create materialized view for signals rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS signals_rollup_mv
TO signals_rollup
AS SELECT
    org_id,
    source,
    kind,
    key,
    toStartOfMinute(ts) as ts,
    count() as count,
    sum(value) as sum_value,
    min(value) as min_value,
    max(value) as max_value,
    avg(value) as avg_value,
    quantile(0.5)(value) as p50_value,
    quantile(0.95)(value) as p95_value,
    quantile(0.99)(value) as p99_value
FROM signals
GROUP BY org_id, source, kind, key, toStartOfMinute(ts);
