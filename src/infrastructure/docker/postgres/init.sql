-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS vector;

-- Create organizations table
CREATE TABLE IF NOT EXISTS orgs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    plan TEXT DEFAULT 'pro',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    email CITEXT UNIQUE NOT NULL,
    password_hash TEXT,
    role TEXT DEFAULT 'member' CHECK (role IN ('admin', 'member', 'viewer')),
    tz TEXT DEFAULT 'UTC',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create connectors table
CREATE TABLE IF NOT EXISTS connectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('prometheus', 'loki', 'elasticsearch', 'github', 'argocd', 'pagerduty')),
    name TEXT NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    scopes TEXT[] DEFAULT '{}',
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'error', 'disabled')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create signals table (main telemetry data)
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    source TEXT NOT NULL, -- connector id or external source
    kind TEXT NOT NULL CHECK (kind IN ('metric', 'log', 'event', 'trace')),
    ts TIMESTAMPTZ NOT NULL,
    key TEXT NOT NULL, -- metric name, log level, etc.
    value DOUBLE PRECISION,
    text TEXT, -- log message, event description
    labels JSONB DEFAULT '{}',
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create topology table (service graph)
CREATE TABLE IF NOT EXISTS topologies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    node TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('service', 'database', 'cache', 'queue', 'load_balancer', 'ingress')),
    attrs JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create topology edges
CREATE TABLE IF NOT EXISTS topology_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    relationship TEXT NOT NULL CHECK (relationship IN ('calls', 'depends_on', 'writes_to', 'reads_from')),
    attrs JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create incidents table
CREATE TABLE IF NOT EXISTS incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    severity TEXT DEFAULT 'medium' CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status TEXT DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'resolved', 'closed')),
    service TEXT,
    slo_impact JSONB DEFAULT '{}',
    confidence NUMERIC DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 1),
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create evidence table
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('metric', 'log', 'trace', 'deployment', 'config')),
    ref JSONB NOT NULL, -- reference to source data
    excerpt TEXT, -- relevant snippet
    score NUMERIC DEFAULT 0 CHECK (score >= 0 AND score <= 1),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create hypotheses table
CREATE TABLE IF NOT EXISTS hypotheses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    statement TEXT NOT NULL,
    confidence NUMERIC DEFAULT 0 CHECK (confidence >= 0 AND confidence <= 1),
    support JSONB DEFAULT '[]', -- array of evidence IDs
    counter JSONB DEFAULT '[]', -- array of counter-evidence
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create knowledge base tables
CREATE TABLE IF NOT EXISTS docs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    source TEXT NOT NULL, -- URL, file path, or external system
    content_type TEXT DEFAULT 'text' CHECK (content_type IN ('text', 'markdown', 'pdf', 'docx')),
    s3_key TEXT, -- for stored documents
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doc_id UUID NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    embedding VECTOR(1536), -- OpenAI embeddings dimension
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create fix plans table
CREATE TABLE IF NOT EXISTS fix_plans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    risk_score NUMERIC DEFAULT 0 CHECK (risk_score >= 0 AND risk_score <= 1),
    preflight JSONB DEFAULT '{}', -- preflight checks config
    steps JSONB NOT NULL, -- array of fix steps
    rollback JSONB DEFAULT '{}', -- rollback plan
    status TEXT DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'executing', 'completed', 'failed')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Create commands table
CREATE TABLE IF NOT EXISTS commands (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id UUID NOT NULL REFERENCES fix_plans(id) ON DELETE CASCADE,
    tool TEXT NOT NULL CHECK (tool IN ('kubectl', 'helm', 'terraform', 'ansible', 'sql', 'shell')),
    script TEXT NOT NULL,
    dryrun_output TEXT,
    result TEXT,
    exit_code INT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create approvals table
CREATE TABLE IF NOT EXISTS approvals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id UUID NOT NULL REFERENCES fix_plans(id) ON DELETE CASCADE,
    approver UUID NOT NULL REFERENCES users(id),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id BIGSERIAL PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id),
    action TEXT NOT NULL,
    target TEXT NOT NULL, -- table name or resource type
    target_id UUID,
    meta JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_signals_org_ts ON signals(org_id, ts);
CREATE INDEX IF NOT EXISTS idx_signals_org_source ON signals(org_id, source);
CREATE INDEX IF NOT EXISTS idx_incidents_org_status ON incidents(org_id, status);
CREATE INDEX IF NOT EXISTS idx_evidence_incident ON evidence(incident_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_incident ON hypotheses(incident_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_commands_plan ON commands(plan_id);
CREATE INDEX IF NOT EXISTS idx_audit_org_created ON audit_log(org_id, created_at);

-- Create hnsw index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- Insert default organization for development
INSERT INTO orgs (id, name, plan) VALUES
('550e8400-e29b-41d4-a716-446655440000', 'Default Organization', 'pro')
ON CONFLICT (id) DO NOTHING;

-- Row Level Security (RLS) Policies
-- Enable RLS on all tables
ALTER TABLE orgs ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE connectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE topologies ENABLE ROW LEVEL SECURITY;
ALTER TABLE topology_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE incidents ENABLE ROW LEVEL SECURITY;
ALTER TABLE evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE hypotheses ENABLE ROW LEVEL SECURITY;
ALTER TABLE docs ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE fix_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE commands ENABLE ROW LEVEL SECURITY;
ALTER TABLE approvals ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- Create function to get current organization context
CREATE OR REPLACE FUNCTION auth.get_current_org_id()
RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    org_id UUID;
BEGIN
    -- Get org_id from session variable set by application
    BEGIN
        org_id := current_setting('app.org_id')::UUID;
    EXCEPTION
        WHEN OTHERS THEN
            -- Fallback: get org_id from current user context
            SELECT u.org_id INTO org_id
            FROM users u
            WHERE u.id = current_setting('app.user_id')::UUID;
    END;

    RETURN org_id;
END;
$$;

-- Create function to check if RLS is enabled
CREATE OR REPLACE FUNCTION auth.rls_enabled()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    BEGIN
        RETURN current_setting('app.enable_rls')::BOOLEAN;
    EXCEPTION
        WHEN OTHERS THEN
            RETURN TRUE; -- Default to enabled
    END;
END;
$$;

-- Create function to check user permissions
CREATE OR REPLACE FUNCTION auth.has_permission(
    required_permission TEXT,
    target_org_id UUID DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    user_role TEXT;
    current_org_id UUID;
BEGIN
    -- Get current user role
    SELECT u.role INTO user_role
    FROM users u
    WHERE u.id = current_setting('app.user_id')::UUID;

    -- Get current org context
    current_org_id := auth.get_current_org_id();

    -- Check if target org matches current org (if specified)
    IF target_org_id IS NOT NULL AND target_org_id != current_org_id THEN
        RETURN FALSE;
    END IF;

    -- Role-based permissions
    CASE user_role
        WHEN 'admin' THEN
            RETURN TRUE; -- Admins can do everything
        WHEN 'member' THEN
            -- Members can read/write most resources but not user management
            RETURN required_permission IN ('read', 'write', 'execute');
        WHEN 'viewer' THEN
            -- Viewers can only read
            RETURN required_permission = 'read';
        ELSE
            RETURN FALSE;
    END CASE;
END;
$$;

-- Organizations table policies
CREATE POLICY orgs_policy ON orgs
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        id = auth.get_current_org_id()
    );

-- Users table policies
CREATE POLICY users_read_policy ON users
    FOR SELECT USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

CREATE POLICY users_write_policy ON users
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        (org_id = auth.get_current_org_id() AND auth.has_permission('write'))
    );

-- Connectors table policies
CREATE POLICY connectors_policy ON connectors
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Signals table policies
CREATE POLICY signals_policy ON signals
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Topologies table policies
CREATE POLICY topologies_policy ON topologies
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Topology edges table policies
CREATE POLICY topology_edges_policy ON topology_edges
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Incidents table policies
CREATE POLICY incidents_policy ON incidents
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Evidence table policies
CREATE POLICY evidence_policy ON evidence
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        EXISTS (
            SELECT 1 FROM incidents i
            WHERE i.id = evidence.incident_id
            AND i.org_id = auth.get_current_org_id()
        )
    );

-- Hypotheses table policies
CREATE POLICY hypotheses_policy ON hypotheses
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        EXISTS (
            SELECT 1 FROM incidents i
            WHERE i.id = hypotheses.incident_id
            AND i.org_id = auth.get_current_org_id()
        )
    );

-- Documents table policies
CREATE POLICY docs_policy ON docs
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Chunks table policies
CREATE POLICY chunks_policy ON chunks
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        EXISTS (
            SELECT 1 FROM docs d
            WHERE d.id = chunks.doc_id
            AND d.org_id = auth.get_current_org_id()
        )
    );

-- Fix plans table policies
CREATE POLICY fix_plans_policy ON fix_plans
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        EXISTS (
            SELECT 1 FROM incidents i
            WHERE i.id = fix_plans.incident_id
            AND i.org_id = auth.get_current_org_id()
        )
    );

-- Commands table policies
CREATE POLICY commands_policy ON commands
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        EXISTS (
            SELECT 1 FROM fix_plans fp
            JOIN incidents i ON i.id = fp.incident_id
            WHERE fp.id = commands.plan_id
            AND i.org_id = auth.get_current_org_id()
        )
    );

-- Approvals table policies
CREATE POLICY approvals_policy ON approvals
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        EXISTS (
            SELECT 1 FROM fix_plans fp
            JOIN incidents i ON i.id = fp.incident_id
            WHERE fp.id = approvals.plan_id
            AND i.org_id = auth.get_current_org_id()
        )
    );

-- Idempotency keys table
CREATE TABLE IF NOT EXISTS idempotency_keys (
    key TEXT PRIMARY KEY,
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    request_hash TEXT NOT NULL,
    request_data JSONB,
    response_data JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Add index for efficient lookups
CREATE INDEX IF NOT EXISTS idx_idempotency_keys_org_created ON idempotency_keys(org_id, created_at);

-- Create policy for idempotency keys
CREATE POLICY idempotency_keys_policy ON idempotency_keys
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Secrets table for encrypted storage
CREATE TABLE IF NOT EXISTS secrets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    org_id UUID NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    encrypted_data TEXT NOT NULL,
    description TEXT,
    created_by UUID REFERENCES users(id),
    updated_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(org_id, name)
);

-- Create policy for secrets
CREATE POLICY secrets_policy ON secrets
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Add index for efficient secret lookups
CREATE INDEX IF NOT EXISTS idx_secrets_org_name ON secrets(org_id, name);

-- Audit log policies
CREATE POLICY audit_log_policy ON audit_log
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );
