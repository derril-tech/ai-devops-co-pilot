-- Row Level Security (RLS) Policies for AI DevOps Copilot
-- Ensures users can only access data from their organization

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

-- Audit log policies
CREATE POLICY audit_log_policy ON audit_log
    FOR ALL USING (
        auth.rls_enabled() = FALSE OR
        org_id = auth.get_current_org_id()
    );

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_org_id ON users(org_id);
CREATE INDEX IF NOT EXISTS idx_connectors_org_id ON connectors(org_id);
CREATE INDEX IF NOT EXISTS idx_signals_org_id_ts ON signals(org_id, ts);
CREATE INDEX IF NOT EXISTS idx_incidents_org_id ON incidents(org_id);
CREATE INDEX IF NOT EXISTS idx_evidence_incident_id ON evidence(incident_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_incident_id ON hypotheses(incident_id);
CREATE INDEX IF NOT EXISTS idx_docs_org_id ON docs(org_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_fix_plans_incident_id ON fix_plans(incident_id);
CREATE INDEX IF NOT EXISTS idx_commands_plan_id ON commands(plan_id);
CREATE INDEX IF NOT EXISTS idx_approvals_plan_id ON approvals(plan_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_org_id ON audit_log(org_id);

-- Create function to log audit events
CREATE OR REPLACE FUNCTION audit.log_action()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    action_type TEXT;
    target_table TEXT;
    target_id UUID;
BEGIN
    -- Determine action type
    IF TG_OP = 'INSERT' THEN
        action_type := 'create';
        target_id := NEW.id;
    ELSIF TG_OP = 'UPDATE' THEN
        action_type := 'update';
        target_id := NEW.id;
    ELSIF TG_OP = 'DELETE' THEN
        action_type := 'delete';
        target_id := OLD.id;
    END IF;

    -- Get table name
    target_table := TG_TABLE_NAME;

    -- Insert audit log entry
    INSERT INTO audit_log (org_id, user_id, action, target, target_id, meta)
    VALUES (
        auth.get_current_org_id(),
        current_setting('app.user_id', TRUE)::UUID,
        action_type,
        target_table,
        target_id,
        jsonb_build_object(
            'timestamp', now(),
            'operation', TG_OP,
            'table', target_table
        )
    );

    -- Return appropriate record
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$;

-- Create audit triggers for sensitive tables
CREATE TRIGGER audit_users_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit.log_action();

CREATE TRIGGER audit_connectors_trigger
    AFTER INSERT OR UPDATE OR DELETE ON connectors
    FOR EACH ROW EXECUTE FUNCTION audit.log_action();

CREATE TRIGGER audit_incidents_trigger
    AFTER INSERT OR UPDATE OR DELETE ON incidents
    FOR EACH ROW EXECUTE FUNCTION audit.log_action();

CREATE TRIGGER audit_fix_plans_trigger
    AFTER INSERT OR UPDATE OR DELETE ON fix_plans
    FOR EACH ROW EXECUTE FUNCTION audit.log_action();

-- Create function to check command execution permissions
CREATE OR REPLACE FUNCTION auth.can_execute_command(plan_id UUID)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    plan_status TEXT;
    approval_count INT;
BEGIN
    -- Check if plan exists and belongs to user's org
    SELECT fp.status INTO plan_status
    FROM fix_plans fp
    JOIN incidents i ON i.id = fp.incident_id
    WHERE fp.id = plan_id
    AND i.org_id = auth.get_current_org_id();

    IF plan_status IS NULL THEN
        RETURN FALSE;
    END IF;

    -- Only approved plans can be executed
    IF plan_status != 'approved' THEN
        RETURN FALSE;
    END IF;

    -- Check if user has execute permission
    RETURN auth.has_permission('execute');
END;
$$;

-- Create function to validate fix plan before execution
CREATE OR REPLACE FUNCTION auth.validate_fix_plan(plan_id UUID)
RETURNS TABLE (
    is_valid BOOLEAN,
    error_message TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    plan_record RECORD;
BEGIN
    -- Get plan details
    SELECT fp.*, i.org_id INTO plan_record
    FROM fix_plans fp
    JOIN incidents i ON i.id = fp.incident_id
    WHERE fp.id = plan_id;

    -- Check if plan exists
    IF plan_record.id IS NULL THEN
        RETURN QUERY SELECT FALSE, 'Fix plan not found';
        RETURN;
    END IF;

    -- Check organization access
    IF plan_record.org_id != auth.get_current_org_id() THEN
        RETURN QUERY SELECT FALSE, 'Access denied: plan belongs to different organization';
        RETURN;
    END IF;

    -- Check plan status
    IF plan_record.status NOT IN ('approved', 'executing') THEN
        RETURN QUERY SELECT FALSE, 'Plan must be approved before execution';
        RETURN;
    END IF;

    -- Check user permissions
    IF NOT auth.has_permission('execute') THEN
        RETURN QUERY SELECT FALSE, 'Insufficient permissions to execute plans';
        RETURN;
    END IF;

    -- All checks passed
    RETURN QUERY SELECT TRUE, '';
END;
$$;
