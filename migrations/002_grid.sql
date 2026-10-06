-- Domain model. Names follow the blueprint's schema; additions are marked (+). Points are x/y kilometres on the synthetic map, not PostGIS.

CREATE TABLE grids (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants UNIQUE, name text NOT NULL, model_seed int NOT NULL, day date NOT NULL, clock_step int NOT NULL, created_at timestamptz NOT NULL DEFAULT now());   -- (+) seed and clock (15-minute steps)
SELECT enable_tenant_rls('grids');

CREATE TABLE grid_nodes (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, name text NOT NULL, voltage_kv numeric NOT NULL, geom_x numeric NOT NULL, geom_y numeric NOT NULL, node_type text NOT NULL, region text NOT NULL, bus int NOT NULL);   -- (+) region, bus number
SELECT enable_tenant_rls('grid_nodes');

CREATE TABLE lines (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, from_node uuid NOT NULL REFERENCES grid_nodes, to_node uuid NOT NULL REFERENCES grid_nodes, reactance numeric NOT NULL, capacity_mw numeric NOT NULL, status text NOT NULL DEFAULT 'in service', line_no int NOT NULL, length_km numeric NOT NULL, status_changed_at timestamptz);   -- (+)
SELECT enable_tenant_rls('lines');

CREATE TABLE assets (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, node_id uuid NOT NULL REFERENCES grid_nodes, asset_type text NOT NULL, capacity_mw numeric NOT NULL, metadata jsonb NOT NULL DEFAULT '{}', name text NOT NULL);   -- (+) name
SELECT enable_tenant_rls('assets');

CREATE TABLE telemetry (asset_id uuid NOT NULL REFERENCES assets, ts timestamptz NOT NULL, metric text NOT NULL, value double precision NOT NULL, unit text NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (asset_id, ts, metric));
SELECT enable_tenant_rls('telemetry');
CREATE INDEX telemetry_ts ON telemetry (tenant_id, ts);

CREATE TABLE forecasts (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, asset_id uuid REFERENCES assets, target text NOT NULL, issue_time timestamptz NOT NULL, target_time timestamptz NOT NULL, value numeric NOT NULL, lower numeric, upper numeric, model_version text NOT NULL, kind text NOT NULL);   -- (+) kind: day-ahead or intraday; asset null = system
SELECT enable_tenant_rls('forecasts');
CREATE INDEX forecasts_issue ON forecasts (tenant_id, target, issue_time);

CREATE TABLE batteries (asset_id uuid PRIMARY KEY REFERENCES assets, energy_mwh numeric NOT NULL, power_mw numeric NOT NULL, soc_min numeric NOT NULL, soc_max numeric NOT NULL, efficiency numeric NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, soc_mwh numeric NOT NULL);   -- (+) current state of charge
SELECT enable_tenant_rls('batteries');

CREATE TABLE dispatch_plans (id uuid PRIMARY KEY, tenant_id uuid NOT NULL REFERENCES tenants, created_at timestamptz NOT NULL DEFAULT now(), horizon_start timestamptz NOT NULL, interval_min int NOT NULL, objective jsonb NOT NULL, status text NOT NULL, created_by uuid, summary jsonb NOT NULL DEFAULT '{}');   -- (+) summary
SELECT enable_tenant_rls('dispatch_plans');

CREATE TABLE dispatch_steps (plan_id uuid NOT NULL REFERENCES dispatch_plans ON DELETE CASCADE, asset_id uuid NOT NULL REFERENCES assets, ts timestamptz NOT NULL, setpoint_mw numeric NOT NULL, tenant_id uuid NOT NULL REFERENCES tenants, PRIMARY KEY (plan_id, asset_id, ts));
SELECT enable_tenant_rls('dispatch_steps');
