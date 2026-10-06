"""Integration, end-to-end and security tests against a real Postgres."""
import uuid

import pytest

from core import db, jobs, scenario
from rg import world as W

from .conftest import bearer

MANAGER, OPERATOR, VIEWER, OTHER = bearer("grid", "control_manager"), bearer("grid", "operator"), bearer("grid", "viewer"), bearer("other", "control_manager")


def test_nothing_works_before_a_grid_is_loaded(client):
    assert client.get("/v1/grid/state", headers=VIEWER).json()["code"] == "no_grid" and client.post("/v1/dispatch/optimize", headers=OPERATOR, json={}).json()["code"] == "no_grid"


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    load = demo["load"]["result"]
    out = W.build()["demo_outage"]["line"]
    assert load["buses"] == 118 == one("SELECT count(*) AS n FROM grid_nodes")["n"] and one("SELECT count(*) AS n FROM lines")["n"] == 180 and one("SELECT count(*) AS n FROM batteries")["n"] == 6
    assert one("SELECT count(*) AS n FROM forecasts WHERE kind = 'day-ahead'")["n"] == 96 * 3
    assert demo["state0"]["overloaded"] == [] and demo["state0"]["lines_out"] == []
    assert demo["poll"]["result"]["clock"] == "12:00" and demo["trip"]["lines_out"] == [out] and one("SELECT status FROM lines WHERE line_no = %s", [out])["status"] == "out"
    assert demo["island"]["code"] == "would_island" and demo["tele"]["accepted"] == 1 and len(demo["tele"]["rejected"]) == 3
    f = demo["fc_ren"]
    assert f["last_hour_ratio"]["solar"] < 0.6 and f["rest_of_day_mwh"]["solar_day_ahead"] > f["rest_of_day_mwh"]["solar_intraday"] > f["simulation_truth"]["solar_rest_of_day_mwh"]
    assert one("SELECT count(*) AS n FROM forecasts WHERE kind = 'intraday'")["n"] == 2 * (96 - 48)
    s1 = demo["state1"]
    assert s1["lines_out"] == [out] and s1["overloaded"] and s1["max_loading"] > 1.5
    n = demo["n1"]
    assert n["lines_tested"] == 179 and n["already_overloaded"] and n["islanding"] > 0
    p = demo["plan"]
    t = p["on_the_true_weather"]
    assert p["status"] == "active" and p["simultaneous_charge_discharge_mw"] == 0 and p["max_planned_loading"] <= 1.001 and one("SELECT count(*) AS n FROM dispatch_steps")["n"] == 48 * 39
    assert t["no_battery_network_blind"]["overload_minutes"] > 120 and t["this_plan_as_made"]["overload_minutes"] <= 60 and t["perfect_knowledge"]["overload_minutes"] == 0
    assert t["perfect_knowledge"]["renewable_utilisation"] >= t["rolling_with_batteries"]["renewable_utilisation"] >= t["rolling_no_batteries"]["renewable_utilisation"]
    assert demo["plan_get"]["status"] == "active" and set(demo["plan_get"]["setpoints_by_type"]) == {"battery", "solar", "wind", "thermal"}
    assert demo["audit"]["chain_valid"] and {"grid.loaded", "telemetry.polled", "line.status", "telemetry.received", "forecast.renewables", "contingencies.run", "dispatch.planned"} <= {e["action"] for e in demo["audit"]["events"]}


def test_powerflow_and_validation(client, demo):
    post = lambda path, body, h=OPERATOR: client.post(path, headers=h, json=body)     # noqa: E731
    future = post("/v1/powerflow", {"step": 70}).json()
    assert future["source"] == "forecast" and future["step"] == "17:30"
    radial = next(ln["idx"] for ln in W.build()["lines"] if len(set(W.connected(W.build(), (ln["idx"],)))) > 1)
    assert post("/v1/powerflow", {"extra_lines_out": [radial]}).json()["code"] == "islanded"
    assert post("/v1/grid/lines/9999/status", {"status": "out", "reason": "test"}).status_code == 404 and post("/v1/grid/lines/1/status", {"status": "broken", "reason": "test"}).status_code == 422
    assert post("/v1/telemetry", {"readings": [{"asset": "Solar 1", "ts": "2026-07-14T23:00:00+00:00", "value": 5}]}).json()["rejected"][0]["why"] == "timestamp is in the future"
    assert post("/v1/telemetry:poll", {"steps": 56}).json()["code"] == "end_of_day"
    assert client.get(f"/v1/plans/{uuid.uuid4()}", headers=VIEWER).status_code == 404


def test_roles(client, demo):
    assert client.get("/v1/grid/state").status_code == 401 and client.get("/v1/grid/state", headers=VIEWER).status_code == 200
    for path in ("/v1/telemetry", "/v1/forecast/renewables", "/v1/forecast/load", "/v1/dispatch/optimize", "/v1/grid/lines/1/status", "/v1/telemetry:poll"):
        assert client.post(path, headers=VIEWER, json={}).status_code == 403, path
    assert client.post("/v1/grid:load", headers=OPERATOR, json={}).status_code == 403 and client.get("/v1/audit", headers=OPERATOR).status_code == 403 and client.get("/v1/audit", headers=MANAGER).status_code == 200


def test_another_operator_sees_nothing(client, demo, seeded):
    assert client.get("/v1/grid/state", headers=OTHER).json()["code"] == "no_grid"
    for table in ("grids", "grid_nodes", "lines", "assets", "telemetry", "forecasts", "batteries", "dispatch_plans", "dispatch_steps"):
        assert one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["other"])["n"] == 0 < one(f"SELECT count(*) AS n FROM {table}", tenant=seeded["grid"])["n"], table
    with pytest.raises(Exception, match="row-level security"), db.tx(seeded["other"]) as c:
        c.execute("INSERT INTO grids (id, tenant_id, name, model_seed, day, clock_step) VALUES (%s,%s,'x',1,'2026-07-14',0)", [uuid.uuid4(), seeded["grid"]])


def test_idempotent_writes(client, demo):
    again = client.post("/v1/dispatch/optimize", headers={**OPERATOR, "Idempotency-Key": "demo-1:plan"}, json={"use_batteries": True, "activate": True})
    assert again.status_code == 201 and again.headers["Idempotent-Replay"] == "true" and again.json()["plan_id"] == demo["plan"]["plan_id"] and one("SELECT count(*) AS n FROM dispatch_plans")["n"] == 1
    assert client.post("/v1/dispatch/optimize", headers={**OPERATOR, "Idempotency-Key": "demo-1:plan"}, json={"use_batteries": False}).json()["code"] == "idempotency_key_reused"
    assert client.post("/v1/grid:load", headers=MANAGER, json={"seed": 17}).json()["code"] == "grid_exists"
