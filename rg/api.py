"""Public API (modular monolith). Blueprint services map to: scada-ingestion (telemetry in, and a poll that stands in for the feed),
grid-topology (nodes, lines, outages), renewable and load forecasting, power-flow (DC), contingency simulator (N-1), battery optimizer
(MILP dispatch), market signal (prices in the scoring) and the operator API. Not built: an AC solver, real SCADA or weather feeds, a
market interface, security-constrained unit commitment.

    uvicorn rg.api:app          python -m core.jobs rg.api      # the worker
"""
import datetime
import functools
import uuid

import numpy as np
from fastapi import Depends, Header, Query
from fastapi.encoders import jsonable_encoder
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine, world as W

READ = {"grid:read", "jobs:read"}
WORK = READ | {"telemetry:write", "forecasts:run", "grid:operate", "dispatch:run"}
PERMISSIONS = {"viewer": READ, "operator": WORK, "control_manager": WORK | {"grid:load", "audit:read"}}
app = create_app("renewable-grid-twin", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
DAY = datetime.date(2026, 7, 14)
LOAD_STEP = 40                                     # the demo loads at 10:00


def ts(step, day_offset=0):
    return datetime.datetime.combine(DAY + datetime.timedelta(days=day_offset), datetime.time(0), tzinfo=datetime.timezone.utc) + datetime.timedelta(minutes=15 * int(step))


def hhmm(step):
    return f"{int(step) * 15 // 60:02d}:{int(step) * 15 % 60:02d}"


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def grid(c):
    g = c.execute("SELECT * FROM grids").fetchone()
    if not g:
        raise Problem(409, "no_grid", "load a grid first: POST /v1/grid:load")
    return g, W.build(g["model_seed"])


def aid(t, kind, k):
    return uuid.uuid5(t, f"{kind}-{k}")


def lines_out(c):
    return tuple(sorted(r["line_no"] for r in c.execute("SELECT line_no FROM lines WHERE status = 'out'")))


def telemetry_day(c, t, w, upto):
    """Demo-day telemetry from the database up to step `upto` (exclusive): per-site solar and wind, and system load. -> arrays"""
    S, Wn = len(w["solar"]), len(w["wind"])
    sol, win, load = np.full((S, upto), np.nan), np.full((Wn, upto), np.nan), np.full(upto, np.nan)
    index = {aid(t, "solar", k): ("s", k) for k in range(S)} | {aid(t, "wind", k): ("w", k) for k in range(Wn)} | {aid(t, "load", 0): ("l", 0)}
    for r in c.execute("SELECT asset_id, ts, value FROM telemetry WHERE ts >= %s AND ts < %s AND metric = 'power_mw'", [ts(0), ts(upto)]):
        kind, k = index.get(r["asset_id"], (None, None))
        step = int((r["ts"] - ts(0)).total_seconds() // 900)
        if kind == "s":
            sol[k, step] = r["value"]
        elif kind == "w":
            win[k, step] = r["value"]
        elif kind == "l":
            load[step] = r["value"]
    return sol, win, load


def telemetry_rows(t, w, steps, day_offset=0, hourly=False):
    d0 = (W.DEMO_DAY + day_offset) * W.STEPS
    rows = []
    rng = range(0, len(steps), 4) if hourly else range(len(steps))
    for i in rng:
        s = steps[i]
        sl = slice(d0 + s, d0 + s + (4 if hourly else 1))
        for k in range(len(w["solar"])):
            rows.append((aid(t, "solar", k), ts(s, day_offset), "power_mw", round(float(w["solar_out"][k, sl].mean()), 2), "MW", t))
        for k in range(len(w["wind"])):
            rows.append((aid(t, "wind", k), ts(s, day_offset), "power_mw", round(float(w["wind_out"][k, sl].mean()), 2), "MW", t))
        rows.append((aid(t, "load", 0), ts(s, day_offset), "power_mw", round(float(w["load"][sl].mean()), 1), "MW", t))
    return rows


# ---------------------------------------------------------------- load and telemetry

class LoadIn(BaseModel):
    seed: int = Field(17, ge=0, le=10 ** 6)


@app.post("/v1/grid:load", status_code=202, tags=["grid"], summary="(+) Load the invented 118-bus network at 10:00: buses, lines, thermal, solar, wind and battery assets, sixty days of hourly telemetry, the day so far at 15 minutes; fit the forecast models and issue the day-ahead forecast")
def load_grid(body: LoadIn, ctx: Ctx = Depends(auth("grid:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM grids LIMIT 1").fetchone():
            raise Problem(409, "grid_exists")
        return 202, jobs.enqueue(c, ctx, "grid.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("grid.load")
def load_job(c, job):
    t, seed = job["tenant_id"], job["payload"]["seed"]
    w = W.build(seed)
    gid = uuid.uuid5(t, "grid")
    c.execute("INSERT INTO grids (id, tenant_id, name, model_seed, day, clock_step) VALUES (%s,%s,'Riverbend 118-bus system',%s,%s,%s)", [gid, t, seed, DAY, LOAD_STEP])
    kinds = {}
    for u in w["thermal"]:
        kinds[u["bus"]] = "generation"
    for s in w["solar"] + w["wind"]:
        kinds.setdefault(s["bus"], "renewable")
    db.load(c, "grid_nodes", ["id", "tenant_id", "name", "voltage_kv", "geom_x", "geom_y", "node_type", "region", "bus"], [(aid(t, "bus", b), t, f"Bus {b + 1}", 345 if w["region"][b] == "metro" else 230, w["xy"][b][0], w["xy"][b][1], kinds.get(b, "load" if b in w["load_share"] else "switching"), w["region"][b], b) for b in range(W.N_BUS)])
    db.load(c, "lines", ["id", "tenant_id", "from_node", "to_node", "reactance", "capacity_mw", "status", "line_no", "length_km"], [(aid(t, "line", ln["idx"]), t, aid(t, "bus", ln["from"]), aid(t, "bus", ln["to"]), ln["x"], ln["cap"], "in service", ln["idx"], ln["km"]) for ln in w["lines"]])
    assets = [(aid(t, "thermal", k), t, aid(t, "bus", u["bus"]), "thermal", round(u["cap"], 1), Jsonb({"cost_per_mwh": round(u["cost"], 2)}), f"Thermal {k + 1}") for k, u in enumerate(w["thermal"])]
    assets += [(aid(t, "solar", k), t, aid(t, "bus", s["bus"]), "solar", round(s["cap"], 1), Jsonb({"region": s["region"]}), f"Solar {k + 1}") for k, s in enumerate(w["solar"])]
    assets += [(aid(t, "wind", k), t, aid(t, "bus", x["bus"]), "wind", round(x["cap"], 1), Jsonb({"region": x["region"]}), f"Wind {k + 1}") for k, x in enumerate(w["wind"])]
    assets += [(aid(t, "battery", k), t, aid(t, "bus", b["bus"]), "battery", round(b["power"], 1), Jsonb({"region": w["region"][b["bus"]]}), f"Battery {k + 1}") for k, b in enumerate(w["batteries"])]
    assets += [(aid(t, "load", 0), t, aid(t, "bus", 0), "load_aggregate", round(float(w["load"].max()), 1), Jsonb({"buses": len(w["load_share"])}), "System load")]
    db.load(c, "assets", ["id", "tenant_id", "node_id", "asset_type", "capacity_mw", "metadata", "name"], assets)
    db.load(c, "batteries", ["asset_id", "energy_mwh", "power_mw", "soc_min", "soc_max", "efficiency", "tenant_id", "soc_mwh"], [(aid(t, "battery", k), round(b["energy"], 1), round(b["power"], 1), 0.1, 0.95, b["eff"], t, round(b["soc0"] * b["energy"], 1)) for k, b in enumerate(w["batteries"])])
    rows = []
    for d in range(W.HISTORY_DAYS):
        rows += telemetry_rows(t, w, list(range(W.STEPS)), day_offset=d - W.HISTORY_DAYS, hourly=True)
    rows += telemetry_rows(t, w, list(range(LOAD_STEP)))
    db.load(c, "telemetry", ["asset_id", "ts", "metric", "value", "unit", "tenant_id"], rows)
    m = engine.models(seed)
    da = engine.day_ahead(seed)
    fc = []
    for s in range(W.STEPS):
        fc.append((uuid.uuid7(), t, None, "solar", ts(0, -1), ts(s), round(float(da["solar"][:, s].sum()), 1), None, None, engine.VERSION, "day-ahead"))
        fc.append((uuid.uuid7(), t, None, "wind", ts(0, -1), ts(s), round(float(da["wind"][:, s].sum()), 1), None, None, engine.VERSION, "day-ahead"))
        fc.append((uuid.uuid7(), t, None, "load", ts(0, -1), ts(s), round(float(da["load"][s]), 1), round(float(da["load_p10"][s]), 1), round(float(da["load_p90"][s]), 1), engine.VERSION, "day-ahead"))
    db.load(c, "forecasts", ["id", "tenant_id", "asset_id", "target", "issue_time", "target_time", "value", "lower", "upper", "model_version", "kind"], fc)
    out = {"grid_id": gid, "day": str(DAY), "clock": hhmm(LOAD_STEP), "buses": W.N_BUS, "lines": len(w["lines"]), "radial_lines": sum(1 for ln in w["lines"] if len(set(W.connected(w, (ln["idx"],)))) > 1),
           "capacity_mw": {"thermal": round(sum(u["cap"] for u in w["thermal"])), "solar": round(sum(s["cap"] for s in w["solar"])), "wind": round(sum(x["cap"] for x in w["wind"])), "battery": round(sum(b["power"] for b in w["batteries"])), "battery_mwh": round(sum(b["energy"] for b in w["batteries"]))},
           "regions": {g: sum(1 for x in w["region"] if x == g) for g in W.REGIONS}, "telemetry_rows": len(rows), "models": {"version": engine.VERSION, **m["report"]},
           "simulation_truth": {"what": "what the day holds, which the forecasts never see", "cloud_bank": {**W.CLOUD_BANK, "from": hhmm(W.CLOUD_BANK["from"]), "to": hhmm(W.CLOUD_BANK["to"])}, "line_outage": {"line": w["demo_outage"]["line"], "at": hhmm(W.CLOCK_STEP)}}}
    audit.record(c, worker_ctx(job), "grid.loaded", "grid", gid, {"buses": W.N_BUS, "lines": len(w["lines"])})
    return out


class PollIn(BaseModel):
    steps: int = Field(ge=1, le=56)


@app.post("/v1/telemetry:poll", status_code=202, tags=["telemetry"], summary="(+) SCADA delivers the next 15-minute readings and the clock moves; stands in for the feed")
def poll(body: PollIn, ctx: Ctx = Depends(auth("telemetry:write")), idem: str | None = IdemKey):
    def work(c):
        g, _ = grid(c)
        if g["clock_step"] + body.steps > W.STEPS:
            raise Problem(422, "end_of_day")
        return 202, jobs.enqueue(c, ctx, "telemetry.poll", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("telemetry.poll")
def poll_job(c, job):
    t = job["tenant_id"]
    g, w = grid(c)
    s0, s1 = g["clock_step"], g["clock_step"] + job["payload"]["steps"]
    rows = telemetry_rows(t, w, list(range(s0, s1)))
    db.load(c, "telemetry", ["asset_id", "ts", "metric", "value", "unit", "tenant_id"], rows)
    c.execute("UPDATE grids SET clock_step = %s WHERE id = %s", [s1, g["id"]])
    audit.record(c, worker_ctx(job), "telemetry.polled", "grid", g["id"], {"from": hhmm(s0), "to": hhmm(s1), "readings": len(rows)})
    d0 = W.DEMO_DAY * W.STEPS
    da = engine.day_ahead(g["model_seed"])
    return {"clock": hhmm(s1), "readings": len(rows), "solar_mw_now": round(float(w["solar_out"][:, d0 + s1 - 1].sum())), "solar_day_ahead_mw_now": round(float(da["solar"][:, s1 - 1].sum())), "wind_mw_now": round(float(w["wind_out"][:, d0 + s1 - 1].sum())), "load_mw_now": round(float(w["load"][d0 + s1 - 1]))}


class Reading(BaseModel):
    asset: str = Field(max_length=40, description="asset name, e.g. 'Solar 3'")
    ts: datetime.datetime
    metric: str = Field("power_mw", max_length=20)
    value: float


class TelemetryIn(BaseModel):
    readings: list[Reading] = Field(min_length=1, max_length=5000)


@app.post("/v1/telemetry", status_code=201, tags=["telemetry"], summary="Readings from SCADA; each is checked (known asset, a timezone, not in the future, within the asset's range) and accepted or refused on its own")
def telemetry(body: TelemetryIn, ctx: Ctx = Depends(auth("telemetry:write")), idem: str | None = IdemKey):
    def work(c):
        g, w = grid(c)
        assets = {r["name"]: r for r in c.execute("SELECT id, name, asset_type, capacity_mw FROM assets")}
        accepted, rejected = 0, []
        for r in body.readings:
            a = assets.get(r.asset)
            why = None
            if not a:
                why = "unknown asset"
            elif r.ts.tzinfo is None:
                why = "timestamp needs a timezone"
            elif r.ts > ts(g["clock_step"]):
                why = "timestamp is in the future"
            elif r.metric == "power_mw" and not (-float(a["capacity_mw"]) * 1.05 <= r.value <= float(a["capacity_mw"]) * 1.05):
                why = f"outside the asset's range (capacity {float(a['capacity_mw'])} MW)"
            if why:
                rejected.append({"asset": r.asset, "why": why})
                continue
            c.execute("INSERT INTO telemetry (asset_id, ts, metric, value, unit, tenant_id) VALUES (%s,%s,%s,%s,'MW',%s) ON CONFLICT (asset_id, ts, metric) DO UPDATE SET value = EXCLUDED.value", [a["id"], r.ts, r.metric, r.value, ctx.tenant_id])
            accepted += 1
        audit.record(c, ctx, "telemetry.received", "grid", g["id"], {"accepted": accepted, "rejected": len(rejected)})
        return 201, {"accepted": accepted, "rejected": rejected}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- topology, state, power flow, contingencies

class LineStatusIn(BaseModel):
    status: str = Field(pattern="^(out|in service)$")
    reason: str = Field(min_length=3, max_length=200)


@app.post("/v1/grid/lines/{line_no}/status", tags=["grid"], summary="(+) A line trips or returns to service; refused if taking it out would island part of the network")
def line_status(line_no: int, body: LineStatusIn, ctx: Ctx = Depends(auth("grid:operate")), idem: str | None = IdemKey):
    def work(c):
        g, w = grid(c)
        ln = c.execute("SELECT * FROM lines WHERE line_no = %s", [line_no]).fetchone()
        if not ln:
            raise Problem(404, "line_not_found")
        out = set(lines_out(c))
        if body.status == "out":
            lab = W.connected(w, tuple(sorted(out | {line_no})))
            if len(set(lab)) > 1:
                raise Problem(409, "would_island", f"taking line {line_no} out cuts off {sum(1 for x in lab if x != lab[0])} buses; open it only with those buses' load and generation handled")
        c.execute("UPDATE lines SET status = %s, status_changed_at = %s WHERE id = %s", [body.status, ts(g["clock_step"]), ln["id"]])
        audit.record(c, ctx, "line.status", "line", ln["id"], {"line": line_no, "status": body.status, "reason": body.reason})
        return 200, {"line": line_no, "status": body.status, "at": hhmm(g["clock_step"]), "lines_out": sorted(set(lines_out(c)))}
    return run(ctx, idem, body, work)


def snapshot(c, g, w, step=None, extra_out=()):
    """Injections at a step: the last telemetry for the past, the latest forecasts for the future; batteries at the active plan's setpoints."""
    t = g["tenant_id"]
    clock = g["clock_step"]
    step = clock - 1 if step is None else step
    if step < clock:
        sol, win, load = telemetry_day(c, t, w, step + 1)
        s, x, L = sol[:, step], win[:, step], float(load[step])
    else:
        fc = forecast_now(c, g, w)
        s, x, L = fc["solar"][:, step - clock], fc["wind"][:, step - clock], float(fc["load"][step - clock])
    bat = np.zeros(len(w["batteries"]))
    plan = c.execute("SELECT id FROM dispatch_plans WHERE status = 'active' ORDER BY created_at DESC LIMIT 1").fetchone()
    if plan:
        for r in c.execute("SELECT a.name, s.setpoint_mw FROM dispatch_steps s JOIN assets a ON a.id = s.asset_id WHERE s.plan_id = %s AND s.ts = %s AND a.asset_type = 'battery'", [plan["id"], ts(step)]):
            bat[int(r["name"].split()[-1]) - 1] = float(r["setpoint_mw"])
    d = W.merit_order(w, 0, solar=np.nan_to_num(s), wind=np.nan_to_num(x), load=L, battery=bat)
    out = tuple(sorted(set(lines_out(c)) | set(extra_out)))
    return d, W.injections(w, d, 0), out, step


def flows_report(w, inj, out, top=12):
    lab = W.connected(w, out)
    if len(set(lab)) > 1:
        raise Problem(422, "islanded", f"with lines {list(out)} out, {sum(1 for x in lab if x != lab[0])} buses are cut off; DC power flow needs one connected network")
    f = W.dc_flow(w, inj, out)
    caps = np.array([ln["cap"] for ln in w["lines"]])
    load = np.abs(f) / caps
    order = [int(k) for k in np.argsort(-load) if int(k) not in out][:top]
    return {"lines": [{"line": k, "from_bus": w["lines"][k]["from"] + 1, "to_bus": w["lines"][k]["to"] + 1, "regions": f"{w['region'][w['lines'][k]['from']]}–{w['region'][w['lines'][k]['to']]}", "flow_mw": round(float(f[k]), 1), "limit_mw": float(caps[k]), "loading": round(float(load[k]), 3)} for k in order],
            "overloaded": [int(k) for k in np.where(load > 1.0)[0] if int(k) not in out], "max_loading": round(float(max(load[k] for k in range(len(caps)) if k not in out)), 3)}


@app.get("/v1/grid/state", tags=["grid"], summary="The grid now: clock, lines out, generation by type and region from the last telemetry, battery state of charge, and the most loaded lines from a DC power flow")
def grid_state(ctx: Ctx = Depends(auth("grid:read"))):
    with db.tx(ctx.tenant_id) as c:
        g, w = grid(c)
        d, inj, out, step = snapshot(c, g, w)
        bats = c.execute("SELECT a.name, b.soc_mwh, b.energy_mwh, b.power_mw FROM batteries b JOIN assets a ON a.id = b.asset_id ORDER BY a.name").fetchall()
        fr = flows_report(w, inj, out)
    return jsonable_encoder({"clock": hhmm(g["clock_step"]), "as_of_step": hhmm(step), "lines_out": list(out), "generation_mw": {"solar": round(float(d["solar"].sum())), "wind": round(float(d["wind"].sum())), "thermal": round(float(d["thermal"].sum())), "battery": round(float(d["battery"].sum()))}, "load_mw": round(float(d["load"])), "shed_mw": round(float(d["shed"]), 1),
                             "batteries": bats, **fr, "method": "DC power flow from the latest telemetry with thermal units balancing cheapest first; batteries at the active plan's setpoints"})


class FlowIn(BaseModel):
    step: int | None = Field(None, ge=0, le=95, description="15-minute step of the day; past steps use telemetry, future ones the latest forecast")
    extra_lines_out: list[int] = Field(default_factory=list, max_length=5)


@app.post("/v1/powerflow", status_code=201, tags=["grid"], summary="DC power flow for a step, optionally with more lines out; islanding is refused with the number of buses cut off")
def powerflow(body: FlowIn, ctx: Ctx = Depends(auth("grid:read")), idem: str | None = IdemKey):
    def work(c):
        g, w = grid(c)
        d, inj, out, step = snapshot(c, g, w, body.step, tuple(body.extra_lines_out))
        return 201, {"step": hhmm(step), "lines_out": list(out), "source": "telemetry" if step < g["clock_step"] else "forecast", "generation_mw": {"solar": round(float(d["solar"].sum())), "wind": round(float(d["wind"].sum())), "thermal": round(float(d["thermal"].sum()))}, "load_mw": round(float(d["load"])), **flows_report(w, inj, out)}
    return run(ctx, idem, body, work)


class ContIn(BaseModel):
    step: int | None = Field(None, ge=0, le=95)
    top: int = Field(10, ge=1, le=50)


@app.post("/v1/contingencies", status_code=201, tags=["grid"], summary="N-1: every further single-line outage from the current topology at a step; which island part of the network and which overload others, ranked")
def contingencies(body: ContIn, ctx: Ctx = Depends(auth("grid:read")), idem: str | None = IdemKey):
    def work(c):
        g, w = grid(c)
        d, inj, out, step = snapshot(c, g, w, body.step)
        r = engine.n_minus_1(w, inj, out, body.top)
        for x in r["top"]:
            ln = w["lines"][x["line"]]
            x["regions"] = f"{w['region'][ln['from']]}–{w['region'][ln['to']]}"
        audit.record(c, ctx, "contingencies.run", "grid", g["id"], {"step": hhmm(step), "tested": r["lines_tested"], "causing_new_overload": r["causing_new_overload"]})
        return 201, {"step": hhmm(step), "base_lines_out": list(out), **r}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- forecasts

def forecast_now(c, g, w):
    """The latest view of the rest of the day: day-ahead per site, scaled by the last hour's telemetry against it (decaying back)."""
    clock = g["clock_step"]
    da = engine.day_ahead(g["model_seed"])
    sol, win, load = telemetry_day(c, g["tenant_id"], w, clock)
    n = W.STEPS - clock
    out = {}
    for key, act, fc in (("solar", sol, da["solar"]), ("wind", win, da["wind"])):
        recent_a, recent_f = np.nansum(act[:, clock - 4:clock], axis=0), fc[:, clock - 4:clock].sum(0)
        total, ratio = engine.nowcast(fc[:, clock:].sum(0), recent_a, recent_f, n)
        share = fc[:, clock:] / np.maximum(1e-9, fc[:, clock:].sum(0))
        out[key], out[key + "_ratio"] = share * total, ratio
    out["load"], out["load_ratio"] = engine.nowcast(da["load"][clock:], np.nan_to_num(load[clock - 4:clock]), da["load"][clock - 4:clock], n)
    out["telemetry"] = {"solar": np.nansum(sol, axis=0), "wind": np.nansum(win, axis=0), "load": load}
    out["day_ahead"] = da
    return out


class FcIn(BaseModel):
    store: bool = True


@app.post("/v1/forecast/renewables", status_code=201, tags=["forecasts"], summary="Solar and wind for the rest of the day: the day-ahead forecast, what telemetry shows so far and how far off it is, and the intraday update scaled by the last hour")
def forecast_renewables(body: FcIn, ctx: Ctx = Depends(auth("forecasts:run")), idem: str | None = IdemKey):
    def work(c):
        g, w = grid(c)
        clock = g["clock_step"]
        fc = forecast_now(c, g, w)
        da, tel = fc["day_ahead"], fc["telemetry"]
        m = engine.models(g["model_seed"])
        series = []
        for s in range(W.STEPS):
            row = {"step": s, "time": hhmm(s), "solar_day_ahead": round(float(da["solar"][:, s].sum()), 1), "wind_day_ahead": round(float(da["wind"][:, s].sum()), 1)}
            if s < clock:
                row |= {"solar_actual": round(float(tel["solar"][s]), 1), "wind_actual": round(float(tel["wind"][s]), 1)}
            else:
                row |= {"solar_intraday": round(float(fc["solar"][:, s - clock].sum()), 1), "wind_intraday": round(float(fc["wind"][:, s - clock].sum()), 1)}
            series.append(row)
        last_hour = slice(clock - 4, clock)
        err = {"solar_mw": round(float(np.mean(tel["solar"][last_hour] - da["solar"][:, last_hour].sum(0))), 1), "wind_mw": round(float(np.mean(tel["wind"][last_hour] - da["wind"][:, last_hour].sum(0))), 1)}
        if body.store:
            rows = []
            for s in range(clock, W.STEPS):
                rows.append((uuid.uuid7(), ctx.tenant_id, None, "solar", ts(clock), ts(s), round(float(fc["solar"][:, s - clock].sum()), 1), None, None, engine.VERSION, "intraday"))
                rows.append((uuid.uuid7(), ctx.tenant_id, None, "wind", ts(clock), ts(s), round(float(fc["wind"][:, s - clock].sum()), 1), None, None, engine.VERSION, "intraday"))
            db.load(c, "forecasts", ["id", "tenant_id", "asset_id", "target", "issue_time", "target_time", "value", "lower", "upper", "model_version", "kind"], rows)
        tr = engine.truth(g["model_seed"], clock)
        audit.record(c, ctx, "forecast.renewables", "grid", g["id"], {"clock": hhmm(clock), "solar_ratio": round(fc["solar_ratio"], 2)})
        return 201, {"clock": hhmm(clock), "series": series, "last_hour_error_mw": err, "last_hour_ratio": {"solar": round(fc["solar_ratio"], 3), "wind": round(fc["wind_ratio"], 3)},
                     "rest_of_day_mwh": {"solar_day_ahead": round(float(da["solar"][:, clock:].sum()) * W.DT_H), "solar_intraday": round(float(fc["solar"].sum()) * W.DT_H), "wind_day_ahead": round(float(da["wind"][:, clock:].sum()) * W.DT_H), "wind_intraday": round(float(fc["wind"].sum()) * W.DT_H)},
                     "models": {k: m["report"][k] for k in ("solar", "wind")}, "how": "gradient-boosted trees from forecast weather (clear-sky, cloud cover, wind speed, hour) to per-unit output, spread over sites by capacity; the intraday update scales the rest of the day by the last hour's actual over forecast, decaying back with a two-hour half-life",
                     "simulation_truth": {"solar_rest_of_day_mwh": round(float(tr["solar"].sum()) * W.DT_H), "wind_rest_of_day_mwh": round(float(tr["wind"].sum()) * W.DT_H)}}
    return run(ctx, idem, body, work)


@app.post("/v1/forecast/load", status_code=201, tags=["forecasts"], summary="System load for the rest of the day with an 80% band, and the intraday update")
def forecast_load(body: FcIn, ctx: Ctx = Depends(auth("forecasts:run")), idem: str | None = IdemKey):
    def work(c):
        g, w = grid(c)
        clock = g["clock_step"]
        fc = forecast_now(c, g, w)
        da, tel = fc["day_ahead"], fc["telemetry"]
        m = engine.models(g["model_seed"])
        series = [{"step": s, "time": hhmm(s), "p10": round(float(da["load_p10"][s])), "p50": round(float(da["load"][s])), "p90": round(float(da["load_p90"][s])), **({"actual": round(float(tel["load"][s]))} if s < clock else {"intraday": round(float(fc["load"][s - clock]))})} for s in range(W.STEPS)]
        tr = engine.truth(g["model_seed"], clock)
        return 201, {"clock": hhmm(clock), "series": series, "peak_p50": max(x["p50"] for x in series), "peak_p90": max(x["p90"] for x in series), "model": m["report"]["load"], "simulation_truth": {"peak_rest_of_day": round(float(tr["load"].max()))}}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- dispatch

@functools.lru_cache(maxsize=8)
def comparison(seed, clock, out):
    """Every strategy for the rest of the day, run on the true weather and true flows."""
    w = W.build(seed)
    tr = engine.truth(seed, clock)
    n = W.STEPS - clock
    res = {}
    res["no_battery_network_blind"] = engine.score(w, tr, np.zeros((len(w["batteries"]), n)), out)
    res["price_battery_network_blind"] = engine.score(w, tr, engine.price_schedule(w, tr["price"]), out)
    for label, ub in (("rolling_no_batteries", False), ("rolling_with_batteries", True)):
        r = engine.rolling(seed, clock, out, use_batteries=ub)
        res[label] = engine.score(w, tr, r["battery"], out, cap_solar=r["cap_solar"] + 1e-6, cap_wind=r["cap_wind"] + 1e-6, thermal_plan=r["thermal"])
    pp = engine.optimize(w, tr["solar"], tr["wind"], tr["load"], out)
    res["perfect_knowledge"] = engine.score(w, tr, pp["discharge"] - pp["charge"], out, cap_solar=pp["solar"] + 1e-6, cap_wind=pp["wind"] + 1e-6, thermal_plan=pp["thermal"])
    return res


class DispatchIn(BaseModel):
    use_batteries: bool = True
    activate: bool = True


@app.post("/v1/dispatch/optimize", status_code=201, tags=["dispatch"], summary="Battery charge and discharge, renewable curtailment and thermal output for the rest of the day: a MILP over the DC network with the current topology, line limits added as they bind; scored with the alternatives on the true weather")
def dispatch(body: DispatchIn, ctx: Ctx = Depends(auth("dispatch:run")), idem: str | None = IdemKey):
    def work(c):
        g, w = grid(c)
        clock = g["clock_step"]
        out = lines_out(c)
        fc = forecast_now(c, g, w)
        soc = [float(r["soc_mwh"]) for r in c.execute("SELECT b.soc_mwh FROM batteries b JOIN assets a ON a.id = b.asset_id ORDER BY a.name")]
        p = engine.optimize(w, fc["solar"], fc["wind"], fc["load"], out, soc0=soc, use_batteries=body.use_batteries)
        tr = engine.truth(g["model_seed"], clock)
        plan_score = engine.score(w, tr, p["discharge"] - p["charge"], out, cap_solar=p["solar"] + 1e-6, cap_wind=p["wind"] + 1e-6, thermal_plan=p["thermal"])
        pid = uuid.uuid7()
        n = W.STEPS - clock
        avail = fc["solar"].sum() + fc["wind"].sum()
        summary = {"steps": n, "expected_cost": round(float(sum(u["cost"] * p["thermal"][k].sum() for k, u in enumerate(w["thermal"])) * W.DT_H)), "curtailed_mwh": round(float(avail - p["solar"].sum() - p["wind"].sum()) * W.DT_H), "battery_throughput_mwh": round(float((p["charge"] + p["discharge"]).sum()) * W.DT_H),
                   "line_constraints_added": p["line_constraints"], "solver_rounds": p["rounds"], "max_planned_loading": round(float(np.max(np.abs(p["flows"]) / np.array([ln["cap"] for ln in w["lines"]]))), 3), "simultaneous_charge_discharge_mw": round(p["simultaneous_charge_discharge"], 6), "lines_out": list(out), "use_batteries": body.use_batteries}
        if body.activate:
            c.execute("UPDATE dispatch_plans SET status = 'superseded' WHERE status = 'active'")
        c.execute("INSERT INTO dispatch_plans (id, tenant_id, horizon_start, interval_min, objective, status, created_by, summary) VALUES (%s,%s,%s,15,%s,%s,%s,%s)", [pid, ctx.tenant_id, ts(clock), Jsonb({"minimise": "thermal fuel + curtailment $1/MWh + battery wear $8/MWh + shed $3000/MWh + overload $2000/MWh", "constraints": ["balance", "battery power, energy, efficiency, never charging and discharging at once", "line limits by PTDF on the current topology"]}), "active" if body.activate else "proposed", ctx.actor_id, Jsonb(summary)])
        rows = []
        for i in range(n):
            for k in range(len(w["batteries"])):
                rows.append((pid, aid(ctx.tenant_id, "battery", k), ts(clock + i), round(float(p["discharge"][k, i] - p["charge"][k, i]), 2), ctx.tenant_id))
            for k in range(len(w["solar"])):
                rows.append((pid, aid(ctx.tenant_id, "solar", k), ts(clock + i), round(float(p["solar"][k, i]), 2), ctx.tenant_id))
            for k in range(len(w["wind"])):
                rows.append((pid, aid(ctx.tenant_id, "wind", k), ts(clock + i), round(float(p["wind"][k, i]), 2), ctx.tenant_id))
            for k in range(len(w["thermal"])):
                rows.append((pid, aid(ctx.tenant_id, "thermal", k), ts(clock + i), round(float(p["thermal"][k, i]), 2), ctx.tenant_id))
        db.load(c, "dispatch_steps", ["plan_id", "asset_id", "ts", "setpoint_mw", "tenant_id"], rows)
        audit.record(c, ctx, "dispatch.planned", "dispatch_plan", pid, {"from": hhmm(clock), "curtailed_mwh": summary["curtailed_mwh"], "lines_out": list(out)})
        comp = comparison(g["model_seed"], clock, out)
        series = [{"time": hhmm(clock + i), "battery_mw": round(float((p["discharge"][:, i] - p["charge"][:, i]).sum()), 1), "battery_north_mw": round(float(sum(p["discharge"][k, i] - p["charge"][k, i] for k, b in enumerate(w["batteries"]) if w["region"][b["bus"]] == "north")), 1), "wind_used": round(float(p["wind"][:, i].sum()), 1), "wind_available": round(float(fc["wind"][:, i].sum()), 1), "solar_used": round(float(p["solar"][:, i].sum()), 1), "thermal": round(float(p["thermal"][:, i].sum()), 1)} for i in range(n)]
        return 201, {"plan_id": pid, "status": "active" if body.activate else "proposed", "from": hhmm(clock), **summary, "series": series,
                     "on_the_true_weather": {"this_plan_as_made": plan_score, **comp},
                     "how": "MILP (HiGHS) at 15-minute steps to midnight on the intraday forecast: thermal by cost, renewables used up to availability, one binary per battery per step so it never charges and discharges at once, state of charge with 92% efficiency ending no emptier than it began, load shed and line overload as costly slack; line limits through the PTDF of the current topology, added in rounds wherever the solution breaks one. Scoring runs every plan on the true weather with thermal units rebalancing cheapest first and true DC flows"}
    return run(ctx, idem, body, work)


@app.get("/v1/plans/{plan_id}", tags=["dispatch"], summary="A stored dispatch plan: objective, summary and setpoints by asset type and step")
def get_plan(plan_id: uuid.UUID, ctx: Ctx = Depends(auth("grid:read"))):
    with db.tx(ctx.tenant_id) as c:
        grid(c)
        p = c.execute("SELECT * FROM dispatch_plans WHERE id = %s", [plan_id]).fetchone()
        if not p:
            raise Problem(404, "plan_not_found")
        steps = c.execute("SELECT a.asset_type, s.ts, sum(s.setpoint_mw) AS mw, count(*) AS assets FROM dispatch_steps s JOIN assets a ON a.id = s.asset_id WHERE s.plan_id = %s GROUP BY 1, 2 ORDER BY 2, 1", [plan_id]).fetchall()
    by = {}
    for r in steps:
        by.setdefault(r["asset_type"], []).append({"time": r["ts"].strftime("%H:%M"), "mw": round(float(r["mw"]), 1)})
    return jsonable_encoder({"plan_id": plan_id, "status": p["status"], "created_at": p["created_at"], "horizon_start": p["horizon_start"].strftime("%H:%M"), "interval_min": p["interval_min"], "objective": p["objective"], "summary": p["summary"], "setpoints_by_type": by})
