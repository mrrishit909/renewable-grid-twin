"""Forecasts (solar, wind, load) with baselines, an intraday update from telemetry, the battery and curtailment dispatch as a MILP over
the DC network with line limits added as they bind, N-1 contingency ranking, and every plan scored on the true weather."""
import functools

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.ensemble import HistGradientBoostingRegressor

from . import world as W

VERSION = "grid-twin-1"
HOLDOUT_DAYS = 10


def _feat_solar(wd, idx):
    steps = idx % W.STEPS
    return np.column_stack([wd["clear_sky"][idx], wd["cloud_fc"]["west"][idx], steps * W.DT_H])


def _feat_wind(wd, idx):
    steps = idx % W.STEPS
    return np.column_stack([wd["speed_fc"]["north"][idx], steps * W.DT_H])


def _feat_load(wd, idx):
    steps = idx % W.STEPS
    return np.column_stack([steps * W.DT_H, (idx // W.STEPS) % 7, wd["temp_fc"][idx]])


@functools.lru_cache(maxsize=4)
def models(seed):
    """Fit on the history minus a holdout; report holdout error against baselines; refit on the whole history for use."""
    w = W.build(seed)
    wd = w["weather"]
    hist = np.arange(W.HISTORY_DAYS * W.STEPS)
    cut = (W.HISTORY_DAYS - HOLDOUT_DAYS) * W.STEPS
    tr, te = hist[:cut], hist[cut:]
    sol_cap, wind_cap = sum(s["cap"] for s in w["solar"]), sum(x["cap"] for x in w["wind"])
    ys = w["solar_out"].sum(axis=0) / sol_cap
    yw = w["wind_out"].sum(axis=0) / wind_cap
    yl = w["load"]
    hgb = lambda **kw: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=5, random_state=0, **kw)     # noqa: E731
    rep = {}
    ms = hgb().fit(_feat_solar(wd, tr), ys[tr])
    ps = np.clip(ms.predict(_feat_solar(wd, te)), 0, 1)
    rep["solar"] = {"mae_mw": round(float(np.abs(ps - ys[te]).mean() * sol_cap), 1), "mae_persistence_mw": round(float(np.abs(ys[te - W.STEPS] - ys[te]).mean() * sol_cap), 1), "mae_clear_sky_mw": round(float(np.abs(wd["clear_sky"][te] - ys[te]).mean() * sol_cap), 1), "capacity_mw": round(sol_cap)}
    mw = hgb().fit(_feat_wind(wd, tr), yw[tr])
    pw = np.clip(mw.predict(_feat_wind(wd, te)), 0, 1)
    curve = W.wind_curve(wd["speed_fc"]["north"][te])
    rep["wind"] = {"mae_mw": round(float(np.abs(pw - yw[te]).mean() * wind_cap), 1), "mae_persistence_mw": round(float(np.abs(yw[te - W.STEPS] - yw[te]).mean() * wind_cap), 1), "mae_power_curve_mw": round(float(np.abs(curve - yw[te]).mean() * wind_cap), 1), "capacity_mw": round(wind_cap)}
    ml = {q: hgb(loss="quantile", quantile=q).fit(_feat_load(wd, tr), yl[tr]) for q in (0.1, 0.5, 0.9)}
    pl = {q: m.predict(_feat_load(wd, te)) for q, m in ml.items()}
    rep["load"] = {"mae_mw": round(float(np.abs(pl[0.5] - yl[te]).mean()), 1), "mae_same_step_last_week_mw": round(float(np.abs(yl[te - 7 * W.STEPS] - yl[te]).mean()), 1), "inside_80pct_band": round(float(np.mean((yl[te] >= pl[0.1]) & (yl[te] <= pl[0.9]))), 2)}
    full = {"solar": hgb().fit(_feat_solar(wd, hist), ys[hist]), "wind": hgb().fit(_feat_wind(wd, hist), yw[hist]), "load": {q: hgb(loss="quantile", quantile=q).fit(_feat_load(wd, hist), yl[hist]) for q in (0.1, 0.5, 0.9)}}
    return {"models": full, "report": rep, "sol_cap": sol_cap, "wind_cap": wind_cap}


def day_ahead(seed):
    """The demo day's forecasts, issued at 00:00 from forecast weather: per-site solar and wind availability and system load with a band."""
    w = W.build(seed)
    m = models(seed)
    idx = np.arange(W.DEMO_DAY * W.STEPS, (W.DEMO_DAY + 1) * W.STEPS)
    ps = np.clip(m["models"]["solar"].predict(_feat_solar(w["weather"], idx)), 0, 1)
    pw = np.clip(m["models"]["wind"].predict(_feat_wind(w["weather"], idx)), 0, 1)
    load = {q: mm.predict(_feat_load(w["weather"], idx)) for q, mm in m["models"]["load"].items()}
    sol_share = np.array([s["cap"] for s in w["solar"]]) / m["sol_cap"]
    wind_share = np.array([x["cap"] for x in w["wind"]]) / m["wind_cap"]
    return {"solar": np.outer(sol_share, ps * m["sol_cap"]), "wind": np.outer(wind_share, pw * m["wind_cap"]), "load": load[0.5], "load_p10": load[0.1], "load_p90": load[0.9]}


def nowcast(fc_row, actual_recent, fc_recent, steps_ahead, half_life=8):
    """Scale the remaining forecast by the ratio actual/forecast over the last hour, decaying back to the forecast (half-life in steps)."""
    ratio = float(np.sum(actual_recent) / max(1e-6, np.sum(fc_recent)))
    decay = 0.5 ** (np.arange(steps_ahead) / half_life)
    return fc_row * (1 + (ratio - 1) * decay), ratio


def truth(seed, t0):
    """The demo day's real availability and load from step t0 to the end of the day."""
    w = W.build(seed)
    d0 = W.DEMO_DAY * W.STEPS
    return {"solar": w["solar_out"][:, d0 + t0:d0 + W.STEPS], "wind": w["wind_out"][:, d0 + t0:d0 + W.STEPS], "load": w["load"][d0 + t0:d0 + W.STEPS], "price": w["price"][d0 + t0:d0 + W.STEPS]}


# ---------------------------------------------------------------- dispatch

def optimize(w, solar, wind, load, out=(), soc0=None, max_rounds=8, time_limit=60, use_batteries=True, soc_end=None):
    """MILP over n steps: thermal, renewables used (the rest curtailed), battery charge/discharge (never both: one binary per battery
    per step), state of charge, load shed, and line limits through the PTDF, added in rounds as they bind. -> plan"""
    n = load.shape[0]
    U, S, Wn, B = len(w["thermal"]), solar.shape[0], wind.shape[0], len(w["batteries"])
    nv = U + S + Wn + 3 * B + B + 1                                        # per step: g, s, w, c, d, e, z, shed
    o_g, o_s, o_w, o_c, o_d, o_e, o_z, o_sh = 0, U, U + S, U + S + Wn, U + S + Wn + B, U + S + Wn + 2 * B, U + S + Wn + 3 * B, U + S + Wn + 4 * B
    P = W.ptdf(w, out)
    caps = np.array([ln["cap"] for ln in w["lines"]])
    share = np.zeros(W.N_BUS)
    for bus, sh in w["load_share"].items():
        share[bus] = sh
    pg = np.array([P[:, u["bus"]] for u in w["thermal"]]).T                # lines x units
    ps_ = np.array([P[:, s["bus"]] for s in w["solar"]]).T
    pw_ = np.array([P[:, x["bus"]] for x in w["wind"]]).T
    pb = np.array([P[:, b["bus"]] for b in w["batteries"]]).T
    pl = P @ share                                                         # flow per MW of load served (negative injection)
    soc0 = soc0 if soc0 is not None else [b["soc0"] * b["energy"] for b in w["batteries"]]
    N = n * nv
    c = np.zeros(N)
    lb, ub = np.zeros(N), np.full(N, np.inf)
    integ = np.zeros(N)
    rows, cols, vals, rlb, rub = [], [], [], [], []

    def add(coefs, lo, hi):
        r = len(rlb)
        for j, v in coefs:
            rows.append(r)
            cols.append(j)
            vals.append(v)
        rlb.append(lo)
        rub.append(hi)
    for t in range(n):
        base = t * nv
        for k, u in enumerate(w["thermal"]):
            ub[base + o_g + k] = u["cap"]
            c[base + o_g + k] = u["cost"] * W.DT_H
        for k in range(S):
            ub[base + o_s + k] = solar[k, t]
            c[base + o_s + k] = -W.CURTAIL_COST * W.DT_H                  # using renewables avoids the curtailment penalty
        for k in range(Wn):
            ub[base + o_w + k] = wind[k, t]
            c[base + o_w + k] = -W.CURTAIL_COST * W.DT_H
        for k, b in enumerate(w["batteries"]):
            ub[base + o_c + k] = b["power"] if use_batteries else 0.0
            ub[base + o_d + k] = b["power"] if use_batteries else 0.0
            lb[base + o_e + k], ub[base + o_e + k] = 0.1 * b["energy"], 0.95 * b["energy"]
            ub[base + o_z + k] = 1
            integ[base + o_z + k] = 1
            c[base + o_c + k] = c[base + o_d + k] = W.DEGRADATION * W.DT_H
            add([(base + o_c + k, 1.0), (base + o_z + k, -b["power"])], -np.inf, 0)           # charge only when z = 1
            add([(base + o_d + k, 1.0), (base + o_z + k, b["power"])], -np.inf, b["power"])   # discharge only when z = 0
            prev = [(base - nv + o_e + k, -1.0)] if t else []
            add([(base + o_e + k, 1.0), (base + o_c + k, -b["eff"] * W.DT_H), (base + o_d + k, W.DT_H / b["eff"])] + prev, soc0[k] if not t else 0, soc0[k] if not t else 0)
        ub[base + o_sh] = load[t]
        c[base + o_sh] = W.SHED_COST * W.DT_H
        add([(base + o_g + k, 1) for k in range(U)] + [(base + o_s + k, 1) for k in range(S)] + [(base + o_w + k, 1) for k in range(Wn)] + [(base + o_d + k, 1) for k in range(B)] + [(base + o_c + k, -1) for k in range(B)] + [(base + o_sh, 1)], load[t], load[t])
    soc_end = soc_end if soc_end is not None else soc0
    for k, b in enumerate(w["batteries"]):
        add([((n - 1) * nv + o_e + k, 1.0)], min(soc_end[k], 0.95 * b["energy"]), np.inf)     # end the day no emptier than it began (the day's start, not this re-plan's)
    added, rounds, ov_cols = set(), 0, []
    while True:
        Nt = N + len(ov_cols)
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(rlb), Nt))
        cc = np.concatenate([c, np.full(len(ov_cols), W.OVERLOAD_COST * W.DT_H)])
        res = milp(cc, constraints=LinearConstraint(A, rlb, rub), integrality=np.concatenate([integ, np.zeros(len(ov_cols))]), bounds=Bounds(np.concatenate([lb, np.zeros(len(ov_cols))]), np.concatenate([ub, np.full(len(ov_cols), np.inf)])), options={"time_limit": time_limit, "mip_rel_gap": 1e-3})
        if res.x is None:
            raise RuntimeError(f"dispatch solve failed: {res.message}")
        x = res.x
        flows = np.zeros((n, len(w["lines"])))
        for t in range(n):
            base = t * nv
            served = load[t] - x[base + o_sh]
            flows[t] = pg @ x[base + o_g:base + o_g + U] + ps_ @ x[base + o_s:base + o_s + S] + pw_ @ x[base + o_w:base + o_w + Wn] + pb @ (x[base + o_d:base + o_d + B] - x[base + o_c:base + o_c + B]) - pl * served
        viol = [(t, l) for t in range(n) for l in np.where(np.abs(flows[t]) > caps * 1.0005)[0] if (t, int(l)) not in added and int(l) not in out]
        rounds += 1
        if not viol or rounds >= max_rounds:
            break
        for t, l in viol:
            added.add((t, int(l)))
            base = t * nv
            ov = N + len(ov_cols)
            ov_cols.append((t, int(l)))
            coefs = [(base + o_g + k, pg[l, k]) for k in range(U)] + [(base + o_s + k, ps_[l, k]) for k in range(S)] + [(base + o_w + k, pw_[l, k]) for k in range(Wn)] + [(base + o_d + k, pb[l, k]) for k in range(B)] + [(base + o_c + k, -pb[l, k]) for k in range(B)] + [(base + o_sh, pl[l])]
            const = -pl[l] * load[t]
            add(coefs + [(ov, -1.0)], -np.inf, caps[l] - const)
            add(coefs + [(ov, 1.0)], -caps[l] - const, np.inf)
    X = x[:N].reshape(n, nv)
    plan = {"thermal": X[:, o_g:o_g + U].T, "solar": X[:, o_s:o_s + S].T, "wind": X[:, o_w:o_w + Wn].T, "charge": X[:, o_c:o_c + B].T, "discharge": X[:, o_d:o_d + B].T, "soc": X[:, o_e:o_e + B].T, "shed": X[:, o_sh], "flows": flows,
            "objective": float(res.fun), "rounds": rounds, "line_constraints": len(added), "status": res.message, "simultaneous_charge_discharge": float(np.max(np.minimum(X[:, o_c:o_c + B], X[:, o_d:o_d + B])))}
    return plan


# ---------------------------------------------------------------- scoring on the truth

def price_schedule(w, price, soc0=None):
    """Baseline: each battery charges in the cheapest steps and discharges in the dearest, as much as power and energy allow, ignoring the network."""
    n = len(price)
    sched = np.zeros((len(w["batteries"]), n))
    order = np.argsort(price)
    for k, b in enumerate(w["batteries"]):
        e = (soc0[k] if soc0 is not None else b["soc0"] * b["energy"])
        steps = int(min(n // 3, (0.95 * b["energy"] - e) / (b["power"] * W.DT_H * b["eff"]) + n // 6))
        cheap, dear = order[:steps], order[::-1][:steps]
        energy = e
        for t in range(n):
            if t in set(cheap) and energy < 0.95 * b["energy"]:
                p = min(b["power"], (0.95 * b["energy"] - energy) / (W.DT_H * b["eff"]))
                sched[k, t] = -p
                energy += p * W.DT_H * b["eff"]
            elif t in set(dear) and energy > 0.1 * b["energy"]:
                p = min(b["power"], (energy - 0.1 * b["energy"]) * b["eff"] / W.DT_H)
                sched[k, t] = p
                energy -= p * W.DT_H / b["eff"]
    return sched


def score(w, tru, battery, out=(), cap_solar=None, cap_wind=None, thermal_plan=None):
    """Run the real day: renewables at their true availability (capped by a plan's curtailment if given), batteries as scheduled, thermal
    units balancing what is left cheapest first (from the plan's thermal if given), then true DC flows. -> KPIs"""
    n = len(tru["load"])
    caps = np.array([ln["cap"] for ln in w["lines"]])
    cost = shed = used = avail = curtailed = overload_steps = overload_mwh = 0.0
    worst = 0.0
    for t in range(n):
        s = tru["solar"][:, t] if cap_solar is None else np.minimum(tru["solar"][:, t], cap_solar[:, t])
        x = tru["wind"][:, t] if cap_wind is None else np.minimum(tru["wind"][:, t], cap_wind[:, t])
        d = W.merit_order(w, 0, solar=s, wind=x, load=tru["load"][t], battery=battery[:, t])
        if thermal_plan is not None:                                       # keep the plan's thermal where possible, adjust the rest cheapest first
            need = tru["load"][t] - d["solar"].sum() - d["wind"].sum() - battery[:, t].sum()
            th = np.minimum(thermal_plan[:, t], [u["cap"] for u in w["thermal"]]).astype(float)
            gap = need - th.sum()
            order = np.argsort([u["cost"] for u in w["thermal"]])
            for k in (order if gap > 0 else order[::-1]):
                step = min(w["thermal"][k]["cap"] - th[k], gap) if gap > 0 else max(-th[k], gap)
                th[k] += step
                gap -= step
            d["thermal"], d["shed"] = th, max(0.0, gap)
        inj = W.injections(w, d, 0)
        f = np.abs(W.dc_flow(w, inj, out))
        over = np.maximum(0, f - caps)
        over[list(out)] = 0
        if over.max() > 0.5:
            overload_steps += 1
            overload_mwh += over.sum() * W.DT_H
        worst = max(worst, float(np.max(np.where(np.isin(np.arange(len(caps)), list(out)), 0, f / caps))))
        cost += sum(u["cost"] * g for u, g in zip(w["thermal"], d["thermal"])) * W.DT_H + W.SHED_COST * d["shed"] * W.DT_H + W.DEGRADATION * np.abs(battery[:, t]).sum() * W.DT_H
        shed += d["shed"] * W.DT_H
        a = tru["solar"][:, t].sum() + tru["wind"][:, t].sum()
        u_ = d["solar"].sum() + d["wind"].sum()
        avail += a * W.DT_H
        used += u_ * W.DT_H
        curtailed += (a - u_) * W.DT_H
    return {"cost": int(round(cost)), "renewable_utilisation": round(float(used / max(1e-9, avail)), 4), "renewable_used_mwh": int(round(used)), "curtailed_mwh": int(round(curtailed)), "shed_mwh": round(float(shed), 1), "overload_minutes": int(overload_steps * 15), "overload_mwh": round(float(overload_mwh), 1), "worst_loading": round(float(worst), 2)}


def n_minus_1(w, inj, base_out=(), top=10):
    """Every further single-line outage from the current topology: does it island buses, and which lines does it newly push over their
    limit (lines already over in the current state are reported once, separately). Ranked by new overloads, then worst loading."""
    caps = np.array([ln["cap"] for ln in w["lines"]])
    base = np.abs(W.dc_flow(w, inj, base_out)) / caps
    already = {int(k) for k in np.where(base > 1)[0] if int(k) not in base_out}
    out = []
    for ln in w["lines"]:
        if ln["idx"] in base_out:
            continue
        r = W.contingency(w, inj, ln["idx"], base_out)
        if r["islanded"]:
            out.append({"line": ln["idx"], "islanded": True, "buses_cut_off": r["buses_cut_off"], "worst_loading": None, "new_overloads": 0})
            continue
        lo = r["loading"]
        new = {k: v for k, v in lo.items() if v > 1 and k not in already}
        out.append({"line": ln["idx"], "islanded": False, "buses_cut_off": 0, "worst_loading": round(max(lo.values()), 2) if lo else 0.0, "worst_line": max(lo, key=lo.get) if lo else None, "new_overloads": len(new)})
    ranked = sorted(out, key=lambda x: (-x["new_overloads"], -(x["worst_loading"] or 0)))
    return {"lines_tested": len(out), "already_overloaded": sorted(already), "islanding": sum(1 for x in out if x["islanded"]), "causing_new_overload": sum(1 for x in out if x["new_overloads"]), "top": ranked[:top]}


def rolling(seed, t0, out=(), use_batteries=True, every=4):
    """Re-plan every hour from t0: nowcast solar, wind and load from the last hour's telemetry, solve to the end of the day, apply the first
    hour, carry the batteries' state forward. -> schedule (batteries, renewable caps, thermal) for scoring, and how many solves it took"""
    w = W.build(seed)
    da = day_ahead(seed)
    ta = truth(seed, 0)
    B = len(w["batteries"])
    n = W.STEPS - t0
    bat, cap_s, cap_w, th = np.zeros((B, n)), np.zeros((len(w["solar"]), n)), np.zeros((len(w["wind"]), n)), np.zeros((len(w["thermal"]), n))
    soc = [b["soc0"] * b["energy"] for b in w["batteries"]]
    solves = 0
    for k in range(t0, W.STEPS, every):
        m = W.STEPS - k
        sol, _ = nowcast(da["solar"][:, k:].sum(0), ta["solar"][:, k - 4:k].sum(0), da["solar"][:, k - 4:k].sum(0), m)
        win, _ = nowcast(da["wind"][:, k:].sum(0), ta["wind"][:, k - 4:k].sum(0), da["wind"][:, k - 4:k].sum(0), m)
        lod, _ = nowcast(da["load"][k:], ta["load"][k - 4:k], da["load"][k - 4:k], m)
        ss = da["solar"][:, k:] / np.maximum(1e-9, da["solar"][:, k:].sum(0)) * sol
        ws = da["wind"][:, k:] / np.maximum(1e-9, da["wind"][:, k:].sum(0)) * win
        p = optimize(w, ss, ws, lod, out, soc0=soc, use_batteries=use_batteries, soc_end=[b["soc0"] * b["energy"] for b in w["batteries"]])
        solves += 1
        h = min(every, m)
        i = k - t0
        bat[:, i:i + h] = (p["discharge"] - p["charge"])[:, :h]
        cap_s[:, i:i + h], cap_w[:, i:i + h], th[:, i:i + h] = p["solar"][:, :h], p["wind"][:, :h], p["thermal"][:, :h]
        for j, b in enumerate(w["batteries"]):
            for s in range(h):
                soc[j] += (p["charge"][j, s] * b["eff"] - p["discharge"][j, s] / b["eff"]) * W.DT_H
    return {"battery": bat, "cap_solar": cap_s, "cap_wind": cap_w, "thermal": th, "solves": solves}
