"""Unit tests for the network, DC power flow, the forecasts, the dispatch MILP and the scoring: no database."""
import numpy as np
import pytest

from rg import engine, world as W

G = W.build()


def test_network_is_connected_meshed_and_the_intact_grid_is_secure_on_history():
    assert len(G["lines"]) == 180 and len(set(W.connected(G))) == 1
    radial = [ln["idx"] for ln in G["lines"] if len(set(W.connected(G, (ln["idx"],)))) > 1]
    assert 0 < len(radial) < 30 and G["demo_outage"]["line"] not in radial
    caps = np.array([ln["cap"] for ln in G["lines"]])
    for t in range(0, W.HISTORY_DAYS * W.STEPS, 97):
        assert (np.abs(W.dc_flow(G, W.injections(G, W.merit_order(G, t), t))) <= caps + 1e-6).all()


def test_dc_flow_matches_a_direct_angle_solve_and_islanding_is_refused():
    t = W.DEMO_DAY * W.STEPS + 50
    inj = W.injections(G, W.merit_order(G, t), t)
    assert abs(inj.sum()) < 1e-6
    B = np.zeros((W.N_BUS, W.N_BUS))
    for ln in G["lines"]:
        b = 1 / ln["x"]
        i, j = ln["from"], ln["to"]
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b
    theta = np.zeros(W.N_BUS)
    theta[1:] = np.linalg.solve(B[1:, 1:], inj[1:])
    direct = np.array([(theta[ln["from"]] - theta[ln["to"]]) / ln["x"] for ln in G["lines"]])
    assert np.allclose(direct, W.dc_flow(G, inj), atol=1e-6)
    for bus in range(W.N_BUS):                                              # flow conservation at every bus
        net = sum(-f if ln["to"] == bus else f if ln["from"] == bus else 0 for ln, f in zip(G["lines"], direct))
        assert abs(net - inj[bus]) < 1e-6
    radial = next(ln["idx"] for ln in G["lines"] if len(set(W.connected(G, (ln["idx"],)))) > 1)
    with pytest.raises(ValueError, match="islanded"):
        W.ptdf_for(G, (radial,))
    assert W.contingency(G, inj, radial)["islanded"]


def test_outage_overloads_the_corridor_and_n_minus_1_counts_only_new_overloads():
    t = W.DEMO_DAY * W.STEPS + 60
    inj = W.injections(G, W.merit_order(G, t), t)
    out = (G["demo_outage"]["line"],)
    caps = np.array([ln["cap"] for ln in G["lines"]])
    assert (np.abs(W.dc_flow(G, inj)) <= caps + 1e-6).all() and (np.abs(W.dc_flow(G, inj, out)) > caps * 1.5).any()
    r = engine.n_minus_1(G, inj, out, top=5)
    assert r["lines_tested"] == 179 and r["already_overloaded"] and r["islanding"] > 0 and r["causing_new_overload"] < r["lines_tested"]


def test_forecasts_beat_persistence_and_the_cloud_is_a_real_miss():
    rep = engine.models(17)["report"]
    assert rep["solar"]["mae_mw"] < rep["solar"]["mae_persistence_mw"] and rep["wind"]["mae_mw"] < rep["wind"]["mae_persistence_mw"] and rep["load"]["mae_mw"] < rep["load"]["mae_same_step_last_week_mw"]
    da, tr = engine.day_ahead(17), engine.truth(17, 0)
    cb = W.CLOUD_BANK
    assert da["solar"][:, cb["from"]:cb["to"]].sum() > 2 * tr["solar"][:, cb["from"]:cb["to"]].sum()
    nc, ratio = engine.nowcast(np.full(10, 100.0), np.full(4, 40.0), np.full(4, 100.0), 10)
    assert abs(ratio - 0.4) < 1e-9 and nc[0] == pytest.approx(40.0) and 40 < nc[-1] < 100


def test_milp_respects_limits_and_never_charges_while_discharging():
    t0 = W.CLOCK_STEP
    tr = engine.truth(17, t0)
    out = (G["demo_outage"]["line"],)
    p = engine.optimize(G, tr["solar"], tr["wind"], tr["load"], out)
    caps = np.array([ln["cap"] for ln in G["lines"]])
    assert p["simultaneous_charge_discharge"] < 1e-6 and (np.abs(p["flows"]) <= caps * 1.001 + 0.5).all() and p["shed"].max() < 1e-6
    for k, b in enumerate(G["batteries"]):
        assert (p["soc"][k] >= 0.1 * b["energy"] - 1e-6).all() and (p["soc"][k] <= 0.95 * b["energy"] + 1e-6).all() and p["soc"][k, -1] >= b["soc0"] * b["energy"] - 1e-4
    nb = engine.optimize(G, tr["solar"], tr["wind"], tr["load"], out, use_batteries=False)
    assert np.abs(nb["charge"]).max() == 0 and p["objective"] <= nb["objective"] + 1e-6           # batteries never make the optimum worse
    used = p["solar"].sum() + p["wind"].sum()
    s = engine.score(G, tr, p["discharge"] - p["charge"], out, cap_solar=p["solar"] + 1e-6, cap_wind=p["wind"] + 1e-6, thermal_plan=p["thermal"])
    blind = engine.score(G, tr, np.zeros((6, tr["load"].shape[0])), out)
    assert s["overload_minutes"] == 0 and blind["overload_minutes"] > 120 and blind["renewable_utilisation"] == 1.0 and abs(s["renewable_used_mwh"] - used * W.DT_H) < 5
