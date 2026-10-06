"""Measure the forecasts, the intraday update, the dispatch strategies and the N-1 scan on networks the demo never uses. Writes
docs/evaluation.md.

    python -m rg.evaluate
"""
import pathlib

import numpy as np

from . import engine, world as W

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "evaluation.md"
SEEDS = (101, 102, 103)
STRAT = [("no_battery_network_blind", "network-blind, no battery"), ("price_battery_network_blind", "network-blind, battery on price"), ("plan_as_made", "12:00 plan, as made"), ("rolling_no_batteries", "re-planned hourly, no batteries"), ("rolling_with_batteries", "re-planned hourly, with batteries"), ("perfect_knowledge", "perfect knowledge")]


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    fc_rows, now_rows, strat_rows, n1_rows, simul = [], [], [], [], 0.0
    for seed in SEEDS:
        w = W.build(seed)
        rep = engine.models(seed)["report"]
        fc_rows.append([seed, rep["solar"]["mae_mw"], rep["solar"]["mae_persistence_mw"], rep["solar"]["mae_clear_sky_mw"], rep["wind"]["mae_mw"], rep["wind"]["mae_persistence_mw"], rep["wind"]["mae_power_curve_mw"], rep["load"]["mae_mw"], rep["load"]["mae_same_step_last_week_mw"], f"{rep['load']['inside_80pct_band']:.0%}"])
        t0 = W.CLOCK_STEP
        da, ta, tr = engine.day_ahead(seed), engine.truth(seed, 0), engine.truth(seed, t0)
        n = W.STEPS - t0
        sol, ratio = engine.nowcast(da["solar"][:, t0:].sum(0), ta["solar"][:, t0 - 4:t0].sum(0), da["solar"][:, t0 - 4:t0].sum(0), n)
        now_rows.append([seed, f"{ratio:.0%}", round(float(da["solar"][:, t0:].sum()) * W.DT_H), round(float(sol.sum()) * W.DT_H), round(float(tr["solar"].sum()) * W.DT_H)])
        out = (w["demo_outage"]["line"],)
        res = {"no_battery_network_blind": engine.score(w, tr, np.zeros((len(w["batteries"]), n)), out), "price_battery_network_blind": engine.score(w, tr, engine.price_schedule(w, tr["price"]), out)}
        ss = da["solar"][:, t0:] / np.maximum(1e-9, da["solar"][:, t0:].sum(0)) * sol
        win, _ = engine.nowcast(da["wind"][:, t0:].sum(0), ta["wind"][:, t0 - 4:t0].sum(0), da["wind"][:, t0 - 4:t0].sum(0), n)
        ws = da["wind"][:, t0:] / np.maximum(1e-9, da["wind"][:, t0:].sum(0)) * win
        lod, _ = engine.nowcast(da["load"][t0:], ta["load"][t0 - 4:t0], da["load"][t0 - 4:t0], n)
        p = engine.optimize(w, ss, ws, lod, out)
        simul = max(simul, p["simultaneous_charge_discharge"])
        res["plan_as_made"] = engine.score(w, tr, p["discharge"] - p["charge"], out, cap_solar=p["solar"] + 1e-6, cap_wind=p["wind"] + 1e-6, thermal_plan=p["thermal"])
        for label, ub in (("rolling_no_batteries", False), ("rolling_with_batteries", True)):
            r = engine.rolling(seed, t0, out, use_batteries=ub)
            res[label] = engine.score(w, tr, r["battery"], out, cap_solar=r["cap_solar"] + 1e-6, cap_wind=r["cap_wind"] + 1e-6, thermal_plan=r["thermal"])
        pp = engine.optimize(w, tr["solar"], tr["wind"], tr["load"], out)
        simul = max(simul, pp["simultaneous_charge_discharge"])
        res["perfect_knowledge"] = engine.score(w, tr, pp["discharge"] - pp["charge"], out, cap_solar=pp["solar"] + 1e-6, cap_wind=pp["wind"] + 1e-6, thermal_plan=pp["thermal"])
        for k, name in STRAT:
            s = res[k]
            strat_rows.append([seed, name, f"{s['cost']:,}", f"{s['renewable_utilisation']:.1%}", f"{s['curtailed_mwh']:,}", s["overload_minutes"], f"{s['worst_loading']:.0%}"])
        d0 = W.DEMO_DAY * W.STEPS
        inj = W.injections(w, W.merit_order(w, d0 + 60), d0 + 60)
        a = engine.n_minus_1(w, inj, (), top=1)
        b = engine.n_minus_1(w, inj, out, top=1)
        n1_rows.append([seed, w["demo_outage"]["line"], a["islanding"], a["causing_new_overload"], len(b["already_overloaded"]), b["causing_new_overload"]])
    doc = f"""# Evaluation

Everything here is measured on synthetic networks generated with seeds the demo does not use (`{SEEDS}`), each with the same kind of day:
strong wind in the north the forecast saw, a cloud bank over the solar region it did not, and the line whose loss most overloads the wind
export corridor tripping at 12:00. Reproduce with `python -m rg.evaluate` (about a minute). The twin that scores the plans is the
generator itself, with DC power flow as the physics; on a real network the scoring would be an approximation and the gaps wider.

## 1. Forecasts, on the last ten days of history

Mean error in MW. Persistence is the same time yesterday; the physical formulas are clear-sky output for solar and the power curve on
forecast wind speed; load's baseline is the same time last week.

{table(["network (seed)", "solar", "persistence", "clear sky", "wind", "persistence", "power curve", "load", "last week", "load inside 80% band"], fc_rows)}

The models beat persistence by a wide margin and the clear-sky formula easily. Wind only matches the power curve, which is how the
generator makes wind: the model learns the formula back, and the page says so.

## 2. The intraday update at 12:00

After an hour of cloud, the last hour's actual solar as a share of the day-ahead forecast, and the rest of the day's solar energy by
the day-ahead forecast, the update and the truth.

{table(["network (seed)", "last hour, actual / forecast", "day-ahead (MWh)", "intraday update (MWh)", "truth (MWh)"], now_rows)}

The update closes about two thirds of the gap between the day-ahead forecast and the truth. It assumes the cloud clears over a couple
of hours; this one stays four more, and a forecaster who could see the cloud bank on a satellite image would do better than any ratio.

## 3. Running the afternoon, scored on the true weather and true flows

{table(["network (seed)", "strategy", "cost ($)", "renewables used", "curtailed (MWh)", "overload minutes", "worst loading"], strat_rows)}

Utilisation is renewables used over what the true weather made available. Running blind to the network is always cheapest and uses
everything, because it ignores the corridor: eight to eleven hours with a line at 274–449% of its limit, which protection would not
permit. A battery scheduled on price alone can make that worse (449% on network 102), because it discharges into the evening peak
wherever it sits. Respecting the limits costs 14–39% more and curtails 14–29% of the renewables, depending on how hard the outage bites.
The plans still show 15–165 overload minutes on the true weather, all within 110% of a limit: the forecast was wrong, and the scoring
rebalances the difference with thermal units cheapest first, blind to the network, as a control room without a re-plan would.

Two results did not go the way the design expected. Re-planning every hour from fresh telemetry did not beat the single plan made at
12:00: it curtailed more on two of the three networks, because every re-plan carries the same assumption that the cloud is about to
clear, and each one re-secures the corridor against a forecast that is then wrong again. And the batteries are worth 0–2 points of
utilisation: only the ones behind the constraint help, and on network 103 the optimiser did not use them at all. Perfect knowledge of
the weather sits 3–6 points above the best plan; that is what the forecast error costs. Across every solve the largest simultaneous charge
and discharge was {simul:.1e} MW: the binary per battery per step does its job.

## 4. N-1

At the windy afternoon step: from the intact network, how many single outages island buses and how many add an overload; then from the
network with the demo line out, how many lines are already over and how many further outages add another.

{table(["network (seed)", "demo outage line", "islanding (intact)", "adding an overload (intact)", "already over (line out)", "adding an overload (line out)"], n1_rows)}

## 5. What is not measured

- AC power flow, voltages, reactive power, stability: DC flow only.
- Security-constrained dispatch (every N-1 case enforced inside the dispatch); the scan ranks outages, the dispatch enforces the current topology.
- Real weather, real networks, real markets.
"""
    OUT.write_text(doc)
    print(doc)


if __name__ == "__main__":
    main()
