"""An invented 118-bus transmission network: four regions on a 100 km square, about 180 lines, thermal units in the metro region,
solar in the west, wind in the north, six batteries, loads on 90 buses; sixty days of 15-minute weather (cloud cover and wind speed,
real and as forecast), output, load and prices; and the demo day, with a cloud bank the forecast missed over the solar region and a
line outage on the wind export corridor. DC power flow is the physics. Everything is a pure function of a seed; this is not an IEEE
case, not a real grid and not real weather."""
import functools

import numpy as np

N_BUS = 118
STEPS = 96                                         # 15-minute steps a day
DT_H = 0.25
HISTORY_DAYS = 60
DEMO_DAY = HISTORY_DAYS                            # day index of the demo day
CLOCK_STEP = 48                                    # 12:00: an hour into the cloud bank, so telemetry shows it
REGIONS = {"metro": (60, 35), "west": (20, 45), "north": (55, 85), "east": (88, 55)}
CLOUD_BANK = {"region": "west", "from": 44, "to": 64, "cloud": 0.9}          # 11:00 to 16:00, not in the forecast
SHED_COST, OVERLOAD_COST, CURTAIL_COST, DEGRADATION = 3000.0, 2000.0, 1.0, 8.0     # $/MWh


def clear_sky(step):
    """Clear-sky per-unit solar by step of day: zero at night, a sine from 06:00 to 19:00."""
    h = step * DT_H
    return float(max(0.0, np.sin(np.pi * (h - 6.0) / 13.0))) if 6.0 <= h <= 19.0 else 0.0


def wind_curve(speed):
    """Per-unit wind output from hub speed (m/s): cut-in 3, rated 12, cut-out 25."""
    s = np.asarray(speed, float)
    return np.where(s < 3, 0.0, np.where(s < 12, ((s - 3) / 9) ** 3, np.where(s < 25, 1.0, 0.0)))


@functools.lru_cache(maxsize=4)
def build(seed=17):
    r = np.random.default_rng(seed)
    names = list(REGIONS)
    region = r.choice(names, N_BUS, p=[0.38, 0.22, 0.2, 0.2])
    xy = np.array([np.array(REGIONS[g]) + r.normal(0, 9, 2) for g in region]).clip(0, 100)
    # lines: each bus to its two nearest neighbours, then join components, then extra meshing to about 180
    edges = set()
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    for i in range(N_BUS):
        for j in np.argsort(d[i])[1:3]:
            edges.add((min(i, int(j)), max(i, int(j))))

    def components(es):
        parent = list(range(N_BUS))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, b in es:
            parent[find(a)] = find(b)
        return [find(a) for a in range(N_BUS)]
    comp = components(edges)
    while len(set(comp)) > 1:
        c0 = comp[0]
        ins, outs = [i for i in range(N_BUS) if comp[i] == c0], [i for i in range(N_BUS) if comp[i] != c0]
        sub = d[np.ix_(ins, outs)]
        a, b = np.unravel_index(np.argmin(sub), sub.shape)
        edges.add((min(ins[a], outs[b]), max(ins[a], outs[b])))
        comp = components(edges)
    while len(edges) < 180:
        i = int(r.integers(0, N_BUS))
        j = int(np.argsort(d[i])[int(r.integers(2, 6))])
        edges.add((min(i, j), max(i, j)))
    lines = [{"idx": k, "from": a, "to": b, "km": round(float(d[a, b]), 1), "x": round(0.0004 * max(2.0, float(d[a, b])) + 0.002, 5)} for k, (a, b) in enumerate(sorted(edges))]
    # assets
    by_region = {g: [i for i in range(N_BUS) if region[i] == g] for g in names}
    thermal = [{"bus": int(b), "cap": float(c), "cost": float(m)} for b, c, m in zip(r.choice(by_region["metro"] + by_region["east"], 8, replace=False), r.uniform(250, 600, 8), np.sort(r.uniform(22, 95, 8)))]
    solar = [{"bus": int(b), "cap": float(r.uniform(60, 120)), "region": "west"} for b in r.choice(by_region["west"], 15, replace=False)]
    wind = [{"bus": int(b), "cap": float(r.uniform(70, 120)), "region": "north", "site_factor": float(r.uniform(0.9, 1.15))} for b in r.choice(by_region["north"], 10, replace=False)]
    bat_buses = list(r.choice(by_region["north"], 2, replace=False)) + list(r.choice(by_region["west"], 2, replace=False)) + list(r.choice(by_region["metro"], 2, replace=False))
    batteries = [{"bus": int(b), "power": float(p), "energy": float(p) * 4, "eff": 0.92, "soc0": 0.5} for b, p in zip(bat_buses, r.uniform(50, 110, 6))]
    load_buses = list(r.choice(range(N_BUS), 90, replace=False))
    w = np.array([2.2 if region[b] == "metro" else 0.8 if region[b] == "north" else 1.0 for b in load_buses]) * r.uniform(0.5, 1.5, 90)
    load_share = {int(b): float(x) for b, x in zip(load_buses, w / w.sum())}
    # weather and demand: HISTORY_DAYS + 1 days of 15-minute steps
    T = (HISTORY_DAYS + 1) * STEPS
    cloud, cloud_fc, speed, speed_fc = {}, {}, {}, {}
    for g in names:
        c = np.zeros(T)
        e = np.zeros(T)
        for t in range(1, T):
            c[t] = np.clip(0.94 * c[t - 1] + 0.06 * 0.35 + r.normal(0, 0.06), 0, 1)
            e[t] = 0.9 * e[t - 1] + r.normal(0, 0.03)
        cloud[g], cloud_fc[g] = c, np.clip(c + e, 0, 1)
        s = np.zeros(T)
        es = np.zeros(T)
        s[0] = 8
        for t in range(1, T):
            s[t] = max(0.0, 0.97 * s[t - 1] + 0.03 * (9.5 if g == "north" else 6.5) + r.normal(0, 0.55))
            es[t] = 0.92 * es[t - 1] + r.normal(0, 0.25)
        speed[g], speed_fc[g] = s, np.maximum(0, s + es)
    day0 = DEMO_DAY * STEPS
    # the demo day: strong wind in the north (forecast saw it), a cloud bank over the west (forecast did not)
    speed["north"][day0 + 36:day0 + 80] = np.maximum(speed["north"][day0 + 36:day0 + 80], 13.5 + r.normal(0, 0.6, 44))
    speed_fc["north"][day0 + 36:day0 + 80] = np.maximum(speed_fc["north"][day0 + 36:day0 + 80], 13.0 + r.normal(0, 0.6, 44))
    cloud["west"][day0:day0 + STEPS] = np.minimum(cloud["west"][day0:day0 + STEPS], 0.15)
    cloud_fc["west"][day0:day0 + STEPS] = np.minimum(cloud_fc["west"][day0:day0 + STEPS], 0.15)
    cb = CLOUD_BANK
    cloud["west"][day0 + cb["from"]:day0 + cb["to"]] = cb["cloud"]
    steps = np.arange(T) % STEPS
    cs = np.array([clear_sky(int(s)) for s in steps])
    temp = 18 + 7 * np.sin(2 * np.pi * (steps * DT_H - 9) / 24) + np.repeat(r.normal(0, 3, HISTORY_DAYS + 1), STEPS) + r.normal(0, 0.6, T)
    temp_fc = temp + np.repeat(r.normal(0, 1.2, HISTORY_DAYS + 1), STEPS)
    hour = steps * DT_H
    profile = 0.62 + 0.18 * np.exp(-((hour - 8.5) / 2.2) ** 2) + 0.28 * np.exp(-((hour - 18.5) / 2.8) ** 2)
    dow = np.repeat([1.0, 1.0, 1.0, 1.0, 0.98, 0.88, 0.85] * ((HISTORY_DAYS + 1) // 7 + 1), STEPS)[:T]
    load = 2600 * profile * dow * (1 + 0.012 * np.maximum(0, temp - 22)) * (1 + r.normal(0, 0.015, T))
    solar_out = np.array([[s["cap"] * cs[t] * (1 - 0.75 * cloud["west"][t]) * float(np.clip(1 + r.normal(0, 0.04), 0.8, 1.1)) for t in range(T)] for s in solar])
    wind_out = np.array([w_["cap"] * wind_curve(speed["north"] * w_["site_factor"] + r.normal(0, 0.4, T)) for w_ in wind])
    net = load - solar_out.sum(axis=0) - wind_out.sum(axis=0)
    price = np.maximum(-10.0, 18 + 0.025 * net + r.normal(0, 4, T))
    world = {"seed": seed, "xy": xy.round(2).tolist(), "region": [str(g) for g in region], "lines": lines, "thermal": thermal, "solar": solar, "wind": wind, "batteries": batteries, "load_share": load_share,
             "weather": {"cloud": cloud, "cloud_fc": cloud_fc, "speed": speed, "speed_fc": speed_fc, "temp": temp, "temp_fc": temp_fc, "clear_sky": cs}, "load": load, "solar_out": solar_out, "wind_out": wind_out, "price": price}
    # line limits: comfortably above what the base network carries under a network-blind merit order on the history, so the intact grid
    # is secure on ordinary days; then the demo outage is the line whose loss overloads the most on the demo day's windy afternoon
    flows_hist = []
    for t in range(0, HISTORY_DAYS * STEPS, 7):
        inj = injections(world, merit_order(world, t), t)
        flows_hist.append(np.abs(dc_flow(world, inj)))
    peak = np.max(np.array(flows_hist), axis=0)
    for k, ln in enumerate(lines):
        ln["cap"] = round(float(max(60.0, peak[k] * r.uniform(1.15, 1.45))), 0)
    t_demo = day0 + 60
    inj = injections(world, merit_order(world, t_demo), t_demo)
    best = None
    for ln in lines:
        res = contingency(world, inj, ln["idx"])
        if res["islanded"]:
            continue
        worst = max(res["loading"].items(), key=lambda kv: kv[1], default=(None, 0))
        corridor = worst[0] is not None and (world["region"][lines[worst[0]]["from"]] == "north" or world["region"][lines[worst[0]]["to"]] == "north")
        if corridor and (best is None or worst[1] > best[1]):
            best = (ln["idx"], worst[1], worst[0])
    world["demo_outage"] = {"line": best[0], "at_step": CLOCK_STEP, "worst_line": best[2], "worst_loading": round(best[1], 2)}
    return world


# ---------------------------------------------------------------- physics

def connected(world, out=()):
    """Bus -> component label with the lines in `out` removed."""
    adj = [[] for _ in range(N_BUS)]
    for ln in world["lines"]:
        if ln["idx"] not in out:
            adj[ln["from"]].append(ln["to"])
            adj[ln["to"]].append(ln["from"])
    label = [-1] * N_BUS
    c = 0
    for s in range(N_BUS):
        if label[s] >= 0:
            continue
        stack = [s]
        label[s] = c
        while stack:
            a = stack.pop()
            for b in adj[a]:
                if label[b] < 0:
                    label[b] = c
                    stack.append(b)
        c += 1
    return label


def ptdf_for(world, out=()):
    """PTDF (lines x buses) for the topology with `out` removed, slack at bus 0. Lines out of service get zero rows. Raises if islanded."""
    lab = connected(world, out)
    if len(set(lab)) > 1:
        raise ValueError(f"islanded: {sum(1 for x in lab if x != lab[0])} buses cut off")
    L = len(world["lines"])
    B = np.zeros((N_BUS, N_BUS))
    A = np.zeros((L, N_BUS))
    bvec = np.zeros(L)
    for ln in world["lines"]:
        if ln["idx"] in out:
            continue
        b = 1.0 / ln["x"]
        i, j = ln["from"], ln["to"]
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b
        A[ln["idx"], i], A[ln["idx"], j] = 1, -1
        bvec[ln["idx"]] = b
    Xr = np.linalg.inv(B[1:, 1:])
    X = np.zeros((N_BUS, N_BUS))
    X[1:, 1:] = Xr
    return (bvec[:, None] * A) @ X


def ptdf(world, out=()):
    """PTDF for a topology, cached on the network itself (one inversion per topology)."""
    key = tuple(sorted(out))
    cache = world.setdefault("_ptdf", {})
    if key not in cache:
        cache[key] = ptdf_for(world, key)
    return cache[key]


def dc_flow(world, inj, out=()):
    """MW flow on each line for bus injections `inj` (generation minus load, summing to zero)."""
    return ptdf(world, out) @ inj


def merit_order(world, t, solar=None, wind=None, load=None, battery=None):
    """Network-blind economic dispatch: all renewables used, batteries as given, thermal units cheapest first to balance; shortfall is shed.
    -> dict of outputs"""
    solar = world["solar_out"][:, t] if solar is None else solar
    wind = world["wind_out"][:, t] if wind is None else wind
    load = world["load"][t] if load is None else load
    bat = np.zeros(len(world["batteries"])) if battery is None else battery                  # + discharge, - charge
    need = load - solar.sum() - wind.sum() - bat.sum()
    th = np.zeros(len(world["thermal"]))
    curtail = 0.0
    if need < 0:                                                                             # surplus: curtail renewables pro rata
        curtail = -need
        scale = max(0.0, 1 - curtail / max(1e-9, solar.sum() + wind.sum()))
        solar, wind, need = solar * scale, wind * scale, 0.0
    for k in np.argsort([u["cost"] for u in world["thermal"]]):
        th[k] = min(world["thermal"][k]["cap"], need)
        need -= th[k]
    return {"thermal": th, "solar": solar, "wind": wind, "battery": bat, "shed": max(0.0, need), "load": load, "curtail": curtail}


def injections(world, disp, t):
    inj = np.zeros(N_BUS)
    for u, g in zip(world["thermal"], disp["thermal"]):
        inj[u["bus"]] += g
    for s, g in zip(world["solar"], disp["solar"]):
        inj[s["bus"]] += g
    for w_, g in zip(world["wind"], disp["wind"]):
        inj[w_["bus"]] += g
    for b, g in zip(world["batteries"], disp["battery"]):
        inj[b["bus"]] += g
    served = disp["load"] - disp["shed"]
    for bus, share in world["load_share"].items():
        inj[bus] -= served * share
    inj[0] -= inj.sum()                                                                     # numerical residue to the slack
    return inj


def contingency(world, inj, line_idx, base_out=()):
    """Flows with one more line out: loading (flow / limit) of every line above 90%, or islanded."""
    out = tuple(sorted(set(base_out) | {line_idx}))
    lab = connected(world, out)
    if len(set(lab)) > 1:
        return {"line": line_idx, "islanded": True, "buses_cut_off": sum(1 for x in lab if x != lab[0]), "loading": {}}
    f = dc_flow(world, inj, out)
    caps = np.array([ln.get("cap", 1e9) for ln in world["lines"]])
    loading = np.abs(f) / caps
    return {"line": line_idx, "islanded": False, "loading": {int(k): float(loading[k]) for k in np.where(loading > 0.9)[0] if k not in out}, "flows": f}
