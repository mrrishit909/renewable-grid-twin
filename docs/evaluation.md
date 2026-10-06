# Evaluation

Everything here is measured on synthetic networks generated with seeds the demo does not use (`(101, 102, 103)`), each with the same kind of day:
strong wind in the north the forecast saw, a cloud bank over the solar region it did not, and the line whose loss most overloads the wind
export corridor tripping at 12:00. Reproduce with `python -m rg.evaluate` (about a minute). The twin that scores the plans is the
generator itself, with DC power flow as the physics; on a real network the scoring would be an approximation and the gaps wider.

## 1. Forecasts, on the last ten days of history

Mean error in MW. Persistence is the same time yesterday; the physical formulas are clear-sky output for solar and the power curve on
forecast wind speed; load's baseline is the same time last week.

| network (seed) | solar | persistence | clear sky | wind | persistence | power curve | load | last week | load inside 80% band |
|---|---|---|---|---|---|---|---|---|---|
| 101 | 19.5 | 60.0 | 122.8 | 70.5 | 373.8 | 94.5 | 24.4 | 41.0 | 74% |
| 102 | 19.1 | 59.8 | 134.7 | 78.2 | 356.6 | 89.7 | 26.5 | 38.2 | 72% |
| 103 | 14.8 | 74.3 | 132.7 | 57.5 | 319.0 | 75.8 | 25.0 | 39.6 | 75% |

The models beat persistence by a wide margin and the clear-sky formula easily. Wind beats the power curve on forecast speed by 12–25%
here (and only barely on the demo network, 46.9 against 49.5 MW): the generator scales wind speed per site, which the plain curve does
not know and the model learns from the history.

## 2. The intraday update at 12:00

After an hour of cloud, the last hour's actual solar as a share of the day-ahead forecast, and the rest of the day's solar energy by
the day-ahead forecast, the update and the truth.

| network (seed) | last hour, actual / forecast | day-ahead (MWh) | intraday update (MWh) | truth (MWh) |
|---|---|---|---|---|
| 101 | 38% | 5389 | 3754 | 2868 |
| 102 | 36% | 5726 | 3937 | 2982 |
| 103 | 38% | 4966 | 3457 | 2654 |

The update closes about two thirds of the gap between the day-ahead forecast and the truth. It assumes the cloud clears over a couple
of hours; this one stays four more, and a forecaster who could see the cloud bank on a satellite image would do better than any ratio.

## 3. Running the afternoon, scored on the true weather and true flows

| network (seed) | strategy | cost ($) | renewables used | curtailed (MWh) | overload minutes | worst loading |
|---|---|---|---|---|---|---|
| 101 | network-blind, no battery | 266,231 | 100.0% | 0 | 525 | 274% |
| 101 | network-blind, battery on price | 258,587 | 100.0% | 0 | 570 | 290% |
| 101 | 12:00 plan, as made | 370,923 | 71.0% | 3,604 | 165 | 101% |
| 101 | re-planned hourly, no batteries | 378,411 | 67.6% | 4,026 | 90 | 101% |
| 101 | re-planned hourly, with batteries | 375,682 | 69.4% | 3,804 | 90 | 101% |
| 101 | perfect knowledge | 356,740 | 74.6% | 3,155 | 0 | 100% |
| 102 | network-blind, no battery | 319,938 | 100.0% | 0 | 480 | 339% |
| 102 | network-blind, battery on price | 313,312 | 100.0% | 0 | 495 | 449% |
| 102 | 12:00 plan, as made | 365,890 | 86.0% | 1,620 | 135 | 103% |
| 102 | re-planned hourly, no batteries | 375,119 | 82.5% | 2,020 | 120 | 110% |
| 102 | re-planned hourly, with batteries | 371,154 | 84.3% | 1,810 | 60 | 101% |
| 102 | perfect knowledge | 352,900 | 89.8% | 1,174 | 0 | 100% |
| 103 | network-blind, no battery | 325,855 | 100.0% | 0 | 645 | 337% |
| 103 | network-blind, battery on price | 316,362 | 100.0% | 0 | 645 | 337% |
| 103 | 12:00 plan, as made | 400,768 | 82.4% | 2,194 | 15 | 101% |
| 103 | re-planned hourly, no batteries | 405,032 | 81.1% | 2,361 | 75 | 102% |
| 103 | re-planned hourly, with batteries | 405,032 | 81.1% | 2,361 | 75 | 102% |
| 103 | perfect knowledge | 378,833 | 87.0% | 1,620 | 0 | 100% |

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
and discharge was 0.0e+00 MW: the binary per battery per step does its job.

## 4. N-1

At the windy afternoon step: from the intact network, how many single outages island buses and how many add an overload; then from the
network with the demo line out, how many lines are already over and how many further outages add another.

| network (seed) | demo outage line | islanding (intact) | adding an overload (intact) | already over (line out) | adding an overload (line out) |
|---|---|---|---|---|---|
| 101 | 142 | 9 | 60 | 3 | 58 |
| 102 | 14 | 11 | 59 | 10 | 57 |
| 103 | 84 | 18 | 38 | 5 | 41 |

## 5. What is not measured

- AC power flow, voltages, reactive power, stability: DC flow only.
- Security-constrained dispatch (every N-1 case enforced inside the dispatch); the scan ranks outages, the dispatch enforces the current topology.
- Real weather, real networks, real markets.
