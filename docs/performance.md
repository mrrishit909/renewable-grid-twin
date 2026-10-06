# Performance

Docker stack (one uvicorn process, one worker, Postgres 16) after the demo: 118 buses, 180 lines, 40 assets, 38,000 telemetry rows,
one dispatch plan of 1,872 setpoints. Reproduce with `make demo`, then `make load-test`.

## Reads: the grid state (DC power flow on the latest telemetry), closed loop, 10 s per level

| Concurrency | Requests | Throughput | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|---|
| 1 | 1530 | 152.9 req/s | 6.4 ms | 7.3 ms | 9.0 ms | 0.00% |
| 8 | 1751 | 174.7 req/s | 44.0 ms | 54.5 ms | 62.0 ms | 0.00% |
| 32 | 1419 | 139.8 req/s | 180.9 ms | 198.5 ms | 206.6 ms | 0.00% |

Target from the blueprint: p95 under 500 ms for reads: met. Every call reads the day's telemetry, runs a merit-order dispatch and a
DC power flow; the PTDF for each topology is computed once (a 117×117 inversion) and kept on the network object, so a line trip costs
one inversion and every later flow is a matrix-vector product.

## Compute (in-process, one core)

| Work | Time |
|---|---|
| Generate the network, sixty days of weather and output, line limits and the demo outage | 0.3 s |
| Fit the solar, wind and load models and score them on ten days | 1.3 s |
| PTDF for a topology | under 10 ms |
| N-1: 179 further outages, islanding check and flows each | about 1 s |
| Dispatch MILP to midnight, 48 steps, 288 binaries, line limits added in two rounds | 0.05–0.3 s |
| Twelve hourly re-plans | 0.3–1.5 s |
| Evaluation: three networks, every strategy | about 8 s |

Not measured: thousands of buses (the PTDF becomes the cost and wants sparse factorisation), or security-constrained dispatch, where
every N-1 case would enter the MILP.
