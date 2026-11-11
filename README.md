# gpuq

A gang scheduler for GPU clusters, with a discrete-event simulator to compare policies and a small HTTP
control plane with Prometheus metrics. Go, standard library only.

It answers one question that comes up on every shared training cluster: **how do you keep a team's
guaranteed share and short jobs responsive without starving big jobs or wasting GPU time on
preemptions?** The policies, the invariants they must keep and the measured trade-offs are below.

> **Scope.** This is a simulation. No job runs on a GPU: a job is a gang of `nodes x gpus_per_node`
> with a declared runtime, and the workload is synthetic (it is generated, not recorded). The results
> describe how the policies behave on that workload, not on any production cluster. See
> [Limits](#limits).

## What is in it

| | |
|---|---|
| `internal/model` | Cluster, nodes, racks, jobs. Best-fit, rack-aware gang placement; all-or-nothing allocation. |
| `internal/sched` | Three policies: `fifo`, `backfill` (EASY: a later job may start only if it cannot delay the head), `fairshare` (quotas, borrowing, reclaim by preemption, aging). |
| `internal/sim` | Event-driven simulator, seeded workload generator, metrics. |
| `internal/api`, `cmd/gpuqd` | `POST /v1/jobs`, `GET /v1/jobs[/{id}]`, `GET /v1/cluster`, `GET /metrics` (Prometheus), `GET /healthz`. |
| `cmd/gpuq-sim` | Policy comparison over N seeds, load sweep, JSON output. |
| `deploy/`, `Dockerfile` | gpuq's quotas as Kueue queues, example gang jobs, a hardened Deployment for `gpuqd`. |

## The fair-share policy

Teams have a **quota** (GPUs they are guaranteed). A team may run above its quota when GPUs are idle
(borrowing); borrowed GPUs are taken back by preemption when the owner needs them. The queue is
served in this order: jobs that have waited `MaxWait` (multi-node gangs only, oldest first), then
jobs that fit their team's quota, then priority, then the team that has used the least of its share
(decayed usage / quota), then arrival.

A job may take GPUs from a running job only if all of these hold (`internal/sched/preempt.go`):

1. the victim is preemptible and has run for at least `MinRun` (default 5 min);
2. inside one team, the victim has a lower priority;
3. across teams, the victim's team runs **above its quota**. A team that stays within its quota is
   never touched by another team, whatever the priorities;
4. such a borrower yields to a job that fits its own quota (reclaim) or has a higher priority.

Only the head of the queue and the queued jobs that fit their team's quota may start a preemption.
Victims are the lowest priority first, then from the team furthest above its quota, then the job that
loses the least work (progress since its last checkpoint plus a restart cost). The victim set is then
reduced to the jobs the new job actually needs.

Rule 3 is what makes the policy stable. A first version let priority and reclaim apply to the same
jobs: a high-priority job above its quota preempted a low-priority job within its quota, which then
reclaimed its share by preempting the first one, forever inside a single scheduling pass (found when
the simulator never returned). Rule 1 also bounds a pass: a job started or restarted in a pass has
run for zero seconds, so it cannot be preempted in that pass.

**Aging.** Ordering by quota alone starves big jobs: a 4-node job (32 GPUs) does not fit the research
team's quota of 24, so it queues behind every smaller job that does, and a reservation for it does
not survive the arrival of a better-ranked job. After `MaxWait` (default 6 h) a multi-node job moves
to the front and the usual backfill reservation keeps the cluster draining for it.

## Results

Synthetic workload: three teams on an 8-node x 8-GPU cluster (4 nodes per rack), 3,000 jobs per run,
offered load 0.85 of capacity, 10 seeds, same workload replayed under each policy. Poisson arrivals,
log-normal runtimes, gangs of 1 to 32 GPUs (no job takes the whole cluster).

| team | quota | share of the offered GPU-time | what it runs |
|---|---:|---:|---|
| research | 24 | 50% | all gang sizes, median 30 min, asks for more than its quota |
| training | 32 | 35% | 8 to 32 GPUs, median 2 h, leaves part of its quota idle |
| eval | 8 | 15% | 1 to 8 GPUs, median 10 min, must not wait behind the others |

`go run ./cmd/gpuq-sim` (mean of 10 seeds; the program also prints the standard deviation):

| policy | wait p50 (min) | wait p95 (min) | p95, jobs of up to 4 GPUs | p95, jobs of 32+ GPUs | p95 eval team | p95 research | p95 training |
|---|---:|---:|---:|---:|---:|---:|---:|
| fifo | 535 | 1168 | 1167 | 1172 | 1166 | 1172 | 1149 |
| backfill | 25 | 348 | 275 | 625 | 266 | 430 | 583 |
| fairshare + preemption | **1** | 354 | **84** | 998 | **107** | 659 | 684 |

| policy | useful utilization | preemptions per 3,000 jobs | wasted GPU-hours | decision p99 |
|---|---:|---:|---:|---:|
| fifo | 76.7% | 0 | 0 | 14 us |
| backfill | 80.6% | 0 | 0 | 24 us |
| fairshare + preemption | 77.3% | 351 | 1,314 | 74 us |

*Useful utilization* is the share of GPU-time, while jobs are still arriving, that ended up in
finished work: time lost to preemption is not counted.

What this shows:

- **Isolation works.** The eval team's p95 wait drops from 266 to 107 min, and jobs of up to 4 GPUs
  wait 3.3x less at p95. Paired by seed, fairshare is better than backfill on both in 10 of 10 seeds
  (eval p95 ratio 0.14 to 0.68).
- **It is not free.** Useful utilization is 3.3 points lower in 10 of 10 seeds (preempted jobs redo
  work and pay a restart), and jobs of 32+ GPUs wait longer at p95 (998 vs 625 min, worse in 9 of 10
  seeds). The research team, which asks for more than its quota, waits longer too (659 vs 430 min):
  that is the policy doing its job.
- **FIFO is not viable** for gang jobs at this load: head-of-line blocking, p95 wait near 20 hours.

Load sweep (`go run ./cmd/gpuq-sim -sweep`, p95 wait in minutes / useful utilization):

| offered load | fifo | backfill | fairshare + preemption |
|---:|---:|---:|---:|
| 0.50 | 246 / 48.4% | 58 / 48.4% | **8** / 48.0% |
| 0.70 | 633 / 66.5% | 174 / 67.2% | **84** / 66.0% |
| 0.80 | 925 / 73.8% | 274 / 76.2% | **206** / 74.1% |
| 0.90 | 1557 / 78.3% | **422** / 84.4% | 501 / 79.4% |
| 0.95 | 1994 / 78.8% | **511** / 87.9% | 602 / 81.9% |

Fairshare wins up to about 0.8 offered load. Above that, preemption overhead outweighs what it
saves on the overall p95, and backfill is the better policy for the cluster as a whole. The eval
team still benefits there (p95 of 127 vs 331 min at 0.90, 154 vs 425 min at 0.95): what changes is
the price, not the isolation.

Sensitivity of the fair-share policy (load 0.85, 10 seeds):

| policy | p95 eval (min) | p95, jobs of 32+ GPUs (min) | useful util. | preemptions | wasted GPU-h |
|---|---:|---:|---:|---:|---:|
| backfill | 266 | 625 | 80.6% | 0 | 0 |
| fairshare, no aging | 41 | **7,993** | 75.3% | 349 | 1,412 |
| fairshare, `MaxWait` 2 h | 108 | 793 | 77.5% | 338 | 1,216 |
| fairshare, `MaxWait` 6 h (default) | 107 | 998 | 77.3% | 351 | 1,314 |
| fairshare, `MaxWait` 24 h | 56 | 1,766 | 76.5% | 334 | 1,261 |
| fairshare, `MinRun` 1 min | 72 | 926 | 77.6% | 374 | 1,078 |
| fairshare, `MinRun` 30 min | 164 | 971 | 76.2% | 259 | 1,706 |

Without aging the isolation is best (eval p95 41 min) and big jobs wait 5.5 days at p95. `MaxWait` is
the dial between the two. A long `MinRun` reduces the number of preemptions but each one loses more
work (a job that has run for 30 minutes without a checkpoint loses 30 minutes), so it wastes more.
The default stays at 5 minutes because a real restart (checkpoint load, process start) can cost
more than the 120 s the simulator charges; a shorter window looks better here partly because
restarts are cheap in the model.

## Invariants and tests

`go test -race ./...` (coverage: model 100%, sim 98%, api 95%, sched 93%). The simulator tests run
every policy on random workloads that overload the cluster (load 1.2) and check, event by event:

- a preemption victim is preemptible and had run at least `MinRun`;
- **a team within its quota never loses a job to another team**; inside a team only a higher
  priority preempts;
- the GPUs in use never exceed the cluster, and the cluster is empty at the end;
- every GPU-second in use is either finished work or work a preemption wasted (accounting identity);
- FIFO starts jobs in submission order; backfill and FIFO never start a head job after the time that
  was promised to it (exact runtimes);
- every job finishes (a scheduling pass always returns);
- the same seed gives the same result.

Scenario tests pin each rule of the policy (reclaim, `MinRun`, checkpoint accounting, cheapest
victim, minimal victim set, aging of a starving gang job, ...). To check that the tests catch real
mistakes, `python3 scripts/mutate.py` applies 17 hand-written faults (a rule removed, an inequality
flipped, a condition dropped) to private copies of the code: all 17 are caught. Writing it exposed
two missing tests (a victim that the new job does not need being preempted; a queued job above its
quota starting a preemption).

## Run it

```sh
go test -race ./...                  # or: make race
go run ./cmd/gpuq-sim                # the comparison above
go run ./cmd/gpuq-sim -sweep         # p95 wait and utilization against offered load
go run ./cmd/gpuq-sim -json          # per-seed results
go test -run xxx -bench . ./internal/sched ./internal/sim

go run ./cmd/gpuqd -policy fairshare -quota research=24,training=32,eval=8 -speed 60
curl -s localhost:8080/v1/jobs -d '{"team":"eval","nodes":1,"gpus_per_node":4,"runtime_s":600}'
curl -s localhost:8080/metrics
```

`gpuqd` runs jobs on a clock `-speed` times faster than the wall clock. Submitting a job that could
never fit answers 422; unknown fields, trailing data and bodies over 1 MiB answer 400.

Benchmarks (Apple M4, a few runs on a machine in use): one scheduling pass over a full cluster with
1,000 queued jobs takes 6 to 10 us for backfill and about 0.2 ms for fairshare (it sorts the queue
and looks for reclaim candidates); a whole 3,000-job simulation takes 70 to 76 ms.

## Deploy

`Dockerfile` (static binary in a distroless, non-root image) and `deploy/kubernetes/`: the quotas of
the simulator as [Kueue](https://kueue.sigs.k8s.io/) queues, two example gang jobs, and a hardened
Deployment for `gpuqd` with a Datadog OpenMetrics annotation. The 19 resources pass `kubeconform`
in strict mode (`make k8s-validate`). None was applied to a cluster and the image was not built (no
container runtime was available), which `deploy/README.md` says again. It also maps each rule of the
fair-share policy to the closest Kueue field and lists where Kueue has no equivalent (aging, a
minimum run time before preemption).

## Limits

- **Synthetic workload, exact runtimes.** Real jobs have noisy time estimates, which weakens the
  backfill guarantee; failures, node drains and heterogeneous GPUs are not modeled.
- **Topology** is two-level (rack, node). Network contention, NVLink domains and fragmentation by
  memory are not modeled.
- **Preemption cost** is a checkpoint interval plus a fixed restart cost per GPU. Real costs depend
  on the model size and the storage.
- **No job takes the whole cluster.** Such jobs need a drain window, not a queue; with them the
  fair-share policy leaves GPUs idle while it drains (measured during development).
- **One process, in memory.** `gpuqd` keeps state in memory, has no authentication and no
  persistence; it demonstrates the control-plane shape, it is not a Kubernetes or Slurm replacement.
- Fair-share usage is decayed per team (half-life 6 h); there is no per-user fairness.

## License

Apache-2.0, see [LICENSE](LICENSE).
