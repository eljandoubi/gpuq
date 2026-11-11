package sim_test

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"testing"

	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
	"github.com/eljandoubi/gpuq/internal/sim"
)

var spec = sim.Spec{Nodes: 8, GPUsPerNode: 8, NodesPerRack: 4}

func quotas(teams []sim.TeamCfg) map[string]int {
	q := map[string]int{}
	for _, t := range teams {
		q[t.Name] = t.Quota
	}
	return q
}

func workload(seed int64, jobs int, load float64) []*model.Job {
	return sim.Generate(sim.WorkloadCfg{Seed: seed, Jobs: jobs, Load: load, Spec: spec})
}

func clone(in []*model.Job) []*model.Job {
	out := make([]*model.Job, len(in))
	for i, j := range in {
		cp := *j
		out[i] = &cp
	}
	return out
}

func TestGenerateIsReproducible(t *testing.T) {
	a, b := workload(3, 500, 0.9), workload(3, 500, 0.9)
	for i := range a {
		if !reflect.DeepEqual(a[i], b[i]) {
			t.Fatalf("job %d differs between two runs with the same seed", i)
		}
	}
	if c := workload(4, 500, 0.9); reflect.DeepEqual(c[0], a[0]) && reflect.DeepEqual(c[1], a[1]) {
		t.Fatal("different seeds gave the same first jobs")
	}
}

func TestGenerateJobsAreWellFormed(t *testing.T) {
	teams := map[string]sim.TeamCfg{}
	for _, tc := range sim.DefaultTeams() {
		teams[tc.Name] = tc
	}
	jobs := workload(1, 5000, 0.9)
	prev := 0.0
	for i, j := range jobs {
		if j.ID != i+1 || j.Submit < prev {
			t.Fatalf("job %d: id=%d submit=%v after %v", i, j.ID, j.Submit, prev)
		}
		prev = j.Submit
		if _, ok := teams[j.Team]; !ok {
			t.Fatalf("job %d: unknown team %q", j.ID, j.Team)
		}
		if j.Nodes < 1 || j.Nodes > spec.Nodes || j.GPUsPerNode < 1 || j.GPUsPerNode > spec.GPUsPerNode {
			t.Fatalf("job %d: %d x %d does not fit the cluster", j.ID, j.Nodes, j.GPUsPerNode)
		}
		if j.GPUs() >= spec.Nodes*spec.GPUsPerNode {
			t.Fatalf("job %d takes the whole cluster", j.ID)
		}
		if j.Runtime < 60 || j.Runtime > 86400 {
			t.Fatalf("job %d: runtime %v outside 60 s to 24 h", j.ID, j.Runtime)
		}
		switch j.Priority {
		case 2:
			if j.Preemptible {
				t.Fatalf("job %d: priority 2 must not be preemptible", j.ID)
			}
		case 0:
			if !j.Preemptible {
				t.Fatalf("job %d: priority 0 must be preemptible", j.ID)
			}
		}
		if j.CheckpointEvery > 0 && !j.Preemptible {
			t.Fatalf("job %d checkpoints but cannot be preempted", j.ID)
		}
		if teams[j.Team].PreemptP == 0 && j.Priority == 1 && j.Preemptible {
			t.Fatalf("job %d: team %s never preempts a priority 1 job", j.ID, j.Team)
		}
	}
}

func TestGenerateOffersTheRequestedLoadAndShares(t *testing.T) {
	const load = 0.8
	jobs := workload(11, 40000, load)
	byTeam := map[string]float64{}
	total := 0.0
	for _, j := range jobs {
		g := j.Runtime * float64(j.GPUs())
		byTeam[j.Team] += g
		total += g
	}
	window := jobs[len(jobs)-1].Submit - jobs[0].Submit
	got := total / (float64(spec.Nodes*spec.GPUsPerNode) * window)
	if math.Abs(got-load) > 0.08 {
		t.Errorf("offered load %.3f, want %.2f within 0.08", got, load)
	}
	sum := 0.0
	for _, tc := range sim.DefaultTeams() {
		sum += tc.Share
	}
	for _, tc := range sim.DefaultTeams() {
		if share := byTeam[tc.Name] / total; math.Abs(share-tc.Share/sum) > 0.05 {
			t.Errorf("team %s offers %.3f of the GPU-time, want %.3f within 0.05", tc.Name, share, tc.Share/sum)
		}
	}
}

func TestRunIsDeterministic(t *testing.T) {
	w := workload(5, 600, 0.9)
	cfg := sched.Config{Kind: sched.FairShare, Quota: quotas(sim.DefaultTeams()), Preempt: true}
	a, b := sim.Run(spec, cfg, w), sim.Run(spec, cfg, w)
	a.DecisionP50us, a.DecisionP99us, b.DecisionP50us, b.DecisionP99us = 0, 0, 0, 0
	if fmt.Sprint(a) != fmt.Sprint(b) {
		t.Fatalf("two runs of the same workload differ:\n%+v\n%+v", a, b)
	}
}

func TestRunDoesNotModifyItsInput(t *testing.T) {
	w := workload(5, 200, 0.9)
	before := clone(w)
	sim.Run(spec, sched.Config{Kind: sched.Backfill}, w)
	for i := range w {
		if !reflect.DeepEqual(w[i], before[i]) {
			t.Fatalf("job %d was modified by Run", i)
		}
	}
}

// The properties below hold for every policy on random workloads that overload the cluster.
func TestInvariantsOnRandomWorkloads(t *testing.T) {
	q := quotas(sim.DefaultTeams())
	configs := []struct {
		name string
		cfg  sched.Config
	}{
		{"fifo", sched.Config{Kind: sched.FIFO}},
		{"backfill", sched.Config{Kind: sched.Backfill}},
		{"fairshare", sched.Config{Kind: sched.FairShare, Quota: q, Preempt: true}},
		{"fairshare-nopreempt", sched.Config{Kind: sched.FairShare, Quota: q}},
		{"fairshare-twitchy", sched.Config{Kind: sched.FairShare, Quota: q, Preempt: true, MinRun: 1, MaxWait: 900, RestartCost: 5}},
	}
	total := spec.Nodes * spec.GPUsPerNode
	for _, c := range configs {
		for seed := int64(1); seed <= 6; seed++ {
			t.Run(fmt.Sprintf("%s/seed%d", c.name, seed), func(t *testing.T) {
				w := workload(seed, 500, 1.2)
				team := map[int]string{}
				prio := map[int]int{}
				for _, j := range w {
					team[j.ID], prio[j.ID] = j.Team, j.Priority
				}
				minRun := c.cfg.MinRun
				if minRun <= 0 {
					minRun = 300
				}

				usage := map[string]int{}
				startedAt := map[int]float64{}
				inUse, lastStart, last, busy := 0, -1, 0.0, 0.0
				observe := func(ev sched.Event) {
					j := ev.Job
					busy += float64(inUse) * (ev.Time - last)
					last = ev.Time
					switch ev.Kind {
					case sched.Start:
						if c.cfg.Kind == sched.FIFO && j.ID < lastStart {
							t.Fatalf("fifo started job %d after job %d", j.ID, lastStart)
						}
						lastStart = j.ID
						startedAt[j.ID] = ev.Time
						usage[j.Team] += j.GPUs()
						inUse += j.GPUs()
						if inUse > total {
							t.Fatalf("t=%v: %d GPUs in use on a cluster of %d", ev.Time, inUse, total)
						}
					case sched.Finish:
						usage[j.Team] -= j.GPUs()
						inUse -= j.GPUs()
					case sched.Preempt:
						if !j.Preemptible {
							t.Fatalf("t=%v: job %d is not preemptible", ev.Time, j.ID)
						}
						if ev.Time-startedAt[j.ID] < minRun {
							t.Fatalf("t=%v: job %d preempted after %v s, MinRun is %v", ev.Time, j.ID, ev.Time-startedAt[j.ID], minRun)
						}
						if by := team[ev.For]; by != j.Team {
							// The guarantee: another team only takes GPUs from a team above its quota.
							if usage[j.Team] <= q[j.Team] {
								t.Fatalf("t=%v: team %s runs %d GPUs, within its quota of %d, and lost job %d to team %s",
									ev.Time, j.Team, usage[j.Team], q[j.Team], j.ID, by)
							}
						} else if prio[ev.For] <= j.Priority {
							t.Fatalf("t=%v: job %d (priority %d) preempted job %d (priority %d) of the same team",
								ev.Time, ev.For, prio[ev.For], j.ID, j.Priority)
						}
						usage[j.Team] -= j.GPUs()
						inUse -= j.GPUs()
					}
				}

				res, js, cl := sim.Replay(spec, c.cfg, w, observe)

				useful, wasted := 0.0, 0.0
				for _, j := range js {
					if j.State != model.Done {
						t.Fatalf("job %d ended %v, want done", j.ID, j.State)
					}
					if j.FirstStart < j.Submit-1e-9 || j.End < j.FirstStart+j.Runtime-1e-6 {
						t.Fatalf("job %d: submit=%v first start=%v end=%v runtime=%v", j.ID, j.Submit, j.FirstStart, j.End, j.Runtime)
					}
					useful += j.Runtime * float64(j.GPUs())
					wasted += j.WastedGPUs
				}
				if cl.Free() != cl.Total() || cl.Check() != nil {
					t.Fatalf("cluster not empty at the end: free=%d of %d, check=%v", cl.Free(), cl.Total(), cl.Check())
				}
				if inUse != 0 || busy < 1 {
					t.Fatalf("inUse=%d busy=%v at the end", inUse, busy)
				}
				// Every GPU-second in use is either finished work or work that a preemption wasted.
				if want := useful + wasted; math.Abs(busy-want) > 1e-6*want {
					t.Fatalf("busy GPU-seconds %.3f, finished work %.3f + wasted %.3f = %.3f", busy, useful, wasted, want)
				}
				if res.Preemptions != countPreemptions(js) {
					t.Fatalf("result says %d preemptions, the jobs say %d", res.Preemptions, countPreemptions(js))
				}
				if c.cfg.Kind != sched.FairShare && res.ReservationViolations != 0 {
					t.Fatalf("%d head jobs started after the time promised to them", res.ReservationViolations)
				}
				if !c.cfg.Preempt && res.Preemptions != 0 {
					t.Fatalf("%d preemptions with preemption off", res.Preemptions)
				}
				if res.Utilization < 0 || res.Utilization > 1+1e-9 || res.UsefulUtilization > res.Utilization+1e-9 {
					t.Fatalf("utilization %v, useful %v", res.Utilization, res.UsefulUtilization)
				}
			})
		}
	}
}

func countPreemptions(js []*model.Job) int {
	n := 0
	for _, j := range js {
		n += j.Preemptions
	}
	return n
}

// Team b asks for both nodes at a lower priority than team a, which always has a one-node job
// ready. Without aging the gang job waits for the whole stream; with aging it waits about MaxWait
// plus the time to drain the two nodes.
func TestAgingStopsAGangJobFromStarving(t *testing.T) {
	two := sim.Spec{Nodes: 2, GPUsPerNode: 8, NodesPerRack: 2}
	jobs := []*model.Job{{ID: 1, Team: "b", Priority: 0, Nodes: 2, GPUsPerNode: 8, Runtime: 100, Submit: 10}}
	for i := 0; i < 400; i++ {
		jobs = append(jobs, &model.Job{ID: i + 2, Team: "a", Priority: 1, Nodes: 1, GPUsPerNode: 8, Runtime: 100, Submit: float64(50 * i)})
	}
	start := func(maxWait float64) float64 {
		cfg := sched.Config{Kind: sched.FairShare, Quota: map[string]int{"a": 16}, MaxWait: maxWait}
		_, js, _ := sim.RunDetailed(two, cfg, jobs)
		for _, j := range js {
			if j.ID == 1 {
				return j.FirstStart
			}
		}
		t.Fatal("job 1 vanished")
		return 0
	}
	if got := start(300); got > 600 {
		t.Errorf("with MaxWait=300 s the gang job started at %v s, want before 600 s", got)
	}
	if got := start(-1); got < 15000 {
		t.Errorf("with aging off the gang job started at %v s: the scenario no longer starves it", got)
	}
}

// The reason the policy exists: a team that stays within its quota keeps short waits while another
// team floods the cluster, which plain backfill cannot promise.
func TestFairShareProtectsTheTeamWithinItsQuota(t *testing.T) {
	teams := sim.DefaultTeams()
	back := sched.Config{Kind: sched.Backfill}
	fair := sched.Config{Kind: sched.FairShare, Quota: quotas(teams), Preempt: true}
	var pBack, pFair []float64
	for seed := int64(1); seed <= 8; seed++ {
		w := workload(seed, 3000, 0.85)
		pBack = append(pBack, sim.Run(spec, back, w).TeamWaitP95["eval"])
		pFair = append(pFair, sim.Run(spec, fair, w).TeamWaitP95["eval"])
	}
	b, f := mean(pBack), mean(pFair)
	if f > 0.6*b {
		t.Errorf("eval p95 wait: fairshare %.0f s, backfill %.0f s; want at most 60%% of backfill", f, b)
	}
}

func TestResultMetricsOnATinyWorkload(t *testing.T) {
	two := sim.Spec{Nodes: 1, GPUsPerNode: 4, NodesPerRack: 1}
	jobs := []*model.Job{
		{ID: 1, Team: "a", Nodes: 1, GPUsPerNode: 4, Runtime: 100, Submit: 0},
		{ID: 2, Team: "a", Nodes: 1, GPUsPerNode: 4, Runtime: 100, Submit: 0},
		{ID: 3, Team: "a", Nodes: 1, GPUsPerNode: 4, Runtime: 100, Submit: 50},
	}
	r := sim.Run(two, sched.Config{Kind: sched.FIFO}, jobs)
	// Jobs start at 0, 100 and 200; they wait 0, 100 and 150 seconds.
	if r.Makespan != 300 {
		t.Errorf("makespan=%v, want 300", r.Makespan)
	}
	if r.WaitMean != 250.0/3 || r.WaitP50 != 100 || r.WaitP95 != 150 {
		t.Errorf("waits mean=%v p50=%v p95=%v, want 83.3, 100 and 150", r.WaitMean, r.WaitP50, r.WaitP95)
	}
	// Bounded slowdown: (wait+run)/max(run, 60) = 1, 2 and 2.5.
	if r.SlowdownP95 != 2.5 {
		t.Errorf("slowdown p95=%v, want 2.5", r.SlowdownP95)
	}
	// While jobs arrive (0 to 50 s) the single node is busy: utilization 1.
	if r.Utilization != 1 || r.UsefulUtilization != 1 {
		t.Errorf("utilization=%v useful=%v, want 1 and 1", r.Utilization, r.UsefulUtilization)
	}
	if r.TeamWaitP95["a"] != 150 || r.WaitP95Small != 150 || r.WaitP95Large != 0 {
		t.Errorf("team p95=%v small=%v large=%v", r.TeamWaitP95, r.WaitP95Small, r.WaitP95Large)
	}
}

func TestEmptyWorkload(t *testing.T) {
	r := sim.Run(spec, sched.Config{Kind: sched.Backfill}, nil)
	if r.Jobs != 0 || r.Makespan != 0 || r.Utilization != 0 {
		t.Fatalf("empty workload gave %+v", r)
	}
}

func mean(x []float64) float64 {
	sort.Float64s(x)
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s / float64(len(x))
}

func BenchmarkRun3000Jobs(b *testing.B) {
	w := workload(1, 3000, 0.85)
	cfg := sched.Config{Kind: sched.FairShare, Quota: quotas(sim.DefaultTeams()), Preempt: true}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sim.Run(spec, cfg, w)
	}
}
