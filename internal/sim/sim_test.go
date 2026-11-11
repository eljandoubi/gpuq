package sim_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
	"github.com/eljandoubi/gpuq/internal/sim"
)

var spec = sim.Spec{Nodes: 8, GPUsPerNode: 8, NodesPerRack: 4}

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
