package sim_test

import (
	"testing"

	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
	"github.com/eljandoubi/gpuq/internal/sim"
)

var spec = sim.Spec{Nodes: 8, GPUsPerNode: 8, NodesPerRack: 4}

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
