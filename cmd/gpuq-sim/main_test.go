package main

import (
	"io"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/eljandoubi/gpuq/internal/sched"
	"github.com/eljandoubi/gpuq/internal/sim"
)

func TestMeanStd(t *testing.T) {
	m, sd := meanStd([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	if m != 5 || math.Abs(sd-2) > 1e-12 {
		t.Fatalf("mean=%v std=%v, want 5 and 2", m, sd)
	}
	if m, sd := meanStd([]float64{3}); m != 3 || sd != 0 {
		t.Fatalf("one value: mean=%v std=%v", m, sd)
	}
}

func TestPolicies(t *testing.T) {
	teams := sim.DefaultTeams()
	cfgs, labels := policies(teams, 6*time.Hour, 5*time.Minute)
	if len(cfgs) != 3 || len(labels) != 3 || cfgs[0].Kind != sched.FIFO || cfgs[1].Kind != sched.Backfill || cfgs[2].Kind != sched.FairShare {
		t.Fatalf("policies = %v %v", cfgs, labels)
	}
	fair := cfgs[2]
	if fair.MaxWait != 6*3600 || fair.MinRun != 300 || !fair.Preempt {
		t.Fatalf("fairshare config = %+v", fair)
	}
	for _, tc := range teams {
		if fair.Quota[tc.Name] != tc.Quota {
			t.Errorf("quota of %s is %d, want %d", tc.Name, fair.Quota[tc.Name], tc.Quota)
		}
	}
	if off, _ := policies(teams, 0, time.Minute); off[2].MaxWait >= 0 {
		t.Fatalf("-max-wait 0 must turn aging off (negative MaxWait), got %v", off[2].MaxWait)
	}
}

func TestReportPrintsEveryTable(t *testing.T) {
	su := setup{spec: sim.Spec{Nodes: 8, GPUsPerNode: 8, NodesPerRack: 4}, teams: sim.DefaultTeams(), jobs: 150, seeds: 2}
	su.configs, su.labels = policies(su.teams, 6*time.Hour, 5*time.Minute)
	out := capture(t, func() {
		printReport(su.runAll(0.8), su.teams)
		printSweep([]float64{0.5, 0.8}, su.labels, [][][]sim.Result{su.runAll(0.5), su.runAll(0.8)})
	})
	for _, want := range []string{
		"**Throughput**", "**Waiting**", "**p95 wait per team**", "| fifo |", "| backfill |", "| fairshare+preempt |",
		"eval (8 GPUs, 15%)", "**p95 wait (minutes)**", "**useful utilization while jobs arrive (work lost to preemption excluded)**", "| 0.50 |", "| 0.80 |",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q", want)
		}
	}
	if strings.Contains(out, "NaN") {
		t.Errorf("output contains NaN:\n%s", out)
	}
}

func capture(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stdout
	os.Stdout = w
	done := make(chan string)
	go func() {
		b, _ := io.ReadAll(r)
		done <- string(b)
	}()
	f()
	w.Close()
	os.Stdout = old
	return <-done
}
