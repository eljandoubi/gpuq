package sched_test

import (
	"testing"

	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
)

func cluster(nodes, per int) *model.Cluster { return model.NewCluster(nodes, per, nodes) }

func job(id int, team string, nodes, per int, runtime float64) *model.Job {
	return &model.Job{ID: id, Team: team, Priority: 1, Nodes: nodes, GPUsPerNode: per, Runtime: runtime}
}

// submit queues the jobs at time now and lets the scheduler act.
func submit(t *testing.T, s *sched.Scheduler, now float64, js ...*model.Job) {
	t.Helper()
	for _, j := range js {
		j.Submit = now
		if err := s.Submit(j); err != nil {
			t.Fatal(err)
		}
	}
	s.Tick(now)
}

func complete(t *testing.T, s *sched.Scheduler, id int, now float64) {
	t.Helper()
	if err := s.Complete(id, now); err != nil {
		t.Fatal(err)
	}
	s.Tick(now)
}

func preemptions(s *sched.Scheduler) []sched.Event {
	var out []sched.Event
	for _, ev := range s.Drain() {
		if ev.Kind == sched.Preempt {
			out = append(out, ev)
		}
	}
	return out
}

func fairShare(quota map[string]int) sched.Config {
	return sched.Config{Kind: sched.FairShare, Quota: quota, Preempt: true}
}

func TestSubmitRejectsAJobThatCanNeverFit(t *testing.T) {
	s := sched.New(sched.Config{}, cluster(2, 4))
	for _, j := range []*model.Job{job(1, "a", 3, 4, 10), job(2, "a", 1, 5, 10)} {
		if err := s.Submit(j); err == nil {
			t.Errorf("job %d asks for more than the cluster has and must be refused", j.ID)
		}
	}
	if s.Pending() != 0 {
		t.Fatalf("pending=%d, want 0", s.Pending())
	}
}

func TestCompleteRejectsAnUnknownJob(t *testing.T) {
	s := sched.New(sched.Config{}, cluster(1, 4))
	if err := s.Complete(42, 1); err == nil {
		t.Fatal("completing a job that is not running must fail")
	}
}

func TestFIFOBlocksBehindTheHead(t *testing.T) {
	s := sched.New(sched.Config{Kind: sched.FIFO}, cluster(2, 4))
	submit(t, s, 0, job(1, "a", 1, 4, 100), job(2, "a", 2, 4, 50), job(3, "a", 1, 4, 10))
	if s.Running() != 1 || s.Pending() != 2 {
		t.Fatalf("running=%d pending=%d: job 3 must wait behind the blocked job 2", s.Running(), s.Pending())
	}
	complete(t, s, 1, 100)
	if s.Running() != 1 || s.Pending() != 1 {
		t.Fatalf("running=%d pending=%d after job 1: job 2 runs and job 3 still waits", s.Running(), s.Pending())
	}
}

func TestBackfillStartsAShortJobWithoutDelayingTheHead(t *testing.T) {
	s := sched.New(sched.Config{Kind: sched.Backfill}, cluster(2, 4))
	j2 := job(2, "a", 2, 4, 50)
	submit(t, s, 0, job(1, "a", 1, 4, 100), j2, job(3, "a", 1, 4, 10))
	if s.Running() != 2 {
		t.Fatalf("running=%d, want 2: job 3 ends before job 2 could start, so it may run now", s.Running())
	}
	if at, ok := s.Reserved(2); !ok || at != 100 {
		t.Fatalf("job 2 was promised %v (%v), want 100", at, ok)
	}
	complete(t, s, 3, 10)
	complete(t, s, 1, 100)
	if j2.State != model.Running || j2.Start != 100 {
		t.Fatalf("job 2: state=%v start=%v, want it started at its reserved time 100", j2.State, j2.Start)
	}
}

func TestBackfillRefusesAJobThatWouldDelayTheHead(t *testing.T) {
	s := sched.New(sched.Config{Kind: sched.Backfill}, cluster(2, 4))
	j3 := job(3, "a", 1, 4, 500)
	submit(t, s, 0, job(1, "a", 1, 4, 100), job(2, "a", 2, 4, 50), j3)
	if j3.State != model.Pending {
		t.Fatalf("job 3 would hold a node until 500, past the head's reservation at 100: state=%v", j3.State)
	}
}

func TestFairShareServesTheJobWithinQuotaFirst(t *testing.T) {
	s := sched.New(sched.Config{Kind: sched.FairShare, Quota: map[string]int{"a": 4}}, cluster(1, 4))
	j1, j2, j3 := job(1, "a", 1, 4, 100), job(2, "b", 1, 4, 100), job(3, "a", 1, 4, 100)
	submit(t, s, 0, j1)
	submit(t, s, 1, j2) // team b has no quota
	submit(t, s, 2, j3)
	complete(t, s, 1, 100)
	if j3.State != model.Running || j2.State != model.Pending {
		t.Fatalf("j3=%v j2=%v: team a is back within its quota and goes before team b, which arrived first",
			j3.State, j2.State)
	}
}

func TestFairShareFavoursTheTeamThatUsedLess(t *testing.T) {
	s := sched.New(sched.Config{Kind: sched.FairShare, Quota: map[string]int{"a": 4, "b": 4}}, cluster(1, 4))
	j1, j2, j3 := job(1, "a", 1, 4, 1000), job(2, "a", 1, 4, 100), job(3, "b", 1, 4, 100)
	submit(t, s, 0, j1)
	submit(t, s, 10, j2)
	submit(t, s, 20, j3)
	complete(t, s, 1, 1000)
	if j3.State != model.Running || j2.State != model.Pending {
		t.Fatalf("j3=%v j2=%v: both teams are within quota, team a ran for 1000 s and team b not at all",
			j3.State, j2.State)
	}
}

// Team "r" runs on both nodes while its quota is half of the cluster; team "e" then asks for its own
// half back.
func TestFairShareReclaimsBorrowedGPUs(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"r": 4, "e": 4}), cluster(2, 4))
	r1, r2 := job(1, "r", 1, 4, 5000), job(2, "r", 1, 4, 5000)
	r1.Preemptible, r2.Preemptible = true, true
	submit(t, s, 0, r1, r2)
	s.Drain()

	e1 := job(3, "e", 1, 4, 100)
	submit(t, s, 600, e1)
	got := preemptions(s)
	if len(got) != 1 {
		t.Fatalf("%d preemptions, want exactly 1", len(got))
	}
	if got[0].Job.ID != 2 || got[0].For != 3 {
		t.Fatalf("preempted job %d for job %d, want job 2 (newest id on a tie) for job 3", got[0].Job.ID, got[0].For)
	}
	if e1.State != model.Running {
		t.Fatalf("job 3 should run on the reclaimed node, state=%v", e1.State)
	}
	if r2.State != model.Pending || r2.Preemptions != 1 {
		t.Fatalf("job 2: state=%v preemptions=%d, want pending and 1", r2.State, r2.Preemptions)
	}
	if run := s.RunningByTeam(); run["r"] != 4 || run["e"] != 4 {
		t.Fatalf("running by team %v, want 4 and 4", run)
	}
}

func TestFairShareDoesNotPreemptWhenNothingIsPreemptible(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"r": 4, "e": 4}), cluster(2, 4))
	submit(t, s, 0, job(1, "r", 1, 4, 5000), job(2, "r", 1, 4, 5000))
	e1 := job(3, "e", 1, 4, 100)
	submit(t, s, 600, e1)
	if e1.State != model.Pending || s.Stats.Preemptions != 0 {
		t.Fatalf("state=%v preemptions=%d: jobs that are not preemptible must keep running", e1.State, s.Stats.Preemptions)
	}
}

func TestMinRunProtectsAFreshJob(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"r": 4, "e": 4}), cluster(2, 4))
	r1, r2 := job(1, "r", 1, 4, 5000), job(2, "r", 1, 4, 5000)
	r1.Preemptible, r2.Preemptible = true, true
	submit(t, s, 0, r1, r2)
	e1 := job(3, "e", 1, 4, 100)
	submit(t, s, 100, e1)
	if s.Stats.Preemptions != 0 {
		t.Fatalf("the borrowers have run for 100 s, less than the default MinRun of 300 s")
	}
	s.Tick(299)
	if s.Stats.Preemptions != 0 {
		t.Fatal("still inside MinRun at 299 s")
	}
	s.Tick(300)
	if s.Stats.Preemptions != 1 || e1.State != model.Running {
		t.Fatalf("preemptions=%d state=%v, want the reclaim to happen at 300 s", s.Stats.Preemptions, e1.State)
	}
}

func TestAJobStartedInThisTickCannotBePreempted(t *testing.T) {
	// MinRun is tiny, yet a job that was just started has run for zero seconds: reclaim and priority
	// cannot undo each other inside one Tick.
	cfg := fairShare(map[string]int{"x": 4, "y": 4})
	cfg.MinRun = 1e-9
	s := sched.New(cfg, cluster(2, 4))
	var js []*model.Job
	for i := 1; i <= 6; i++ {
		team := "x"
		if i%2 == 0 {
			team = "y"
		}
		j := job(i, team, 1, 4, 5000)
		j.Priority = i % 3
		j.Preemptible = true
		js = append(js, j)
	}
	submit(t, s, 10, js...)
	if n := s.Stats.Preemptions; n != 0 {
		t.Fatalf("%d preemptions at the very first Tick, want 0", n)
	}
}

func TestATeamWithinItsQuotaIsNeverPreemptedByAnotherTeam(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"a": 4, "b": 4}), cluster(2, 4))
	a1, b1 := job(1, "a", 1, 4, 5000), job(2, "b", 1, 4, 5000)
	a1.Priority, b1.Priority = 2, 0
	b1.Preemptible = true // b is at its quota, not above it
	submit(t, s, 0, a1, b1)
	a2 := job(3, "a", 1, 4, 100)
	a2.Priority = 2
	submit(t, s, 1000, a2)
	if a2.State != model.Pending || b1.State != model.Running || s.Stats.Preemptions != 0 {
		t.Fatalf("a2=%v b1=%v preemptions=%d: priority 2 must not take the guaranteed GPUs of team b",
			a2.State, b1.State, s.Stats.Preemptions)
	}
}

func TestPriorityPreemptsInsideATeam(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"t": 4}), cluster(1, 4))
	low := job(1, "t", 1, 4, 5000)
	low.Priority, low.Preemptible = 0, true
	submit(t, s, 0, low)
	high := job(2, "t", 1, 4, 100)
	high.Priority = 1
	submit(t, s, 600, high)
	if high.State != model.Running || low.State != model.Pending {
		t.Fatalf("high=%v low=%v, want the urgent job to take over inside its own team", high.State, low.State)
	}
}

func TestPriorityPreemptsABorrowerOfAnotherTeam(t *testing.T) {
	for _, tc := range []struct {
		name     string
		borrower int // priority of the borrowing jobs
		want     model.JobState
	}{
		{"lower priority borrower yields", 0, model.Running},
		{"equal priority borrower stays", 1, model.Pending},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := sched.New(fairShare(map[string]int{"x": 4, "y": 0}), cluster(2, 4))
			x1, x2 := job(1, "x", 1, 4, 5000), job(2, "x", 1, 4, 5000)
			for _, j := range []*model.Job{x1, x2} {
				j.Priority, j.Preemptible = tc.borrower, true
			}
			submit(t, s, 0, x1, x2) // x runs 8 GPUs on a quota of 4: it borrows
			y1 := job(3, "y", 1, 4, 100)
			y1.Priority = 1
			submit(t, s, 600, y1) // y has no quota, so it cannot reclaim, only outrank
			if y1.State != tc.want {
				t.Fatalf("y1 is %v, want %v", y1.State, tc.want)
			}
		})
	}
}

func TestPreemptionKeepsCheckpointedWork(t *testing.T) {
	cfg := fairShare(map[string]int{"t": 4})
	cfg.RestartCost, cfg.MinRun = 60, 10
	s := sched.New(cfg, cluster(1, 4))
	low := job(1, "t", 1, 4, 5000)
	low.Priority, low.Preemptible, low.CheckpointEvery = 0, true, 600
	submit(t, s, 0, low)
	s.Drain()
	high := job(2, "t", 1, 4, 100)
	high.Priority = 2
	submit(t, s, 1500, high)

	// 1500 s ran, the last checkpoint is at 1200 s: 300 s of work are lost, plus the restart cost.
	if want := 5000.0 - 1200 + 60; low.Remaining != want {
		t.Fatalf("remaining=%v, want %v", low.Remaining, want)
	}
	if want := (300.0 + 60) * 4; low.WastedGPUs != want {
		t.Fatalf("wasted=%v GPU-seconds, want %v", low.WastedGPUs, want)
	}
	ev := preemptions(s)
	if len(ev) != 1 || ev[0].Remaining != low.Remaining || ev[0].Gen != 0 || low.Gen != 1 {
		t.Fatalf("event=%+v gen=%d, want one event with the remaining time and the old generation 0", ev, low.Gen)
	}
	complete(t, s, 2, 1600)
	if low.State != model.Running || low.Start != 1600 {
		t.Fatalf("the preempted job should restart when the GPUs are free: state=%v start=%v", low.State, low.Start)
	}
}

func TestPreemptionWithoutCheckpointLosesEverything(t *testing.T) {
	cfg := fairShare(map[string]int{"t": 4})
	cfg.RestartCost, cfg.MinRun = 60, 10
	s := sched.New(cfg, cluster(1, 4))
	low := job(1, "t", 1, 4, 5000)
	low.Priority, low.Preemptible = 0, true
	submit(t, s, 0, low)
	high := job(2, "t", 1, 4, 100)
	high.Priority = 2
	submit(t, s, 1500, high)
	if want := 5000.0 + 60; low.Remaining != want {
		t.Fatalf("remaining=%v, want %v", low.Remaining, want)
	}
	if want := (1500.0 + 60) * 4; low.WastedGPUs != want {
		t.Fatalf("wasted=%v, want %v", low.WastedGPUs, want)
	}
}

func TestPreemptsTheVictimThatLosesTheLeastWork(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"b": 0, "a": 4}), cluster(3, 4))
	var bs []*model.Job
	for i := 1; i <= 3; i++ {
		j := job(i, "b", 1, 4, 9000)
		j.Preemptible = true
		if i == 1 {
			j.CheckpointEvery = 600 // only job 1 checkpoints
		}
		bs = append(bs, j)
	}
	submit(t, s, 0, bs...)
	s.Drain()
	submit(t, s, 1190, job(4, "a", 1, 4, 100))
	got := preemptions(s)
	if len(got) != 1 || got[0].Job.ID != 1 {
		t.Fatalf("preempted %v, want only job 1: its checkpoint at 600 s makes it the cheapest to lose", got)
	}
}

func TestPreemptsNoMoreThanItTakes(t *testing.T) {
	for _, nodes := range []int{1, 2} {
		s := sched.New(fairShare(map[string]int{"b": 0, "a": 8}), cluster(3, 4))
		for i := 1; i <= 3; i++ {
			j := job(i, "b", 1, 4, 9000)
			j.Preemptible = true
			submit(t, s, 0, j)
		}
		s.Drain()
		submit(t, s, 1000, job(4, "a", nodes, 4, 100))
		if got := len(preemptions(s)); got != nodes {
			t.Errorf("a gang of %d nodes preempted %d jobs, want %d", nodes, got, nodes)
		}
	}
}

// Node 0 holds c1 (a borrower) and a job that cannot be preempted, node 1 a job that cannot be
// preempted, node 2 holds a1 and a2 (borrowers). Taking victims in order, c1 comes first although
// its GPUs never free a whole node; only a1 and a2 are needed.
func TestPreemptsOnlyTheJobsThatFreeAWholeNode(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"x": 0, "y": 4}), cluster(3, 4))
	n1, z4 := job(5, "z", 1, 2, 9000), job(6, "z", 1, 4, 9000)
	c1, a1, a2 := job(9, "x", 1, 2, 9000), job(2, "x", 1, 2, 9000), job(1, "x", 1, 2, 9000)
	c1.Preemptible, a1.Preemptible, a2.Preemptible = true, true, true
	for _, j := range []*model.Job{n1, c1, z4, a1, a2} {
		submit(t, s, 0, j)
	}
	if n1.Placement[0].Node != 0 || c1.Placement[0].Node != 0 || z4.Placement[0].Node != 1 ||
		a1.Placement[0].Node != 2 || a2.Placement[0].Node != 2 {
		t.Fatalf("unexpected placement: %v %v %v %v %v", n1.Placement, c1.Placement, z4.Placement, a1.Placement, a2.Placement)
	}
	s.Drain()
	y1 := job(10, "y", 1, 4, 100)
	submit(t, s, 1000, y1)
	got := preemptions(s)
	if len(got) != 2 || c1.State != model.Running || a1.State != model.Pending || a2.State != model.Pending {
		t.Fatalf("%d preemptions, c1=%v a1=%v a2=%v: want a1 and a2 preempted and c1 left alone", len(got), c1.State, a1.State, a2.State)
	}
	if y1.State != model.Running {
		t.Fatalf("y1=%v, want running on node 2", y1.State)
	}
}

// Only the head and the jobs that fit their team's quota may start a preemption. The urgent job
// behind them waits, even though it outranks the borrowers it could take GPUs from.
func TestOnlyGuaranteedJobsAndTheHeadStartAPreemption(t *testing.T) {
	s := sched.New(fairShare(map[string]int{"x": 0, "y": 12, "w": 0}), cluster(3, 4))
	n1 := job(1, "z", 1, 4, 9000) // cannot be preempted
	x1, x2 := job(2, "x", 1, 4, 9000), job(3, "x", 1, 4, 9000)
	x1.Priority, x2.Priority = 0, 0
	x1.Preemptible, x2.Preemptible = true, true
	submit(t, s, 0, n1, x1, x2)
	s.Drain()

	// h needs the whole cluster and is within its quota, but n1 will never give way: h cannot start.
	h := job(4, "y", 3, 4, 100)
	// j is above its quota (w has none) and has priority 2: it could displace x1 or x2.
	j := job(5, "w", 1, 4, 100)
	j.Priority = 2
	submit(t, s, 1000, h, j)
	if j.State != model.Pending || s.Stats.Preemptions != 0 {
		t.Fatalf("j=%v preemptions=%d: a job that is neither at the head nor within its quota must wait",
			j.State, s.Stats.Preemptions)
	}
}

func TestRunningByTeamAndRunningJobs(t *testing.T) {
	s := sched.New(sched.Config{Kind: sched.Backfill}, cluster(2, 4))
	submit(t, s, 0, job(1, "a", 1, 4, 10), job(2, "b", 1, 2, 10))
	if got := s.RunningByTeam(); got["a"] != 4 || got["b"] != 2 {
		t.Fatalf("by team %v, want a=4 b=2", got)
	}
	if len(s.RunningJobs()) != 2 {
		t.Fatalf("%d running jobs, want 2", len(s.RunningJobs()))
	}
}

func TestKindString(t *testing.T) {
	for k, want := range map[sched.Kind]string{sched.FIFO: "fifo", sched.Backfill: "backfill", sched.FairShare: "fairshare", sched.Kind(9): "unknown"} {
		if k.String() != want {
			t.Errorf("%d: %q, want %q", k, k.String(), want)
		}
	}
}

func TestDrainReturnsEachEventOnce(t *testing.T) {
	s := sched.New(sched.Config{Kind: sched.FIFO}, cluster(1, 4))
	submit(t, s, 0, job(1, "a", 1, 4, 10))
	if n := len(s.Drain()); n != 1 {
		t.Fatalf("%d events, want the one start", n)
	}
	if n := len(s.Drain()); n != 0 {
		t.Fatalf("%d events on the second call, want 0", n)
	}
}

// BenchmarkTick measures one scheduling pass over a full cluster and a deep queue, the case that
// costs the most: nothing can start, so every candidate is examined.
func BenchmarkTick(b *testing.B) {
	for _, tc := range []struct {
		name string
		cfg  sched.Config
	}{
		{"backfill", sched.Config{Kind: sched.Backfill}},
		{"fairshare", sched.Config{Kind: sched.FairShare, Quota: map[string]int{"a": 24, "b": 24, "c": 16}, Preempt: true}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			s := sched.New(tc.cfg, model.NewCluster(8, 8, 4))
			teams := []string{"a", "b", "c"}
			for i := 1; i <= 8; i++ {
				j := job(i, teams[i%3], 1, 8, 1e9)
				j.Submit = 0
				if err := s.Submit(j); err != nil {
					b.Fatal(err)
				}
			}
			s.Tick(0)
			for i := 9; i <= 1008; i++ {
				j := job(i, teams[i%3], 1+i%4, 1+i%8, 3600)
				j.Submit = float64(i)
				if err := s.Submit(j); err != nil {
					b.Fatal(err)
				}
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s.Tick(2000)
			}
		})
	}
}
