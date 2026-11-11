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
