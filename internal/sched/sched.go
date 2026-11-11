// Package sched implements gang scheduling for a GPU cluster: strict FIFO, EASY-style backfill and
// a fair-share policy with per-team quotas, borrowing, reclaim by preemption and aging.
package sched

import (
	"fmt"
	"math"
	"sort"

	"github.com/eljandoubi/gpuq/internal/model"
)

// Kind selects the scheduling policy.
type Kind int

const (
	// FIFO starts only the job at the head of the queue (head-of-line blocking).
	FIFO Kind = iota
	// Backfill is FIFO plus backfilling: later jobs may start if they cannot delay the head job.
	Backfill
	// FairShare orders the queue by aging, quota, priority and decayed usage, preempts borrowers to
	// give a team its guaranteed share back, and backfills.
	FairShare
)

func (k Kind) String() string {
	switch k {
	case FIFO:
		return "fifo"
	case Backfill:
		return "backfill"
	case FairShare:
		return "fairshare"
	}
	return "unknown"
}

// Config tunes the scheduler.
type Config struct {
	Kind Kind
	// LookAhead bounds how many pending jobs are examined for backfill (default 256).
	LookAhead int
}

// EventKind tells what happened to a job.
type EventKind int

const (
	Start EventKind = iota
	Preempt
	Finish
)

// Event is a scheduling decision; Gen and Remaining are snapshots taken when it happened. For is
// the job a preemption made room for (0 for the other kinds).
type Event struct {
	Kind      EventKind
	Job       *model.Job
	Time      float64
	Gen       int
	Remaining float64
	For       int
}

// Stats counts scheduling decisions.
type Stats struct {
	Starts          int
	MultiNodeStarts int
	CrossRackStarts int
	Preemptions     int
}

// Scheduler owns the queue and the cluster state. It is not safe for concurrent use.
type Scheduler struct {
	cfg      Config
	cl       *model.Cluster
	pending  []*model.Job
	running  map[int]*model.Job
	events   []Event
	reserved map[int]float64
	Stats    Stats
}

// New returns a scheduler that allocates from cl.
func New(cfg Config, cl *model.Cluster) *Scheduler {
	if cfg.LookAhead <= 0 {
		cfg.LookAhead = 256
	}
	return &Scheduler{
		cfg:      cfg,
		cl:       cl,
		running:  map[int]*model.Job{},
		reserved: map[int]float64{},
	}
}

// Cluster returns the cluster the scheduler allocates from.
func (s *Scheduler) Cluster() *model.Cluster { return s.cl }

// Pending is the number of queued jobs.
func (s *Scheduler) Pending() int { return len(s.pending) }

// Running is the number of running jobs.
func (s *Scheduler) Running() int { return len(s.running) }

// RunningByTeam returns the GPUs each team is running on.
func (s *Scheduler) RunningByTeam() map[string]int {
	out := map[string]int{}
	for _, r := range s.running {
		out[r.Team] += r.GPUs()
	}
	return out
}

// RunningJobs returns the running jobs in no particular order.
func (s *Scheduler) RunningJobs() []*model.Job {
	out := make([]*model.Job, 0, len(s.running))
	for _, r := range s.running {
		out = append(out, r)
	}
	return out
}

// Reserved returns the start time promised to a job while it was the head of the queue.
func (s *Scheduler) Reserved(id int) (float64, bool) {
	t, ok := s.reserved[id]
	return t, ok
}

// Drain returns the events produced since the last call.
func (s *Scheduler) Drain() []Event {
	out := s.events
	s.events = nil
	return out
}

// Submit queues a job. A job that could never fit on the cluster is rejected.
func (s *Scheduler) Submit(j *model.Job) error {
	if !s.cl.Fits(j.Nodes, j.GPUsPerNode) {
		return fmt.Errorf("job %d asks for %d x %d GPUs, which this cluster can never provide", j.ID, j.Nodes, j.GPUsPerNode)
	}
	j.State = model.Pending
	j.Remaining = j.Runtime
	j.FirstStart = -1
	s.insertPending(j)
	return nil
}

// Complete marks a running job as finished and frees its GPUs.
func (s *Scheduler) Complete(id int, now float64) error {
	r, ok := s.running[id]
	if !ok {
		return fmt.Errorf("job %d is not running", id)
	}
	if err := s.cl.Release(r.Placement); err != nil {
		return err
	}
	delete(s.running, id)
	r.State = model.Done
	r.End = now
	r.Remaining = 0
	s.events = append(s.events, Event{Kind: Finish, Job: r, Time: now, Gen: r.Gen})
	return nil
}

// Tick starts every job the policy allows at time now.
func (s *Scheduler) Tick(now float64) {
	for s.step(now) {
	}
}

func (s *Scheduler) less(a, b *model.Job) bool {
	if a.Submit != b.Submit {
		return a.Submit < b.Submit
	}
	return a.ID < b.ID
}

func (s *Scheduler) insertPending(j *model.Job) {
	i := sort.Search(len(s.pending), func(i int) bool { return s.less(j, s.pending[i]) })
	s.pending = append(s.pending, nil)
	copy(s.pending[i+1:], s.pending[i:])
	s.pending[i] = j
}

func (s *Scheduler) removePending(j *model.Job) {
	for i, p := range s.pending {
		if p == j {
			s.pending = append(s.pending[:i], s.pending[i+1:]...)
			return
		}
	}
}

func (s *Scheduler) step(now float64) bool {
	if len(s.pending) == 0 {
		return false
	}
	q := append([]*model.Job(nil), s.pending...)
	head := q[0]
	if s.tryStart(head, now) {
		return true
	}
	if s.cfg.Kind == FIFO {
		return false
	}
	shadow := s.shadow(head)
	if _, seen := s.reserved[head.ID]; !seen {
		s.reserved[head.ID] = shadow
	}
	limit := min(len(q), 1+s.cfg.LookAhead)
	for _, j := range q[1:limit] {
		if now+j.Remaining <= shadow+1e-9 && s.tryStart(j, now) {
			return true
		}
	}
	return false
}

func (s *Scheduler) tryStart(j *model.Job, now float64) bool {
	al, cross, ok := s.cl.Place(j.Nodes, j.GPUsPerNode)
	if !ok {
		return false
	}
	if err := s.cl.Allocate(al); err != nil {
		panic(err) // Place just verified the capacity
	}
	s.removePending(j)
	j.State = model.Running
	j.Start = now
	if j.FirstStart < 0 {
		j.FirstStart = now
	}
	j.Placement = al
	s.running[j.ID] = j
	s.Stats.Starts++
	if j.Nodes > 1 {
		s.Stats.MultiNodeStarts++
		if cross {
			s.Stats.CrossRackStarts++
		}
	}
	s.events = append(s.events, Event{Kind: Start, Job: j, Time: now, Gen: j.Gen, Remaining: j.Remaining})
	return true
}

// shadow is the earliest time the head job could start if running jobs end as planned.
func (s *Scheduler) shadow(head *model.Job) float64 {
	trial := s.cl.Clone()
	ends := make([]*model.Job, 0, len(s.running))
	for _, r := range s.running {
		ends = append(ends, r)
	}
	sort.Slice(ends, func(a, b int) bool {
		ea, eb := ends[a].Start+ends[a].Remaining, ends[b].Start+ends[b].Remaining
		if ea != eb {
			return ea < eb
		}
		return ends[a].ID < ends[b].ID
	})
	for _, r := range ends {
		if err := trial.Release(r.Placement); err != nil {
			panic(err)
		}
		if _, _, ok := trial.Place(head.Nodes, head.GPUsPerNode); ok {
			return r.Start + r.Remaining
		}
	}
	return math.Inf(1)
}
