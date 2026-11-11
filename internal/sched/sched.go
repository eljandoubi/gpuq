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
	// Quota is the number of GPUs each team is guaranteed. A team may run above its quota when
	// GPUs are idle (borrowing); borrowed GPUs are reclaimed by preemption when another team needs
	// its own share back. Teams without an entry have no guarantee.
	Quota map[string]int
	// Preempt allows FairShare to preempt preemptible jobs. The rules are in canPreempt: a team
	// within its quota never loses a job to another team, and only the head of the queue and the
	// queued jobs that fit their team's quota may start a preemption.
	Preempt bool
	// LookAhead bounds how many pending jobs are examined for backfill (default 256).
	LookAhead int
	// HalfLife is the half-life in seconds of the usage decay used for fairness (default 6h).
	HalfLife float64
	// MinRun is how long a job must have been running before it can be preempted (default 300 s,
	// like Slurm's PreemptExemptTime). It also makes every Tick terminate: a job started or
	// restarted during a Tick has run for zero seconds, so it cannot be preempted in that Tick.
	MinRun float64
	// RestartCost is the time in seconds, per GPU, that a preempted job spends restarting
	// (scheduling, process start, checkpoint load) on top of the work it lost since its last
	// checkpoint (default 120 s). It is added to the job's remaining time and to the wasted GPU-time.
	RestartCost float64
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
	usage    map[string]float64
	lastTick float64
	events   []Event
	reserved map[int]float64
	Stats    Stats
}

// New returns a scheduler that allocates from cl.
func New(cfg Config, cl *model.Cluster) *Scheduler {
	if cfg.LookAhead <= 0 {
		cfg.LookAhead = 256
	}
	if cfg.HalfLife <= 0 {
		cfg.HalfLife = 6 * 3600
	}
	if cfg.MinRun <= 0 {
		cfg.MinRun = 300
	}
	if cfg.RestartCost <= 0 {
		cfg.RestartCost = 120
	}
	return &Scheduler{
		cfg:      cfg,
		cl:       cl,
		running:  map[int]*model.Job{},
		usage:    map[string]float64{},
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
	s.accrue(now)
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
	s.accrue(now)
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

// accrue integrates GPU usage per team with exponential decay (fair-share accounting).
func (s *Scheduler) accrue(now float64) {
	dt := now - s.lastTick
	if dt <= 0 {
		return
	}
	decay := math.Pow(0.5, dt/s.cfg.HalfLife)
	for t := range s.usage {
		s.usage[t] *= decay
	}
	for _, r := range s.running {
		s.usage[r.Team] += float64(r.GPUs()) * dt
	}
	s.lastTick = now
}

// ordered returns the pending jobs in the order the policy wants to serve them. FairShare puts the
// jobs that fit their team's quota first, then orders by priority, then by decayed usage divided by
// quota (the team that has used the least of its share goes first), then by arrival.
func (s *Scheduler) ordered() []*model.Job {
	q := append([]*model.Job(nil), s.pending...)
	if s.cfg.Kind != FairShare {
		return q
	}
	run := s.RunningByTeam()
	type ranked struct {
		j    *model.Job
		over bool
		norm float64
	}
	r := make([]ranked, len(q))
	for i, j := range q {
		quota := float64(s.cfg.Quota[j.Team])
		r[i] = ranked{
			j:    j,
			over: run[j.Team]+j.GPUs() > s.cfg.Quota[j.Team],
			norm: s.usage[j.Team] / math.Max(1, quota),
		}
	}
	sort.Slice(r, func(a, b int) bool {
		x, y := r[a], r[b]
		if x.over != y.over {
			return !x.over
		}
		if x.j.Priority != y.j.Priority {
			return x.j.Priority > y.j.Priority
		}
		if x.norm != y.norm {
			return x.norm < y.norm
		}
		return s.less(x.j, y.j)
	})
	for i := range r {
		q[i] = r[i].j
	}
	return q
}

func (s *Scheduler) step(now float64) bool {
	if len(s.pending) == 0 {
		return false
	}
	q := s.ordered()
	head := q[0]
	if s.tryStart(head, now) {
		return true
	}
	if s.cfg.Kind == FIFO {
		return false
	}
	limit := min(len(q), 1+s.cfg.LookAhead)
	if s.cfg.Kind == FairShare && s.cfg.Preempt {
		// The head may preempt, and so may any queued job that fits its team's quota: getting a
		// guaranteed share back must not wait for the jobs ahead, which may be over quota.
		run := s.RunningByTeam()
		for i, j := range q[:limit] {
			if i > 0 && run[j.Team]+j.GPUs() > s.cfg.Quota[j.Team] {
				continue
			}
			if s.preemptFor(j, now, run) {
				if !s.tryStart(j, now) {
					panic(fmt.Sprintf("job %d does not fit after its victims were preempted", j.ID))
				}
				return true
			}
		}
	}
	shadow := s.shadow(head)
	if _, seen := s.reserved[head.ID]; !seen {
		s.reserved[head.ID] = shadow
	}
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
