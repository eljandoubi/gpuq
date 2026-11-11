package sched

import (
	"math"
	"sort"

	"github.com/eljandoubi/gpuq/internal/model"
)

// canPreempt reports whether head may take the GPUs of the running job v. left is what each team
// would still be running if the victims chosen so far were gone; headWithin tells whether head fits
// inside its own team's quota. The rules are:
//
//  1. only a preemptible job that has run for at least MinRun seconds can be preempted;
//  2. inside one team, a job yields to a job of higher priority;
//  3. across teams, only a team that runs above its quota can lose GPUs: a team that stays within
//     its quota is never touched by another team, whatever the priorities;
//  4. such a borrower yields to a head that fits its own quota (reclaim) or has a higher priority.
//
// Rule 3 is what keeps reclaim and priority from undoing each other. Rule 1 guarantees that a Tick
// ends: a job that was started or restarted in this Tick has run for zero seconds.
func (s *Scheduler) canPreempt(head, v *model.Job, now float64, left map[string]int, headWithin bool) bool {
	if !v.Preemptible || now-v.Start < s.cfg.MinRun {
		return false
	}
	if v.Team == head.Team {
		return v.Priority < head.Priority
	}
	if left[v.Team] <= s.cfg.Quota[v.Team] {
		return false
	}
	return headWithin || v.Priority < head.Priority
}

// preemptFor frees room for head by preempting the jobs canPreempt allows. It acts only when the
// freed GPUs are enough for head to start, and it preempts no more jobs than that takes. Victims
// are taken lowest priority first, then from the team furthest above its quota, then the job that
// loses the least work (progress since its last checkpoint plus the restart cost, in GPU-seconds).
// run is the GPUs each team is running on.
func (s *Scheduler) preemptFor(head *model.Job, now float64, run map[string]int) bool {
	quota := s.cfg.Quota
	headWithin := run[head.Team]+head.GPUs() <= quota[head.Team]

	var victims []*model.Job
	for _, r := range s.running {
		if s.canPreempt(head, r, now, run, headWithin) {
			victims = append(victims, r)
		}
	}
	if len(victims) == 0 {
		return false
	}
	lost := make(map[int]float64, len(victims))
	for _, v := range victims {
		lost[v.ID] = (now - v.Start - progressKept(v, now) + s.cfg.RestartCost) * float64(v.GPUs())
	}
	sort.Slice(victims, func(a, b int) bool {
		va, vb := victims[a], victims[b]
		if va.Priority != vb.Priority {
			return va.Priority < vb.Priority
		}
		if oa, ob := run[va.Team]-quota[va.Team], run[vb.Team]-quota[vb.Team]; oa != ob {
			return oa > ob
		}
		if lost[va.ID] != lost[vb.ID] {
			return lost[va.ID] < lost[vb.ID]
		}
		return va.ID > vb.ID
	})

	left := make(map[string]int, len(run))
	for t, g := range run {
		left[t] = g
	}
	trial := s.cl.Clone()
	var chosen []*model.Job
	fits := false
	for _, v := range victims {
		if !s.canPreempt(head, v, now, left, headWithin) {
			continue
		}
		if err := trial.Release(v.Placement); err != nil {
			panic(err)
		}
		left[v.Team] -= v.GPUs()
		chosen = append(chosen, v)
		if _, _, ok := trial.Place(head.Nodes, head.GPUsPerNode); ok {
			fits = true
			break
		}
	}
	if !fits {
		return false
	}
	for _, v := range s.minimal(head, chosen) {
		s.preempt(v, now, head.ID)
	}
	return true
}

// minimal drops, last chosen first, every victim whose GPUs head does not need. Dropping a victim
// never breaks rule 3: its team can only run more than it did when the others were chosen.
func (s *Scheduler) minimal(head *model.Job, chosen []*model.Job) []*model.Job {
	for i := len(chosen) - 1; i >= 0; i-- {
		trial := s.cl.Clone()
		for k, v := range chosen {
			if k == i {
				continue
			}
			if err := trial.Release(v.Placement); err != nil {
				panic(err)
			}
		}
		if _, _, ok := trial.Place(head.Nodes, head.GPUsPerNode); ok {
			chosen = append(chosen[:i:i], chosen[i+1:]...)
		}
	}
	return chosen
}

// progressKept is the part of the current run, in seconds, that survives a preemption at now:
// everything up to the last checkpoint, or nothing for a job that does not checkpoint.
func progressKept(v *model.Job, now float64) float64 {
	if v.CheckpointEvery <= 0 {
		return 0
	}
	return math.Floor((now-v.Start)/v.CheckpointEvery) * v.CheckpointEvery
}

// preempt stops v, keeps the work it checkpointed and puts it back in the queue. forID is the job
// the GPUs are freed for.
func (s *Scheduler) preempt(v *model.Job, now float64, forID int) {
	elapsed := now - v.Start
	saved := progressKept(v, now)
	v.Remaining += s.cfg.RestartCost - saved
	v.WastedGPUs += (elapsed - saved + s.cfg.RestartCost) * float64(v.GPUs())
	if err := s.cl.Release(v.Placement); err != nil {
		panic(err)
	}
	delete(s.running, v.ID)
	v.State = model.Pending
	v.Queued = now
	v.Placement = nil
	v.Preemptions++
	v.Gen++
	s.Stats.Preemptions++
	s.events = append(s.events, Event{Kind: Preempt, Job: v, Time: now, Gen: v.Gen - 1, Remaining: v.Remaining, For: forID})
	s.insertPending(v)
}
