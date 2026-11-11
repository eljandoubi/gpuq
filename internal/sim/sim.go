// Package sim replays a workload through the scheduler on a virtual clock.
package sim

import (
	"container/heap"
	"math"
	"sort"
	"time"

	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
)

// Spec describes the simulated cluster.
type Spec struct {
	Nodes        int
	GPUsPerNode  int
	NodesPerRack int
}

// Result summarises one run. Waits are in seconds, latencies in microseconds.
type Result struct {
	Policy                string
	Jobs                  int
	Makespan              float64 // first submission to last completion
	Utilization           float64 // GPU-time in use between the first and the last submission
	UsefulUtilization     float64 // the part of it that ended up in finished work
	WaitMean              float64 // submission to first start
	WaitP50               float64
	WaitP95               float64
	WaitP95Large          float64 // jobs with 32 GPUs or more
	WaitP95Small          float64 // jobs with 4 GPUs or fewer
	TeamWaitP95           map[string]float64
	SlowdownP95           float64 // (submission to completion) / max(runtime, 60 s), at least 1
	Preemptions           int
	WastedGPUHours        float64
	CrossRackFraction     float64 // of multi-node starts
	DecisionP50us         float64
	DecisionP99us         float64
	ReservationViolations int // head jobs that started later than the start time promised to them
}

type fin struct {
	t   float64
	id  int
	gen int
}

type finHeap []fin

func (h finHeap) Len() int { return len(h) }
func (h finHeap) Less(i, j int) bool {
	if h[i].t != h[j].t {
		return h[i].t < h[j].t
	}
	return h[i].id < h[j].id
}
func (h finHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *finHeap) Push(x any)   { *h = append(*h, x.(fin)) }
func (h *finHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// clock integrates the GPU-seconds in use over the whole run and over the arrival window.
type clock struct {
	first, lastSubmit, last float64
	busy, busyArriving      float64
}

func (c *clock) advance(t float64, gpusInUse int) {
	c.busy += float64(gpusInUse) * (t - c.last)
	c.busyArriving += float64(gpusInUse) * (math.Min(t, c.lastSubmit) - math.Min(c.last, c.lastSubmit))
	c.last = t
}

// Run replays jobs through a fresh scheduler. The input jobs are copied, so one workload can be
// replayed under several policies.
func Run(spec Spec, cfg sched.Config, jobs []*model.Job) Result {
	r, _, _ := RunDetailed(spec, cfg, jobs)
	return r
}

// RunDetailed is Run plus the final state of every job and of the cluster.
func RunDetailed(spec Spec, cfg sched.Config, jobs []*model.Job) (Result, []*model.Job, *model.Cluster) {
	return Replay(spec, cfg, jobs, nil)
}

// Replay is RunDetailed that also hands every scheduling event to observe, in the order the
// scheduler produced them.
func Replay(spec Spec, cfg sched.Config, jobs []*model.Job, observe func(sched.Event)) (Result, []*model.Job, *model.Cluster) {
	js := make([]*model.Job, len(jobs))
	byID := make(map[int]*model.Job, len(jobs))
	for i, j := range jobs {
		cp := *j
		js[i] = &cp
		byID[cp.ID] = &cp
	}
	sort.SliceStable(js, func(a, b int) bool { return js[a].Submit < js[b].Submit })

	cl := model.NewCluster(spec.Nodes, spec.GPUsPerNode, spec.NodesPerRack)
	s := sched.New(cfg, cl)
	h := &finHeap{}
	var lat []float64
	var c clock
	if len(js) > 0 {
		c.first, c.last, c.lastSubmit = js[0].Submit, js[0].Submit, js[len(js)-1].Submit
	}
	next, inUse := 0, 0

	for {
		for h.Len() > 0 {
			top := (*h)[0]
			if j := byID[top.id]; j.State == model.Running && j.Gen == top.gen {
				break
			}
			heap.Pop(h)
		}
		hasFin, hasArr := h.Len() > 0, next < len(js)
		if !hasFin && !hasArr {
			break
		}
		var t float64
		switch {
		case hasFin && hasArr:
			t = math.Min((*h)[0].t, js[next].Submit)
		case hasFin:
			t = (*h)[0].t
		default:
			t = js[next].Submit
		}
		c.advance(t, inUse)

		for h.Len() > 0 && (*h)[0].t <= t {
			f := heap.Pop(h).(fin)
			j := byID[f.id]
			if j.State != model.Running || j.Gen != f.gen {
				continue
			}
			if err := s.Complete(f.id, t); err != nil {
				panic(err)
			}
			inUse -= j.GPUs()
		}
		for next < len(js) && js[next].Submit <= t {
			if err := s.Submit(js[next]); err != nil {
				panic(err)
			}
			next++
		}

		began := time.Now()
		s.Tick(t)
		lat = append(lat, float64(time.Since(began).Nanoseconds())/1e3)

		for _, ev := range s.Drain() {
			if observe != nil {
				observe(ev)
			}
			switch ev.Kind {
			case sched.Start:
				heap.Push(h, fin{t: ev.Time + ev.Remaining, id: ev.Job.ID, gen: ev.Gen})
				inUse += ev.Job.GPUs()
			case sched.Preempt:
				inUse -= ev.Job.GPUs()
			}
		}
	}

	return summarize(cfg, s, cl, js, c, lat), js, cl
}

func summarize(cfg sched.Config, s *sched.Scheduler, cl *model.Cluster, js []*model.Job, c clock, lat []float64) Result {
	r := Result{Policy: cfg.Kind.String(), Jobs: len(js), Makespan: c.last - c.first, Preemptions: s.Stats.Preemptions}
	var waits, large, small, slow []float64
	byTeam := map[string][]float64{}
	wasted := 0.0
	for _, j := range js {
		w := j.FirstStart - j.Submit
		waits = append(waits, w)
		if j.GPUs() >= 32 {
			large = append(large, w)
		}
		if j.GPUs() <= 4 {
			small = append(small, w)
		}
		byTeam[j.Team] = append(byTeam[j.Team], w)
		slow = append(slow, math.Max(1, (j.End-j.Submit)/math.Max(j.Runtime, 60)))
		wasted += j.WastedGPUs
		if res, ok := s.Reserved(j.ID); ok && j.FirstStart > res+1e-6 {
			r.ReservationViolations++
		}
	}
	if window := c.lastSubmit - c.first; window > 0 {
		r.Utilization = c.busyArriving / (float64(cl.Total()) * window)
	}
	if c.busy > 0 {
		r.UsefulUtilization = r.Utilization * (c.busy - wasted) / c.busy
	}
	r.WastedGPUHours = wasted / 3600
	for _, sl := range [][]float64{waits, large, small, slow, lat} {
		sort.Float64s(sl)
	}
	r.WaitMean = mean(waits)
	r.WaitP50, r.WaitP95 = pct(waits, 0.50), pct(waits, 0.95)
	r.WaitP95Large, r.WaitP95Small = pct(large, 0.95), pct(small, 0.95)
	r.SlowdownP95 = pct(slow, 0.95)
	r.TeamWaitP95 = make(map[string]float64, len(byTeam))
	for team, w := range byTeam {
		sort.Float64s(w)
		r.TeamWaitP95[team] = pct(w, 0.95)
	}
	r.DecisionP50us, r.DecisionP99us = pct(lat, 0.50), pct(lat, 0.99)
	if s.Stats.MultiNodeStarts > 0 {
		r.CrossRackFraction = float64(s.Stats.CrossRackStarts) / float64(s.Stats.MultiNodeStarts)
	}
	return r
}

func mean(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	s := 0.0
	for _, v := range x {
		s += v
	}
	return s / float64(len(x))
}

// pct returns the p-quantile of an ascending slice (nearest rank).
func pct(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(math.Ceil(p*float64(len(sorted)))) - 1
	if i < 0 {
		i = 0
	}
	return sorted[i]
}
