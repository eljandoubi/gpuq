package sim

import (
	"math"
	"math/rand"

	"github.com/eljandoubi/gpuq/internal/model"
)

// Shape is a gang size (Nodes x PerNode GPUs) and the probability that a team submits it.
type Shape struct {
	Nodes, PerNode int
	P              float64
}

// TeamCfg is one tenant: its guaranteed GPUs, its share of the offered GPU-time and what it runs.
type TeamCfg struct {
	Name   string
	Quota  int     // GPUs guaranteed to the team
	Share  float64 // share of the offered GPU-time (the shares are normalised)
	Mix    []Shape // gang shapes the team submits
	Median float64 // median runtime in seconds (log-normal, sigma 1.1, 60 s to 24 h)
	// Priorities is the probability of priority 0, 1 and 2. Priority 2 is never preemptible and
	// priority 0 always is; PreemptP is the chance that a priority 1 job is preemptible.
	Priorities [3]float64
	PreemptP   float64
	// CheckpointP is the chance that a preemptible job checkpoints every 10 minutes.
	CheckpointP float64
}

// WorkloadCfg parameterises the synthetic workload.
type WorkloadCfg struct {
	Seed  int64
	Jobs  int
	Load  float64 // offered load as a fraction of cluster capacity (1.0 = exactly saturated)
	Spec  Spec
	Teams []TeamCfg
}

// DefaultTeams is a research group that asks for more than its quota (so it borrows), a training
// team that runs few, large, long jobs and leaves part of its quota idle, and an evaluation team
// with many short jobs that must not wait behind the others. Quotas add up to 64 GPUs. No job takes
// the whole cluster: such jobs need a dedicated window, not a queue.
func DefaultTeams() []TeamCfg {
	return []TeamCfg{
		{
			Name: "research", Quota: 24, Share: 0.50, Median: 1800,
			Mix:        []Shape{{1, 1, 0.30}, {1, 2, 0.15}, {1, 4, 0.17}, {1, 8, 0.20}, {2, 8, 0.12}, {4, 8, 0.06}},
			Priorities: [3]float64{0.40, 0.55, 0.05}, PreemptP: 0.5, CheckpointP: 0.7,
		},
		{
			Name: "training", Quota: 32, Share: 0.35, Median: 7200,
			Mix:        []Shape{{1, 8, 0.20}, {2, 8, 0.40}, {4, 8, 0.40}},
			Priorities: [3]float64{0.30, 0.65, 0.05}, PreemptP: 0.8, CheckpointP: 0.95,
		},
		{
			Name: "eval", Quota: 8, Share: 0.15, Median: 600,
			Mix:        []Shape{{1, 1, 0.35}, {1, 2, 0.25}, {1, 4, 0.25}, {1, 8, 0.15}},
			Priorities: [3]float64{0, 0.7, 0.3},
		},
	}
}

func pickShape(rng *rand.Rand, mix []Shape, spec Spec) (int, int) {
	u := rng.Float64()
	acc := 0.0
	n, per := 1, 1
	for _, s := range mix {
		acc += s.P
		n, per = s.Nodes, s.PerNode
		if u < acc {
			break
		}
	}
	return min(n, spec.Nodes), min(per, spec.GPUsPerNode)
}

func duration(rng *rand.Rand, median float64) float64 {
	d := math.Exp(math.Log(median) + 1.1*rng.NormFloat64())
	return math.Min(math.Max(d, 60), 86400)
}

// meanGPUSeconds estimates the GPU-seconds one job of the team asks for.
func meanGPUSeconds(seed int64, spec Spec, t TeamCfg) float64 {
	rng := rand.New(rand.NewSource(seed))
	const n = 20000
	sum := 0.0
	for i := 0; i < n; i++ {
		nodes, per := pickShape(rng, t.Mix, spec)
		sum += float64(nodes*per) * duration(rng, t.Median)
	}
	return sum / n
}

// Generate returns a reproducible workload: independent Poisson arrivals per team, each team with
// its own gang sizes, log-normal runtimes, priority classes and preemptible jobs. Load fixes the
// total GPU-time offered, split between the teams by Share.
func Generate(c WorkloadCfg) []*model.Job {
	rng := rand.New(rand.NewSource(c.Seed))
	capacity := float64(c.Spec.Nodes * c.Spec.GPUsPerNode)
	teams := c.Teams
	if len(teams) == 0 {
		teams = DefaultTeams()
	}
	shares := 0.0
	for _, t := range teams {
		shares += t.Share
	}
	rate := make([]float64, len(teams))
	total := 0.0
	for i, t := range teams {
		rate[i] = c.Load * capacity * (t.Share / shares) / meanGPUSeconds(c.Seed+1, c.Spec, t)
		total += rate[i]
	}

	jobs := make([]*model.Job, 0, c.Jobs)
	now := 0.0
	for id := 1; id <= c.Jobs; id++ {
		now += rng.ExpFloat64() / total
		u, acc, k := rng.Float64()*total, 0.0, len(teams)-1
		for i, r := range rate {
			acc += r
			if u < acc {
				k = i
				break
			}
		}
		t := teams[k]
		nodes, per := pickShape(rng, t.Mix, c.Spec)
		j := &model.Job{ID: id, Team: t.Name, Nodes: nodes, GPUsPerNode: per, Runtime: duration(rng, t.Median), Submit: now}
		switch p := rng.Float64(); {
		case p < t.Priorities[0]:
			j.Priority, j.Preemptible = 0, true
		case p < t.Priorities[0]+t.Priorities[1]:
			j.Priority, j.Preemptible = 1, rng.Float64() < t.PreemptP
		default:
			j.Priority = 2
		}
		if j.Preemptible && rng.Float64() < t.CheckpointP {
			j.CheckpointEvery = 600
		}
		jobs = append(jobs, j)
	}
	return jobs
}
