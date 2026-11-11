// Command gpuq-sim compares scheduling policies on a synthetic GPU-cluster workload.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
	"github.com/eljandoubi/gpuq/internal/sim"
)

type setup struct {
	spec    sim.Spec
	teams   []sim.TeamCfg
	configs []sched.Config
	labels  []string
	jobs    int
	seeds   int
}

func main() {
	nodes := flag.Int("nodes", 8, "nodes in the cluster")
	gpus := flag.Int("gpus", 8, "GPUs per node")
	rack := flag.Int("rack", 4, "nodes per rack")
	jobs := flag.Int("jobs", 3000, "jobs per run")
	load := flag.Float64("load", 0.85, "offered load, 1.0 saturates the cluster")
	seeds := flag.Int("seeds", 10, "number of seeds to average over")
	maxWait := flag.Duration("max-wait", 6*time.Hour, "fairshare: serve a job first once it has waited this long (0 turns aging off)")
	minRun := flag.Duration("min-run", 5*time.Minute, "fairshare: a job runs this long before it can be preempted")
	sweep := flag.Bool("sweep", false, "print wait and utilization against the offered load instead of one table")
	asJSON := flag.Bool("json", false, "print the per-seed results as JSON instead of tables")
	flag.Parse()

	su := setup{
		spec:  sim.Spec{Nodes: *nodes, GPUsPerNode: *gpus, NodesPerRack: *rack},
		teams: sim.DefaultTeams(),
		jobs:  *jobs,
		seeds: *seeds,
	}
	su.configs, su.labels = policies(su.teams, *maxWait, *minRun)

	switch {
	case *sweep:
		loads := []float64{0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
		byLoad := make([][][]sim.Result, len(loads))
		for i, l := range loads {
			byLoad[i] = su.runAll(l)
		}
		fmt.Printf("cluster: %d GPUs, %d jobs per run, mean of %d seeds\n\n", su.spec.Nodes*su.spec.GPUsPerNode, su.jobs, su.seeds)
		printSweep(loads, su.labels, byLoad)
	case *asJSON:
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(su.runAll(*load)); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	default:
		all := su.runAll(*load)
		fmt.Printf("cluster: %d nodes x %d GPUs (%d GPUs), %d per rack; %d jobs per run, offered load %.2f, mean (±std) of %d seeds\n",
			*nodes, *gpus, *nodes**gpus, *rack, su.jobs, *load, su.seeds)
		printReport(all, su.teams)
	}
}

// policies returns the three policies under comparison. FairShare gets the quotas of the teams.
func policies(teams []sim.TeamCfg, maxWait, minRun time.Duration) ([]sched.Config, []string) {
	quota := map[string]int{}
	for _, t := range teams {
		quota[t.Name] = t.Quota
	}
	wait := maxWait.Seconds()
	if wait <= 0 {
		wait = -1 // a negative MaxWait turns aging off
	}
	return []sched.Config{
		{Kind: sched.FIFO},
		{Kind: sched.Backfill},
		{Kind: sched.FairShare, Quota: quota, Preempt: true, MaxWait: wait, MinRun: minRun.Seconds()},
	}, []string{"fifo", "backfill", "fairshare+preempt"}
}

// runAll replays the same seeded workloads under every policy.
func (su setup) runAll(load float64) [][]sim.Result {
	all := make([][]sim.Result, len(su.configs))
	for seed := 1; seed <= su.seeds; seed++ {
		w := sim.Generate(sim.WorkloadCfg{Seed: int64(seed), Jobs: su.jobs, Load: load, Spec: su.spec, Teams: su.teams})
		for i, cfg := range su.configs {
			r := sim.Run(su.spec, cfg, cloneJobs(w))
			r.Policy = su.labels[i]
			all[i] = append(all[i], r)
		}
	}
	return all
}

func cloneJobs(in []*model.Job) []*model.Job {
	out := make([]*model.Job, len(in))
	for i, j := range in {
		cp := *j
		out[i] = &cp
	}
	return out
}
