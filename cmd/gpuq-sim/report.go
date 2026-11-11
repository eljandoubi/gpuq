package main

import (
	"fmt"
	"math"

	"github.com/eljandoubi/gpuq/internal/sim"
)

type col struct {
	name string
	get  func(sim.Result) float64
	fmt  string
}

func minutes(f func(sim.Result) float64) func(sim.Result) float64 {
	return func(r sim.Result) float64 { return f(r) / 60 }
}

// printReport prints the three tables of one comparison as Markdown.
func printReport(all [][]sim.Result, teams []sim.TeamCfg) {
	fmt.Print("\n**Throughput**\n\n")
	printTable(all, []col{
		{"utilization", func(r sim.Result) float64 { return 100 * r.Utilization }, "%.1f%%"},
		{"useful util.", func(r sim.Result) float64 { return 100 * r.UsefulUtilization }, "%.1f%%"},
		{"makespan (days)", func(r sim.Result) float64 { return r.Makespan / 86400 }, "%.1f"},
		{"preemptions", func(r sim.Result) float64 { return float64(r.Preemptions) }, "%.0f"},
		{"wasted GPU-h", func(r sim.Result) float64 { return r.WastedGPUHours }, "%.0f"},
		{"cross-rack", func(r sim.Result) float64 { return 100 * r.CrossRackFraction }, "%.1f%%"},
		{"decision p99 (us)", func(r sim.Result) float64 { return r.DecisionP99us }, "%.0f"},
	})

	fmt.Print("\n**Waiting** (minutes from submission to first start)\n\n")
	printTable(all, []col{
		{"p50", minutes(func(r sim.Result) float64 { return r.WaitP50 }), "%.1f"},
		{"p95", minutes(func(r sim.Result) float64 { return r.WaitP95 }), "%.0f"},
		{"p95, jobs of 32+ GPUs", minutes(func(r sim.Result) float64 { return r.WaitP95Large }), "%.0f"},
		{"p95, jobs of up to 4 GPUs", minutes(func(r sim.Result) float64 { return r.WaitP95Small }), "%.0f"},
		{"p95 slowdown", func(r sim.Result) float64 { return r.SlowdownP95 }, "%.1f"},
	})

	fmt.Print("\n**p95 wait per team** (minutes; quota in GPUs, share of the offered GPU-time)\n\n")
	var cols []col
	for _, t := range teams {
		name := t.Name
		cols = append(cols, col{
			fmt.Sprintf("%s (%d GPUs, %.0f%%)", name, t.Quota, 100*t.Share),
			func(r sim.Result) float64 { return r.TeamWaitP95[name] / 60 },
			"%.0f",
		})
	}
	printTable(all, cols)
}

func printTable(all [][]sim.Result, cols []col) {
	fmt.Print("| policy |")
	for _, c := range cols {
		fmt.Printf(" %s |", c.name)
	}
	fmt.Print("\n|---|")
	for range cols {
		fmt.Print("---:|")
	}
	fmt.Println()
	for _, runs := range all {
		fmt.Printf("| %s |", runs[0].Policy)
		for _, c := range cols {
			vals := make([]float64, len(runs))
			for i, r := range runs {
				vals[i] = c.get(r)
			}
			m, sd := meanStd(vals)
			fmt.Printf(" "+c.fmt+" (±"+c.fmt+") |", m, sd)
		}
		fmt.Println()
	}
}

// printSweep prints p95 wait and utilization for each offered load.
func printSweep(loads []float64, labels []string, byLoad [][][]sim.Result) {
	show := func(title string, get func(sim.Result) float64, f string) {
		fmt.Printf("**%s**\n\n| offered load |", title)
		for _, l := range labels {
			fmt.Printf(" %s |", l)
		}
		fmt.Print("\n|---:|")
		for range labels {
			fmt.Print("---:|")
		}
		fmt.Println()
		for i, load := range loads {
			fmt.Printf("| %.2f |", load)
			for _, runs := range byLoad[i] {
				vals := make([]float64, len(runs))
				for k, r := range runs {
					vals[k] = get(r)
				}
				m, _ := meanStd(vals)
				fmt.Printf(" "+f+" |", m)
			}
			fmt.Println()
		}
		fmt.Println()
	}
	show("p95 wait (minutes)", minutes(func(r sim.Result) float64 { return r.WaitP95 }), "%.0f")
	show("useful utilization while jobs arrive (work lost to preemption excluded)", func(r sim.Result) float64 { return 100 * r.UsefulUtilization }, "%.1f%%")
}

func meanStd(x []float64) (float64, float64) {
	m := 0.0
	for _, v := range x {
		m += v
	}
	m /= float64(len(x))
	v := 0.0
	for _, a := range x {
		v += (a - m) * (a - m)
	}
	return m, math.Sqrt(v / float64(len(x)))
}
