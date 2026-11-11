// Package model holds the cluster and job types shared by the scheduler and the simulator.
package model

import (
	"fmt"
	"sort"
)

// JobState is the lifecycle state of a job.
type JobState int

const (
	Pending JobState = iota
	Running
	Done
)

func (s JobState) String() string {
	switch s {
	case Pending:
		return "pending"
	case Running:
		return "running"
	case Done:
		return "done"
	}
	return "unknown"
}

// Alloc is the number of GPUs a gang holds on one node.
type Alloc struct {
	Node int
	GPUs int
}

// Job is a gang-scheduled training job: Nodes x GPUsPerNode GPUs start together or not at all.
type Job struct {
	ID              int
	Team            string
	Priority        int
	Nodes           int
	GPUsPerNode     int
	Runtime         float64 // seconds of work
	Submit          float64
	Preemptible     bool
	CheckpointEvery float64 // seconds between checkpoints; 0 means a preemption loses all progress

	State       JobState
	Queued      float64 // when the job last entered the queue: submission or last preemption
	Remaining   float64
	Start       float64 // start of the current run
	FirstStart  float64
	End         float64
	Placement   []Alloc
	Preemptions int
	WastedGPUs  float64 // GPU-seconds of work lost to preemption (progress since the last checkpoint)
	Gen         int     // bumped on every preemption so stale completion events can be ignored
}

// GPUs is the total gang size.
func (j *Job) GPUs() int { return j.Nodes * j.GPUsPerNode }

// Node is one machine with a fixed number of interchangeable GPUs.
type Node struct {
	ID    int
	Rack  int
	Total int
	Free  int
}

// Cluster is a set of nodes grouped into racks.
type Cluster struct {
	Nodes []Node
}

// NewCluster builds a homogeneous cluster. Nodes are numbered rack by rack.
func NewCluster(nodes, gpusPerNode, nodesPerRack int) *Cluster {
	if nodesPerRack <= 0 {
		nodesPerRack = nodes
	}
	c := &Cluster{Nodes: make([]Node, nodes)}
	for i := range c.Nodes {
		c.Nodes[i] = Node{ID: i, Rack: i / nodesPerRack, Total: gpusPerNode, Free: gpusPerNode}
	}
	return c
}

// Total is the number of GPUs in the cluster.
func (c *Cluster) Total() int {
	t := 0
	for _, n := range c.Nodes {
		t += n.Total
	}
	return t
}

// Free is the number of idle GPUs.
func (c *Cluster) Free() int {
	t := 0
	for _, n := range c.Nodes {
		t += n.Free
	}
	return t
}

// Fits reports whether a gang of nodes x per GPUs could ever run on an empty cluster.
func (c *Cluster) Fits(nodes, per int) bool {
	if nodes < 1 || per < 1 || nodes > len(c.Nodes) {
		return false
	}
	have := 0
	for _, n := range c.Nodes {
		if n.Total >= per {
			have++
		}
	}
	return have >= nodes
}

// Clone returns an independent copy, used for what-if planning.
func (c *Cluster) Clone() *Cluster {
	cp := &Cluster{Nodes: make([]Node, len(c.Nodes))}
	copy(cp.Nodes, c.Nodes)
	return cp
}

type cand struct{ idx, rack, free int }

// Place picks nodes for a gang of nodes x per GPUs without changing the cluster.
//
// Best fit (fullest feasible node first) keeps whole nodes free for large jobs. A multi-node gang
// prefers a single rack so its collective traffic stays under one switch; crossRack reports when
// that was impossible.
func (c *Cluster) Place(nodes, per int) (allocs []Alloc, crossRack bool, ok bool) {
	if nodes < 1 || per < 1 {
		return nil, false, false
	}
	var cs []cand
	for i, n := range c.Nodes {
		if n.Free >= per {
			cs = append(cs, cand{i, n.Rack, n.Free})
		}
	}
	if len(cs) < nodes {
		return nil, false, false
	}
	if nodes > 1 {
		byRack := map[int][]cand{}
		var racks []int
		for _, x := range cs {
			if _, seen := byRack[x.rack]; !seen {
				racks = append(racks, x.rack)
			}
			byRack[x.rack] = append(byRack[x.rack], x)
		}
		sort.Ints(racks)
		var best []cand
		bestLeft := 1 << 30
		for _, r := range racks {
			g := byRack[r]
			if len(g) < nodes {
				continue
			}
			pick, left := bestFit(g, nodes, per)
			if left < bestLeft {
				best, bestLeft = pick, left
			}
		}
		if best != nil {
			return toAllocs(best, per), false, true
		}
	}
	pick, _ := bestFit(cs, nodes, per)
	cross := false
	for _, p := range pick {
		if p.rack != pick[0].rack {
			cross = true
		}
	}
	return toAllocs(pick, per), cross, true
}

func bestFit(cs []cand, nodes, per int) ([]cand, int) {
	cp := append([]cand(nil), cs...)
	sort.Slice(cp, func(a, b int) bool {
		if cp[a].free != cp[b].free {
			return cp[a].free < cp[b].free
		}
		return cp[a].idx < cp[b].idx
	})
	pick := cp[:nodes]
	left := 0
	for _, p := range pick {
		left += p.free - per
	}
	return pick, left
}

func toAllocs(pick []cand, per int) []Alloc {
	out := make([]Alloc, len(pick))
	for i, p := range pick {
		out[i] = Alloc{Node: p.idx, GPUs: per}
	}
	sort.Slice(out, func(a, b int) bool { return out[a].Node < out[b].Node })
	return out
}

// Allocate reserves GPUs; it fails without changing anything if a node lacks capacity. Entries for
// the same node add up.
func (c *Cluster) Allocate(a []Alloc) error {
	need, err := c.sum(a)
	if err != nil {
		return err
	}
	for node, gpus := range need {
		if c.Nodes[node].Free < gpus {
			return fmt.Errorf("cannot allocate %d GPUs on node %d", gpus, node)
		}
	}
	for node, gpus := range need {
		c.Nodes[node].Free -= gpus
	}
	return nil
}

// Release returns GPUs; it fails without changing anything if that would exceed a node's capacity.
func (c *Cluster) Release(a []Alloc) error {
	give, err := c.sum(a)
	if err != nil {
		return err
	}
	for node, gpus := range give {
		if c.Nodes[node].Free+gpus > c.Nodes[node].Total {
			return fmt.Errorf("cannot release %d GPUs on node %d", gpus, node)
		}
	}
	for node, gpus := range give {
		c.Nodes[node].Free += gpus
	}
	return nil
}

// sum adds up the GPUs per node and rejects unknown nodes and non-positive counts.
func (c *Cluster) sum(a []Alloc) (map[int]int, error) {
	out := make(map[int]int, len(a))
	for _, x := range a {
		if x.Node < 0 || x.Node >= len(c.Nodes) || x.GPUs <= 0 {
			return nil, fmt.Errorf("invalid allocation of %d GPUs on node %d", x.GPUs, x.Node)
		}
		out[x.Node] += x.GPUs
	}
	return out, nil
}

// Check verifies the bookkeeping invariant 0 <= Free <= Total on every node.
func (c *Cluster) Check() error {
	for _, n := range c.Nodes {
		if n.Free < 0 || n.Free > n.Total {
			return fmt.Errorf("node %d: free=%d total=%d", n.ID, n.Free, n.Total)
		}
	}
	return nil
}
