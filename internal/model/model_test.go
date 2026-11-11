package model

import (
	"math/rand"
	"testing"
)

func TestNewClusterLayout(t *testing.T) {
	c := NewCluster(8, 8, 4)
	if c.Total() != 64 || c.Free() != 64 {
		t.Fatalf("total=%d free=%d, want 64 and 64", c.Total(), c.Free())
	}
	for i, n := range c.Nodes {
		if n.ID != i || n.Rack != i/4 {
			t.Fatalf("node %d: id=%d rack=%d", i, n.ID, n.Rack)
		}
	}
	if got := NewCluster(3, 8, 0).Nodes[2].Rack; got != 0 {
		t.Fatalf("with no rack size every node is in rack 0, got rack %d", got)
	}
}

func TestFits(t *testing.T) {
	c := NewCluster(4, 8, 2)
	cases := []struct {
		nodes, per int
		want       bool
	}{
		{1, 1, true}, {4, 8, true}, {5, 1, false}, {1, 9, false}, {0, 1, false}, {1, 0, false}, {-1, 4, false},
	}
	for _, tc := range cases {
		if got := c.Fits(tc.nodes, tc.per); got != tc.want {
			t.Errorf("Fits(%d,%d)=%v, want %v", tc.nodes, tc.per, got, tc.want)
		}
	}
}

func TestPlaceBestFitKeepsWholeNodesFree(t *testing.T) {
	c := NewCluster(3, 8, 3)
	mustAllocate(t, c, []Alloc{{Node: 1, GPUs: 6}}) // node 1 has 2 GPUs left
	al, _, ok := c.Place(1, 2)
	if !ok || len(al) != 1 || al[0].Node != 1 {
		t.Fatalf("a 2-GPU job should fill the nearly full node 1, got %v ok=%v", al, ok)
	}
	al, _, ok = c.Place(1, 3)
	if !ok || al[0].Node != 0 {
		t.Fatalf("a 3-GPU job does not fit node 1 and should take the lowest empty node, got %v", al)
	}
}

func TestPlacePrefersOneRack(t *testing.T) {
	c := NewCluster(4, 8, 2) // racks {0,1} and {2,3}
	mustAllocate(t, c, []Alloc{{Node: 0, GPUs: 8}})
	al, cross, ok := c.Place(2, 8)
	if !ok || cross {
		t.Fatalf("nodes 2 and 3 share a rack, got %v cross=%v ok=%v", al, cross, ok)
	}
	if al[0].Node != 2 || al[1].Node != 3 {
		t.Fatalf("want nodes 2 and 3, got %v", al)
	}
}

func TestPlaceReportsCrossRack(t *testing.T) {
	c := NewCluster(4, 8, 2)
	mustAllocate(t, c, []Alloc{{Node: 1, GPUs: 8}, {Node: 2, GPUs: 8}})
	al, cross, ok := c.Place(2, 8) // free nodes: 0 (rack 0) and 3 (rack 1)
	if !ok || !cross {
		t.Fatalf("the only free nodes are in different racks, got %v cross=%v ok=%v", al, cross, ok)
	}
}

func TestPlaceFailsWhenItCannotFit(t *testing.T) {
	c := NewCluster(2, 4, 2)
	mustAllocate(t, c, []Alloc{{Node: 0, GPUs: 1}})
	if _, _, ok := c.Place(2, 4); ok {
		t.Fatal("node 0 has only 3 free GPUs, a 2x4 gang must not fit")
	}
	if _, _, ok := c.Place(0, 1); ok {
		t.Fatal("a gang of zero nodes is not a job")
	}
	if _, _, ok := c.Place(1, 0); ok {
		t.Fatal("a gang of zero GPUs is not a job")
	}
}

func TestPlaceDoesNotChangeTheCluster(t *testing.T) {
	c := NewCluster(4, 8, 2)
	before := c.Clone()
	c.Place(3, 4)
	for i := range c.Nodes {
		if c.Nodes[i] != before.Nodes[i] {
			t.Fatalf("Place changed node %d", i)
		}
	}
}

func TestAllocateAndReleaseRoundTrip(t *testing.T) {
	c := NewCluster(2, 8, 2)
	a := []Alloc{{Node: 0, GPUs: 3}, {Node: 1, GPUs: 8}}
	mustAllocate(t, c, a)
	if c.Free() != 5 {
		t.Fatalf("free=%d, want 5", c.Free())
	}
	if err := c.Release(a); err != nil {
		t.Fatal(err)
	}
	if c.Free() != 16 {
		t.Fatalf("free=%d after release, want 16", c.Free())
	}
}

func TestAllocateIsAllOrNothing(t *testing.T) {
	c := NewCluster(2, 4, 2)
	err := c.Allocate([]Alloc{{Node: 0, GPUs: 4}, {Node: 1, GPUs: 5}})
	if err == nil {
		t.Fatal("node 1 has 4 GPUs, 5 must be refused")
	}
	if c.Free() != 8 {
		t.Fatalf("a refused allocation changed the cluster: free=%d", c.Free())
	}
}

func TestAllocateAddsUpEntriesForOneNode(t *testing.T) {
	c := NewCluster(1, 8, 1)
	err := c.Allocate([]Alloc{{Node: 0, GPUs: 5}, {Node: 0, GPUs: 5}})
	if err == nil {
		t.Fatal("two entries of 5 GPUs on an 8-GPU node must not slip through")
	}
	if c.Free() != 8 {
		t.Fatalf("free=%d, want 8", c.Free())
	}
	if err := c.Allocate([]Alloc{{Node: 0, GPUs: 5}, {Node: 0, GPUs: 3}}); err != nil || c.Free() != 0 {
		t.Fatalf("5+3 fits an 8-GPU node, err=%v free=%d", err, c.Free())
	}
}

func TestRejectsInvalidEntries(t *testing.T) {
	c := NewCluster(2, 4, 2)
	for _, a := range [][]Alloc{{{Node: -1, GPUs: 1}}, {{Node: 2, GPUs: 1}}, {{Node: 0, GPUs: 0}}, {{Node: 0, GPUs: -2}}} {
		if err := c.Allocate(a); err == nil {
			t.Errorf("Allocate(%v) should fail", a)
		}
		if err := c.Release(a); err == nil {
			t.Errorf("Release(%v) should fail", a)
		}
	}
	if err := c.Check(); err != nil || c.Free() != 8 {
		t.Fatalf("invalid calls changed the cluster: %v free=%d", err, c.Free())
	}
}

func TestReleaseCannotExceedCapacity(t *testing.T) {
	c := NewCluster(1, 4, 1)
	if err := c.Release([]Alloc{{Node: 0, GPUs: 1}}); err == nil {
		t.Fatal("releasing GPUs that were never allocated must fail")
	}
}

func TestCloneIsIndependent(t *testing.T) {
	c := NewCluster(2, 4, 2)
	cp := c.Clone()
	mustAllocate(t, cp, []Alloc{{Node: 0, GPUs: 4}})
	if c.Free() != 8 || cp.Free() != 4 {
		t.Fatalf("clone shares state: original %d, clone %d", c.Free(), cp.Free())
	}
}

func TestCheckCatchesCorruption(t *testing.T) {
	c := NewCluster(2, 4, 2)
	c.Nodes[1].Free = 5
	if c.Check() == nil {
		t.Fatal("free above total must be reported")
	}
	c.Nodes[1].Free = -1
	if c.Check() == nil {
		t.Fatal("negative free must be reported")
	}
}

// Random placements and releases never break the invariants, and Place always returns something
// Allocate accepts.
func TestRandomAllocationsKeepInvariants(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	c := NewCluster(8, 8, 4)
	var held [][]Alloc
	for step := 0; step < 20000; step++ {
		if rng.Intn(2) == 0 && len(held) > 0 {
			i := rng.Intn(len(held))
			if err := c.Release(held[i]); err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			held = append(held[:i], held[i+1:]...)
		} else {
			nodes, per := 1+rng.Intn(4), 1+rng.Intn(8)
			if al, _, ok := c.Place(nodes, per); ok {
				if len(al) != nodes {
					t.Fatalf("step %d: %d nodes asked, %d placed", step, nodes, len(al))
				}
				mustAllocate(t, c, al)
				held = append(held, al)
			}
		}
		if err := c.Check(); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		used := 0
		for _, h := range held {
			for _, a := range h {
				used += a.GPUs
			}
		}
		if c.Free() != c.Total()-used {
			t.Fatalf("step %d: free=%d, want %d", step, c.Free(), c.Total()-used)
		}
	}
}

func TestJobStateString(t *testing.T) {
	for s, want := range map[JobState]string{Pending: "pending", Running: "running", Done: "done", JobState(9): "unknown"} {
		if s.String() != want {
			t.Errorf("%d: %q, want %q", s, s.String(), want)
		}
	}
	if g := (&Job{Nodes: 3, GPUsPerNode: 8}).GPUs(); g != 24 {
		t.Errorf("GPUs()=%d, want 24", g)
	}
}

func mustAllocate(t *testing.T, c *Cluster, a []Alloc) {
	t.Helper()
	if err := c.Allocate(a); err != nil {
		t.Fatal(err)
	}
}
