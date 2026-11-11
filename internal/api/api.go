// Package api exposes the scheduler over HTTP and Prometheus.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
)

// Server is an in-memory control plane: jobs "run" for their declared runtime on a clock that the
// caller supplies, so the whole service can be exercised without GPUs.
type Server struct {
	mu    sync.Mutex
	s     *sched.Scheduler
	jobs  map[int]*model.Job
	next  int
	clock func() float64
}

// New wraps a scheduler. clock returns the current time in seconds.
func New(s *sched.Scheduler, clock func() float64) *Server {
	return &Server{s: s, jobs: map[int]*model.Job{}, next: 1, clock: clock}
}

// Handler returns the HTTP routes.
func (a *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/jobs", a.submit)
	mux.HandleFunc("GET /v1/jobs", a.list)
	mux.HandleFunc("GET /v1/jobs/{id}", a.get)
	mux.HandleFunc("GET /v1/cluster", a.cluster)
	mux.HandleFunc("GET /metrics", a.metrics)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "ok\n")
	})
	return mux
}

// Advance completes the jobs that are due at now and schedules what fits.
func (a *Server) Advance(now float64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.advanceLocked(now)
}

// Run advances the clock every tick until ctx is cancelled.
func (a *Server) Run(ctx context.Context, every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			a.Advance(a.clock())
		}
	}
}

func (a *Server) advanceLocked(now float64) {
	for {
		var due *model.Job
		var dueEnd float64
		for _, r := range a.s.RunningJobs() {
			end := r.Start + r.Remaining
			if end <= now && (due == nil || end < dueEnd || (end == dueEnd && r.ID < due.ID)) {
				due, dueEnd = r, end
			}
		}
		if due == nil {
			break
		}
		if err := a.s.Complete(due.ID, dueEnd); err != nil {
			break
		}
		a.s.Tick(dueEnd)
	}
	a.s.Tick(now)
	a.s.Drain()
}

type submitReq struct {
	Team             string  `json:"team"`
	Priority         int     `json:"priority"`
	Nodes            int     `json:"nodes"`
	GPUsPerNode      int     `json:"gpus_per_node"`
	RuntimeS         float64 `json:"runtime_s"`
	Preemptible      bool    `json:"preemptible"`
	CheckpointEveryS float64 `json:"checkpoint_every_s"`
}

type jobView struct {
	ID          int      `json:"id"`
	Team        string   `json:"team"`
	State       string   `json:"state"`
	Priority    int      `json:"priority"`
	Nodes       int      `json:"nodes"`
	GPUsPerNode int      `json:"gpus_per_node"`
	Preemptible bool     `json:"preemptible"`
	Preemptions int      `json:"preemptions"`
	WaitS       *float64 `json:"wait_s,omitempty"`
}

func view(j *model.Job) jobView {
	v := jobView{ID: j.ID, Team: j.Team, State: j.State.String(), Priority: j.Priority, Nodes: j.Nodes,
		GPUsPerNode: j.GPUsPerNode, Preemptible: j.Preemptible, Preemptions: j.Preemptions}
	if j.FirstStart >= 0 {
		w := j.FirstStart - j.Submit
		v.WaitS = &w
	}
	return v
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func fail(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]string{"error": msg})
}

func (a *Server) submit(w http.ResponseWriter, r *http.Request) {
	var in submitReq
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		fail(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		fail(w, http.StatusBadRequest, "invalid JSON: unexpected data after the object")
		return
	}
	switch {
	case strings.TrimSpace(in.Team) == "":
		fail(w, http.StatusBadRequest, "team is required")
		return
	case in.Nodes < 1 || in.GPUsPerNode < 1:
		fail(w, http.StatusBadRequest, "nodes and gpus_per_node must be at least 1")
		return
	case in.RuntimeS <= 0:
		fail(w, http.StatusBadRequest, "runtime_s must be positive")
		return
	case in.CheckpointEveryS < 0:
		fail(w, http.StatusBadRequest, "checkpoint_every_s cannot be negative")
		return
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	now := a.clock()
	a.advanceLocked(now)
	j := &model.Job{ID: a.next, Team: in.Team, Priority: in.Priority, Nodes: in.Nodes, GPUsPerNode: in.GPUsPerNode,
		Runtime: in.RuntimeS, Submit: now, Preemptible: in.Preemptible, CheckpointEvery: in.CheckpointEveryS}
	if err := a.s.Submit(j); err != nil {
		fail(w, http.StatusUnprocessableEntity, err.Error())
		return
	}
	a.next++
	a.jobs[j.ID] = j
	a.s.Tick(now)
	a.s.Drain()
	writeJSON(w, http.StatusCreated, view(j))
}

func (a *Server) list(w http.ResponseWriter, r *http.Request) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.advanceLocked(a.clock())
	want := r.URL.Query().Get("state")
	out := make([]jobView, 0, len(a.jobs))
	for _, j := range a.jobs {
		if want == "" || j.State.String() == want {
			out = append(out, view(j))
		}
	}
	sort.Slice(out, func(i, k int) bool { return out[i].ID < out[k].ID })
	writeJSON(w, http.StatusOK, out)
}

func (a *Server) get(w http.ResponseWriter, r *http.Request) {
	id, err := strconv.Atoi(r.PathValue("id"))
	if err != nil {
		fail(w, http.StatusBadRequest, "job id must be an integer")
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.advanceLocked(a.clock())
	j, ok := a.jobs[id]
	if !ok {
		fail(w, http.StatusNotFound, fmt.Sprintf("job %d not found", id))
		return
	}
	writeJSON(w, http.StatusOK, view(j))
}

func (a *Server) cluster(w http.ResponseWriter, r *http.Request) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.advanceLocked(a.clock())
	cl := a.s.Cluster()
	type node struct {
		ID   int `json:"id"`
		Rack int `json:"rack"`
		Free int `json:"free"`
	}
	nodes := make([]node, len(cl.Nodes))
	for i, n := range cl.Nodes {
		nodes[i] = node{n.ID, n.Rack, n.Free}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"gpus_total": cl.Total(), "gpus_free": cl.Free(),
		"jobs_pending": a.s.Pending(), "jobs_running": a.s.Running(), "nodes": nodes,
	})
}

func (a *Server) metrics(w http.ResponseWriter, r *http.Request) {
	a.mu.Lock()
	a.advanceLocked(a.clock())
	cl := a.s.Cluster()
	st := a.s.Stats
	byTeam := a.s.RunningByTeam()
	total, free := cl.Total(), cl.Free()
	pending, running := a.s.Pending(), a.s.Running()
	a.mu.Unlock()

	var b strings.Builder
	metric := func(name, kind, help string, v float64) {
		fmt.Fprintf(&b, "# HELP %s %s\n# TYPE %s %s\n%s %g\n", name, help, name, kind, name, v)
	}
	metric("gpuq_gpus_total", "gauge", "GPUs in the cluster.", float64(total))
	metric("gpuq_gpus_free", "gauge", "Idle GPUs.", float64(free))
	metric("gpuq_jobs_pending", "gauge", "Jobs waiting in the queue.", float64(pending))
	metric("gpuq_jobs_running", "gauge", "Jobs holding GPUs.", float64(running))
	metric("gpuq_job_starts_total", "counter", "Job starts, restarts after preemption included.", float64(st.Starts))
	metric("gpuq_preemptions_total", "counter", "Jobs preempted to give GPUs to another job.", float64(st.Preemptions))
	metric("gpuq_cross_rack_starts_total", "counter", "Multi-node gangs placed across racks.", float64(st.CrossRackStarts))
	teams := make([]string, 0, len(byTeam))
	for t := range byTeam {
		teams = append(teams, t)
	}
	sort.Strings(teams)
	b.WriteString("# HELP gpuq_team_gpus_running GPUs each team is running on.\n# TYPE gpuq_team_gpus_running gauge\n")
	for _, t := range teams {
		fmt.Fprintf(&b, "gpuq_team_gpus_running{team=%q} %d\n", t, byTeam[t])
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = io.WriteString(w, b.String())
}
