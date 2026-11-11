package api_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/eljandoubi/gpuq/internal/api"
	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
)

type fakeClock struct {
	mu  sync.Mutex
	now float64
}

func (c *fakeClock) Now() float64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) Set(v float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = v
}

// newServer is a 2-node, 8-GPU cluster with two teams that each have half of it.
func newServer(t *testing.T) (*httptest.Server, *fakeClock, *api.Server) {
	t.Helper()
	clk := &fakeClock{}
	cfg := sched.Config{Kind: sched.FairShare, Quota: map[string]int{"a": 4, "b": 4}, Preempt: true}
	srv := api.New(sched.New(cfg, model.NewCluster(2, 4, 2)), clk.Now)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts, clk, srv
}

func do(t *testing.T, method, url, body string) (int, map[string]any, string) {
	t.Helper()
	req, err := http.NewRequest(method, url, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	var out map[string]any
	_ = json.Unmarshal(raw, &out)
	return resp.StatusCode, out, string(raw)
}

func submit(t *testing.T, ts *httptest.Server, team string, nodes, per int, runtime float64, extra string) map[string]any {
	t.Helper()
	body := fmt.Sprintf(`{"team":%q,"nodes":%d,"gpus_per_node":%d,"runtime_s":%v%s}`, team, nodes, per, runtime, extra)
	code, out, raw := do(t, "POST", ts.URL+"/v1/jobs", body)
	if code != http.StatusCreated {
		t.Fatalf("submit %s: status %d: %s", body, code, raw)
	}
	return out
}

func TestSubmitStartsAJobWhenThereIsRoom(t *testing.T) {
	ts, _, _ := newServer(t)
	j := submit(t, ts, "a", 1, 4, 100, "")
	if j["id"] != 1.0 || j["state"] != "running" || j["wait_s"] != 0.0 {
		t.Fatalf("got %v, want job 1 running with no wait", j)
	}
}

func TestAQueuedJobStartsWhenTimePasses(t *testing.T) {
	ts, clk, _ := newServer(t)
	submit(t, ts, "a", 2, 4, 100, "")
	j := submit(t, ts, "b", 1, 4, 10, "")
	if j["state"] != "pending" || j["wait_s"] != nil {
		t.Fatalf("got %v, want pending without a wait yet", j)
	}
	clk.Set(100)
	code, got, _ := do(t, "GET", ts.URL+"/v1/jobs/2", "")
	if code != 200 || got["state"] != "running" || got["wait_s"] != 100.0 {
		t.Fatalf("got %d %v, want job 2 running after waiting 100 s", code, got)
	}
	clk.Set(110)
	_, got, _ = do(t, "GET", ts.URL+"/v1/jobs/2", "")
	if got["state"] != "done" {
		t.Fatalf("job 2 is %v at 110 s, want done", got["state"])
	}
}

func TestSubmitValidation(t *testing.T) {
	ts, _, _ := newServer(t)
	for name, tc := range map[string]struct {
		body string
		want int
	}{
		"invalid json":      {`{`, 400},
		"unknown field":     {`{"team":"a","nodes":1,"gpus_per_node":1,"runtime_s":1,"gpu":3}`, 400},
		"missing team":      {`{"nodes":1,"gpus_per_node":1,"runtime_s":1}`, 400},
		"blank team":        {`{"team":"  ","nodes":1,"gpus_per_node":1,"runtime_s":1}`, 400},
		"no nodes":          {`{"team":"a","nodes":0,"gpus_per_node":1,"runtime_s":1}`, 400},
		"no gpus":           {`{"team":"a","nodes":1,"gpus_per_node":0,"runtime_s":1}`, 400},
		"no runtime":        {`{"team":"a","nodes":1,"gpus_per_node":1,"runtime_s":0}`, 400},
		"negative runtime":  {`{"team":"a","nodes":1,"gpus_per_node":1,"runtime_s":-5}`, 400},
		"negative interval": {`{"team":"a","nodes":1,"gpus_per_node":1,"runtime_s":1,"checkpoint_every_s":-1}`, 400},
		"too many nodes":    {`{"team":"a","nodes":3,"gpus_per_node":1,"runtime_s":1}`, 422},
		"too many gpus":     {`{"team":"a","nodes":1,"gpus_per_node":5,"runtime_s":1}`, 422},
		"body over one MiB": {`{"team":"` + strings.Repeat("x", 1<<20) + `"}`, 400},
		"valid":             {`{"team":"a","nodes":1,"gpus_per_node":1,"runtime_s":1}`, 201},
		"trailing data":     {`{"team":"a","nodes":1,"gpus_per_node":1,"runtime_s":1} {}`, 400},
	} {
		code, out, raw := do(t, "POST", ts.URL+"/v1/jobs", tc.body)
		if code != tc.want {
			t.Errorf("%s: status %d, want %d (%.80s)", name, code, tc.want, raw)
		}
		if tc.want >= 400 && out["error"] == nil {
			t.Errorf("%s: no error message in %.80s", name, raw)
		}
	}
	_, list, raw := do(t, "GET", ts.URL+"/v1/jobs", "")
	_ = list
	if got := strings.Count(raw, `"id"`); got != 1 {
		t.Errorf("%d jobs were created by the requests above, want only the valid one", got)
	}
}

func TestGetRejectsBadAndUnknownIDs(t *testing.T) {
	ts, _, _ := newServer(t)
	if code, _, _ := do(t, "GET", ts.URL+"/v1/jobs/7", ""); code != 404 {
		t.Errorf("unknown id: %d, want 404", code)
	}
	if code, _, _ := do(t, "GET", ts.URL+"/v1/jobs/abc", ""); code != 400 {
		t.Errorf("non-numeric id: %d, want 400", code)
	}
	if code, _, _ := do(t, "DELETE", ts.URL+"/v1/jobs", ""); code != 405 {
		t.Errorf("DELETE: %d, want 405", code)
	}
}

func TestListFiltersByState(t *testing.T) {
	ts, _, _ := newServer(t)
	submit(t, ts, "a", 2, 4, 100, "")
	submit(t, ts, "b", 1, 4, 100, "")
	count := func(query string) int {
		resp, err := http.Get(ts.URL + "/v1/jobs" + query)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var jobs []map[string]any
		if err := json.NewDecoder(resp.Body).Decode(&jobs); err != nil {
			t.Fatal(err)
		}
		return len(jobs)
	}
	if count("") != 2 || count("?state=running") != 1 || count("?state=pending") != 1 || count("?state=done") != 0 {
		t.Fatalf("all=%d running=%d pending=%d done=%d, want 2 1 1 0",
			count(""), count("?state=running"), count("?state=pending"), count("?state=done"))
	}
}

func TestClusterEndpoint(t *testing.T) {
	ts, _, _ := newServer(t)
	submit(t, ts, "a", 1, 4, 100, "")
	_, c, _ := do(t, "GET", ts.URL+"/v1/cluster", "")
	nodes, _ := c["nodes"].([]any)
	if c["gpus_total"] != 8.0 || c["gpus_free"] != 4.0 || c["jobs_running"] != 1.0 || c["jobs_pending"] != 0.0 || len(nodes) != 2 {
		t.Fatalf("cluster = %v", c)
	}
}

func TestHealthz(t *testing.T) {
	ts, _, _ := newServer(t)
	resp, err := http.Get(ts.URL + "/healthz")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 || strings.TrimSpace(string(raw)) != "ok" {
		t.Fatalf("healthz = %d %q", resp.StatusCode, raw)
	}
}

func TestMetrics(t *testing.T) {
	ts, _, _ := newServer(t)
	submit(t, ts, "a", 1, 4, 100, "")
	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	text := string(raw)
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Errorf("content type %q", ct)
	}
	for _, want := range []string{
		"# TYPE gpuq_gpus_total gauge", "gpuq_gpus_total 8\n", "gpuq_gpus_free 4\n", "gpuq_jobs_running 1\n",
		"# TYPE gpuq_preemptions_total counter", "gpuq_job_starts_total 1\n", `gpuq_team_gpus_running{team="a"} 4`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("metrics lack %q", want)
		}
	}
}

func TestPreemptionShowsInTheAPI(t *testing.T) {
	ts, clk, _ := newServer(t)
	submit(t, ts, "b", 1, 4, 5000, `,"preemptible":true,"checkpoint_every_s":600`)
	submit(t, ts, "b", 1, 4, 5000, `,"preemptible":true,"checkpoint_every_s":600`)
	clk.Set(1000)
	a := submit(t, ts, "a", 1, 4, 100, "")
	if a["state"] != "running" {
		t.Fatalf("team a is within its quota and should have reclaimed a node, got %v", a)
	}
	_, victim, _ := do(t, "GET", ts.URL+"/v1/jobs/2", "")
	if victim["state"] != "pending" || victim["preemptions"] != 1.0 {
		t.Fatalf("job 2 = %v, want pending after one preemption", victim)
	}
	_, _, text := do(t, "GET", ts.URL+"/metrics", "")
	if !strings.Contains(text, "gpuq_preemptions_total 1\n") {
		t.Errorf("metrics do not count the preemption:\n%s", text)
	}
}

func TestAdvanceCompletesDueJobsInOrder(t *testing.T) {
	ts, clk, srv := newServer(t)
	submit(t, ts, "a", 2, 4, 50, "")
	submit(t, ts, "b", 2, 4, 30, "")
	submit(t, ts, "a", 1, 4, 10, "")
	clk.Set(1000)
	srv.Advance(1000)
	_, jobs, raw := do(t, "GET", ts.URL+"/v1/jobs?state=done", "")
	_ = jobs
	if strings.Count(raw, `"state":"done"`) != 3 {
		t.Fatalf("all three jobs should be done by 1000 s: %s", raw)
	}
	// When job 1 ends at 50, job 3 fits team a's quota and job 2 (8 GPUs for a team with 4) does
	// not, so job 3 goes first. Job 2 needs both nodes and starts when job 3 ends at 60. The jobs
	// start when they are due, not when Advance is called.
	for id, wait := range map[int]float64{3: 50, 2: 60} {
		_, j, _ := do(t, "GET", fmt.Sprintf("%s/v1/jobs/%d", ts.URL, id), "")
		if j["wait_s"] != wait {
			t.Fatalf("job %d waited %v s, want %v", id, j["wait_s"], wait)
		}
	}
}

func TestConcurrentRequests(t *testing.T) {
	ts, clk, _ := newServer(t)
	const workers, each = 8, 25
	ids := make(chan int, workers*each)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				clk.Set(float64(w*each + i))
				team := []string{"a", "b"}[(w+i)%2]
				body := fmt.Sprintf(`{"team":%q,"nodes":1,"gpus_per_node":%d,"runtime_s":5,"preemptible":true}`, team, 1+i%4)
				resp, err := http.Post(ts.URL+"/v1/jobs", "application/json", strings.NewReader(body))
				if err != nil {
					t.Error(err)
					return
				}
				var j map[string]any
				_ = json.NewDecoder(resp.Body).Decode(&j)
				resp.Body.Close()
				if resp.StatusCode != http.StatusCreated {
					t.Errorf("status %d: %v", resp.StatusCode, j)
					return
				}
				ids <- int(j["id"].(float64))
				for _, path := range []string{"/v1/cluster", "/metrics", "/v1/jobs"} {
					if r, err := http.Get(ts.URL + path); err == nil {
						io.Copy(io.Discard, r.Body)
						r.Body.Close()
					}
				}
			}
		}(w)
	}
	wg.Wait()
	close(ids)
	seen := map[int]bool{}
	for id := range ids {
		if seen[id] {
			t.Fatalf("job id %d handed out twice", id)
		}
		seen[id] = true
	}
	if len(seen) != workers*each {
		t.Fatalf("%d distinct ids, want %d", len(seen), workers*each)
	}
}
