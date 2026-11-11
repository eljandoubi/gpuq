// Command gpuqd serves the gpuq scheduler over HTTP with Prometheus metrics.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	"github.com/eljandoubi/gpuq/internal/api"
	"github.com/eljandoubi/gpuq/internal/model"
	"github.com/eljandoubi/gpuq/internal/sched"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	nodes := flag.Int("nodes", 8, "nodes in the cluster")
	gpus := flag.Int("gpus", 8, "GPUs per node")
	rack := flag.Int("rack", 4, "nodes per rack")
	policy := flag.String("policy", "fairshare", "fifo, backfill or fairshare")
	quotas := flag.String("quota", "", "guaranteed GPUs per team, e.g. research=24,training=32,eval=8")
	speed := flag.Float64("speed", 60, "simulated seconds per wall-clock second")
	flag.Parse()

	quota, err := parseQuota(*quotas)
	if err != nil {
		log.Fatal(err)
	}
	cfg := sched.Config{Quota: quota, Preempt: true}
	switch *policy {
	case "fifo":
		cfg.Kind = sched.FIFO
	case "backfill":
		cfg.Kind = sched.Backfill
	case "fairshare":
		cfg.Kind = sched.FairShare
	default:
		log.Fatalf("unknown policy %q", *policy)
	}

	begin := time.Now()
	clock := func() float64 { return time.Since(begin).Seconds() * *speed }
	srv := api.New(sched.New(cfg, model.NewCluster(*nodes, *gpus, *rack)), clock)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go srv.Run(ctx, 200*time.Millisecond)

	hs := &http.Server{Addr: *addr, Handler: srv.Handler(), ReadHeaderTimeout: 5 * time.Second}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		_ = hs.Shutdown(shutdown)
	}()
	log.Printf("gpuqd: %d nodes x %d GPUs, policy %s, listening on %s", *nodes, *gpus, *policy, *addr)
	if err := hs.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}

func parseQuota(s string) (map[string]int, error) {
	out := map[string]int{}
	if strings.TrimSpace(s) == "" {
		return out, nil
	}
	for _, part := range strings.Split(s, ",") {
		name, num, ok := strings.Cut(strings.TrimSpace(part), "=")
		n, err := strconv.Atoi(num)
		if !ok || name == "" || err != nil || n < 0 {
			return nil, fmt.Errorf("bad quota %q, want team=GPUs", part)
		}
		out[name] = n
	}
	return out, nil
}
