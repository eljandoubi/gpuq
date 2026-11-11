# Deploying and mapping to Kubernetes

Two sets of manifests, checked against the Kubernetes and Kueue schemas. Neither was applied to a
live cluster: no cluster was available.

## `kubernetes/queues.yaml` and `example-workloads.yaml`: gpuq's quotas as Kueue objects

The default teams of the simulator (research 24, training 32, eval 8 GPUs) as
[Kueue](https://kueue.sigs.k8s.io/) `ClusterQueue`s in one cohort, with a `LocalQueue` per team
namespace, three `WorkloadPriorityClass`es, and two example `Job`s (a 1-GPU eval job and a 4 x 8-GPU
gang).

| gpuq | Kueue (`kueue.x-k8s.io/v1beta1`) | how close |
|---|---|---|
| team quota | `ClusterQueue.spec.resourceGroups[].flavors[].resources[].nominalQuota` | same idea |
| borrowing idle GPUs | queues in one `cohort`, no `borrowingLimit` | same idea |
| reclaim by preemption (rule 3 and 4a) | `preemption.reclaimWithinCohort: Any` | Kueue preempts workloads of queues that borrow, as gpuq does |
| higher priority takes from a borrower (rule 4b) | `preemption.borrowWithinCohort.policy: LowerPriority` | closest equivalent, not identical |
| higher priority inside a team (rule 2) | `preemption.withinClusterQueue: LowerPriority` | same idea |
| gang | a `Job` is one `Workload`: all its pods fit the quota or none starts | same idea |
| `fifo` / `backfill` | `queueingStrategy: StrictFIFO` / `BestEffortFIFO` | `BestEffortFIFO` lets later workloads start but gives the head **no reservation**: it is not EASY backfill |
| aging (`MaxWait`) | none | the `v1beta1` schema has no equivalent: under `BestEffortFIFO` a big gang can starve behind smaller workloads, the problem aging solves in gpuq |
| `MinRun` | none | no field for a minimum run time before preemption in `ClusterQueue.preemption` |
| checkpoint and restart cost | the application | Kueue requeues the Job; the job must checkpoint itself |
| rack-aware placement | `ResourceFlavor.spec.topologyName` (topology-aware scheduling) | not used here |

Kueue also has a fair-sharing feature (`ClusterQueue.spec.fairSharing`) that this mapping does not
use. The table lists what the simulator showed to matter (isolation by quota, reclaim, aging); check
Kueue's documentation for your version before relying on a field.

## `kubernetes/gpuqd.yaml`: the control plane

A `Deployment` (one replica: the daemon keeps its state in memory), `Service`, `ServiceAccount` and a
`restricted` Pod Security namespace. Non-root, read-only root filesystem, no capabilities, resource
limits, `/healthz` probes. The pod carries a Datadog Agent Autodiscovery annotation
(`ad.datadoghq.com/gpuqd.checks`) that scrapes `/metrics` with the OpenMetrics check; the annotation
follows the format in Datadog's integration documentation and was not run against an Agent.

## How it was checked

```sh
go install github.com/yannh/kubeconform/cmd/kubeconform@latest
kubeconform -strict -summary -kubernetes-version 1.30.0 \
  -schema-location default \
  -schema-location 'https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json' \
  deploy/kubernetes/
# Summary: 19 resources found in 3 files - Valid: 19, Invalid: 0, Errors: 0, Skipped: 0
```

(`make k8s-validate` runs the same command.) The binary was also run with the manifest's arguments:
`/healthz`, job submission, a 422 for a job larger than the cluster, `/metrics`, and a clean exit on
SIGINT.

**Not verified here:** applying the manifests to a cluster, and building the `Dockerfile` (Docker
Desktop did not start on the machine used). The Dockerfile's build step was run directly:
`CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" ./cmd/gpuqd` gives a 6.7 MB static
binary.
