#!/usr/bin/env python3
"""Hand-written mutation check: every mutant of the scheduler, the cluster model or the simulator
must make `go test ./...` fail. Run from anywhere: python3 scripts/mutate.py [mutant numbers...]

A surviving mutant means the tests do not pin that behaviour. Each mutant is applied to a private
copy of the tree, so the working tree is never touched.
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

MUTANTS = [
    ("a team within its quota can be preempted by another team", "internal/sched/preempt.go",
     "\tif left[v.Team] <= s.cfg.Quota[v.Team] {\n\t\treturn false\n\t}\n", ""),
    ("MinRun is ignored", "internal/sched/preempt.go",
     "!v.Preemptible || now-v.Start < s.cfg.MinRun", "!v.Preemptible"),
    ("the victim set is not reduced to what the job needs", "internal/sched/preempt.go",
     "for _, v := range s.minimal(head, chosen) {", "for _, v := range chosen {"),
    ("victims are not ordered by the work they lose", "internal/sched/preempt.go",
     "\t\tif lost[va.ID] != lost[vb.ID] {\n\t\t\treturn lost[va.ID] < lost[vb.ID]\n\t\t}\n", ""),
    ("aging is off", "internal/sched/sched.go",
     "aged: s.cfg.MaxWait > 0 && j.Nodes > 1 && now-j.Queued >= s.cfg.MaxWait,", "aged: false,"),
    ("backfill ignores the head's reservation", "internal/sched/sched.go",
     "if now+j.Remaining <= shadow+1e-9 && s.tryStart(j, now) {", "if s.tryStart(j, now) {"),
    ("the restart cost is not charged", "internal/sched/preempt.go",
     "v.Remaining += s.cfg.RestartCost - saved", "v.Remaining -= saved"),
    ("a checkpoint is credited before it is taken", "internal/sched/preempt.go",
     "return math.Floor((now-v.Start)/v.CheckpointEvery) * v.CheckpointEvery",
     "return math.Ceil((now-v.Start)/v.CheckpointEvery) * v.CheckpointEvery"),
    ("a job of equal priority preempts a job of its own team", "internal/sched/preempt.go",
     "\t\treturn v.Priority < head.Priority\n\t}\n\tif left", "\t\treturn v.Priority <= head.Priority\n\t}\n\tif left"),
    ("a multi-node gang ignores racks", "internal/model/model.go",
     "\tif nodes > 1 {\n\t\tbyRack", "\tif false {\n\t\tbyRack"),
    ("utilization is not limited to the arrival window", "internal/sim/sim.go",
     "r.Utilization = c.busyArriving /", "r.Utilization = c.busy /"),
    ("completing a job does not free its GPUs", "internal/sched/sched.go",
     "\tif err := s.cl.Release(r.Placement); err != nil {\n\t\treturn err\n\t}\n\tdelete(s.running, id)",
     "\tdelete(s.running, id)"),
    ("only the head may start a preemption", "internal/sched/sched.go",
     "\t\t\tif i > 0 && run[j.Team]+j.GPUs() > s.cfg.Quota[j.Team] {", "\t\t\tif i > 0 && run[j.Team] >= 0 {"),
    ("any queued job may start a preemption, even above its quota", "internal/sched/sched.go",
     "\t\t\tif i > 0 && run[j.Team]+j.GPUs() > s.cfg.Quota[j.Team] {", "\t\t\tif i > 0 && run[j.Team]+j.GPUs() > 1<<30 {"),
    ("two allocations on one node overwrite each other", "internal/model/model.go",
     "\t\tout[x.Node] += x.GPUs\n", "\t\tout[x.Node] = x.GPUs\n"),
    ("the queue order ignores the team's quota", "internal/sched/sched.go",
     "\t\tif x.over != y.over {\n\t\t\treturn !x.over\n\t\t}\n", ""),
    ("the queue order ignores past usage", "internal/sched/sched.go",
     "\t\tif x.norm != y.norm {\n\t\t\treturn x.norm < y.norm\n\t\t}\n", ""),
]


def run(rel, old, new):
    with tempfile.TemporaryDirectory(prefix="gpuq-mutant-") as tmp:
        dst = pathlib.Path(tmp) / "gpuq"
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git"))
        path = dst / rel
        text = path.read_text()
        if text.count(old) != 1:
            return "BAD-ANCHOR", f"{text.count(old)} matches in {rel}"
        path.write_text(text.replace(old, new))
        try:
            r = subprocess.run(["go", "test", "-count=1", "./..."], cwd=dst, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            return "killed", "timeout"
        if r.returncode == 0:
            return "SURVIVED", ""
        failed = [line.split()[2] for line in r.stdout.splitlines() if line.startswith("--- FAIL")]
        return "killed", ", ".join(failed[:2]) or "does not compile"


def main():
    only = set(sys.argv[1:])
    bad = 0
    for i, (name, rel, old, new) in enumerate(MUTANTS):
        if only and str(i) not in only:
            continue
        status, why = run(rel, old, new)
        print(f"{i:2d} {status:10s} {name}  [{why}]", flush=True)
        bad += status != "killed"
    print(f"{len(only) or len(MUTANTS)} mutants, {bad} not killed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
