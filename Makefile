.PHONY: all build test race bench lint sim sweep mutate

all: lint race

build:
	go build ./...

test:
	go test -count=1 ./...

race:
	go test -race -count=1 -cover ./...

bench:
	go test -run xxx -bench . -benchmem ./internal/sched ./internal/sim

lint:
	@test -z "$$(gofmt -l .)" || (gofmt -l . && echo "gofmt: the files above need formatting" && exit 1)
	go vet ./...

# The comparison reported in the README (10 seeds, offered load 0.85).
sim:
	go run ./cmd/gpuq-sim

sweep:
	go run ./cmd/gpuq-sim -sweep

# Every hand-written mutant of the scheduler must be caught by the tests.
mutate:
	python3 scripts/mutate.py
