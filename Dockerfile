# syntax=docker/dockerfile:1
FROM golang:1.24 AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gpuqd ./cmd/gpuqd

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/gpuqd /gpuqd
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/gpuqd"]
