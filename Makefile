# Development tasks. Everything here is optional: "go test ./..." works on its
# own, and skips the tests whose data is not present.

GO        ?= go
LID176    := lid176/model/lid.176.bin.0 lid176/model/lid.176.bin.1 lid176/model/lid.176.bin.2
PLATFORMS := linux/amd64 linux/arm64 linux/386 linux/arm linux/riscv64 linux/ppc64le linux/s390x \
             linux/loong64 darwin/amd64 darwin/arm64 windows/amd64 windows/arm64 \
             freebsd/amd64 openbsd/amd64 netbsd/amd64 js/wasm wasip1/wasm

export CGO_ENABLED = 0

.PHONY: all test test-simd golden race bench lint vet nocgo testdata accuracy accuracy-docker clean help

all: vet test ## vet and run the quick tests

help: ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

go.work:
	$(GO) work init . ./lid176 ./accuracy

test: go.work ## quick tests of both modules (-short)
	$(GO) test -short ./... ./lid176/...

test-simd: go.work ## quick tests with the SIMD kernels (Go 1.27+)
	GOEXPERIMENT=simd $(GO) test -short ./... ./lid176/...

golden: go.work testdata ## every test, including all golden dumps and the exhaustive libm check
	$(GO) test -count=1 ./... ./lid176/...
	GOEXPERIMENT=simd $(GO) test -count=1 ./...

race: go.work ## quick tests under the race detector
	CGO_ENABLED=1 $(GO) test -race -short ./... ./lid176/...

bench: go.work ## Go benchmarks (run "make testdata" first for every model)
	$(GO) test -run='^$$' -bench=. -benchmem ./... ./lid176/...

vet: go.work ## go vet and gofmt
	$(GO) vet ./... ./lid176/... ./accuracy/...
	@test -z "$$(gofmt -l . | tee /dev/stderr)"

lint: vet ## golangci-lint and govulncheck
	golangci-lint run ./... ./lid176/...
	$(GO) run golang.org/x/vuln/cmd/govulncheck@latest ./... ./lid176/...

nocgo: go.work ## cross-compile every package with CGO_ENABLED=0 for many platforms
	@for p in $(PLATFORMS); do \
		echo "GOOS=$${p%/*} GOARCH=$${p#*/}"; \
		GOOS=$${p%/*} GOARCH=$${p#*/} $(GO) build ./... ./lid176/... || exit 1; \
	done
	@test -z "$$($(GO) list -f '{{if .CgoFiles}}{{.ImportPath}}{{end}}' ./... ./lid176/...)" || \
		(echo "packages with cgo files found" && exit 1)
	@echo "OK: no cgo anywhere"

testdata: accuracy/models/lid.176.bin ## fetch the Git LFS test files and assemble lid.176.bin
	@if command -v git-lfs >/dev/null 2>&1; then git lfs pull; else echo "git-lfs not installed: large test models stay unavailable"; fi

accuracy/models/lid.176.bin: $(LID176)
	cat $(LID176) > $@
	cd accuracy/models && (sha256sum -c --ignore-missing SHA256SUMS 2>/dev/null || shasum -a 256 -c --ignore-missing SHA256SUMS)

accuracy: testdata ## compare with C++ fastText bit for bit (Linux; see accuracy-docker)
	accuracy/verify.sh

accuracy-docker: testdata ## the same, inside the reference environment (needs docker)
	docker build -t fasttext-reference accuracy/reference
	docker run --rm -v "$(CURDIR):/src" -w /src fasttext-reference accuracy/verify.sh

clean: ## remove build output
	rm -rf accuracy/build accuracy/models/lid.176.bin
