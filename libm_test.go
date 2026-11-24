package fasttext

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/slavabobik/fasttext/internal/testdata"
)

// accuracyDir returns the directory with the reference material: models,
// corpora, golden files and results.
func accuracyDir() string { return testdata.Dir() }

// TestLibmSanity is a quick check that runs with -short too: on a spread of
// ordinary inputs the ports must agree with Go's math package, an independent
// implementation, after rounding to float32. The check against glibc itself
// is TestLibmExhaustive.
func TestLibmSanity(t *testing.T) {
	for x := float32(-80); x < 80; x += 0.37 {
		if got, want := expf(x), float32(math.Exp(float64(x))); got != want {
			t.Errorf("expf(%v) = %v, want %v", x, got, want)
		}
		if got, want := float32(exp64(float64(x))), float32(math.Exp(float64(x))); got != want {
			t.Errorf("exp64(%v) = %v, want %v", x, got, want)
		}
	}
	for x := float32(0); x < 1.5; x += 0.0013 {
		if got, want := stdLog(x), float32(math.Log(float64(x)+1e-5)); got != want {
			t.Errorf("stdLog(%v) = %v, want %v", x, got, want)
		}
	}
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	if v := expf(-inf); v != 0 {
		t.Errorf("expf(-Inf) = %v", v)
	}
	if v := expf(inf); v != inf {
		t.Errorf("expf(+Inf) = %v", v)
	}
	if v := expf(89); v != inf {
		t.Errorf("expf(89) = %v", v)
	}
	if v := expf(-104); v != 0 {
		t.Errorf("expf(-104) = %v", v)
	}
	if v := expf(nan); v == v {
		t.Errorf("expf(NaN) = %v", v)
	}
	if v := stdLog(-1); v == v {
		t.Errorf("stdLog(-1) = %v", v)
	}
	if v := stdLog(inf); v != inf {
		t.Errorf("stdLog(+Inf) = %v", v)
	}
}

// TestLibmExhaustive runs the three maths functions on every float32 bit
// pattern, in the forms fastText uses them, and compares fingerprints of the
// results with those of glibc 2.41 (accuracy/libm, measured on an
// x86-64 machine; the non-FMA build of glibc).
//
// It takes about a minute of processor time; -short skips it.
func TestLibmExhaustive(t *testing.T) {
	if testing.Short() {
		t.Skip("exhaustive check skipped with -short")
	}
	path := filepath.Join(accuracyDir(), "results", "libm-variants-n4.tsv")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("no fingerprints: %v", err)
	}
	defer f.Close()

	funcs := map[string]func(float32) float32{
		"expf": expf,
		"exp":  func(x float32) float32 { return float32(exp64(float64(x))) },
		"log":  stdLog,
	}
	type job struct {
		name   string
		lo, hi uint64
		want   uint64
	}
	var jobs []job
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		fields := strings.Split(sc.Text(), "\t")
		if len(fields) != 7 {
			t.Fatalf("%s: unexpected line %q", path, sc.Text())
		}
		if funcs[fields[0]] == nil {
			continue // logf is not used for prediction
		}
		var j job
		j.name = fields[0]
		if _, err := fmt.Sscanf(fields[1]+" "+fields[2]+" "+fields[6], "%x %x %x", &j.lo, &j.hi, &j.want); err != nil {
			t.Fatalf("%s: %q: %v", path, sc.Text(), err)
		}
		jobs = append(jobs, j)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if len(jobs) != 48 {
		t.Fatalf("%s: %d ranges, want 48", path, len(jobs))
	}
	var covered [3]uint64
	for _, j := range jobs {
		covered[map[string]int{"expf": 0, "exp": 1, "log": 2}[j.name]] += j.hi - j.lo
	}
	for i, c := range covered {
		if c != 1<<32 {
			t.Fatalf("function %d: ranges cover %d inputs, want 2^32", i, c)
		}
	}

	var wg sync.WaitGroup
	got := make([]uint64, len(jobs))
	sem := make(chan struct{}, 8)
	for i, j := range jobs {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			fn := funcs[j.name]
			h := uint64(14695981039346656037)
			for u := j.lo; u < j.hi; u++ {
				y := fn(math.Float32frombits(uint32(u)))
				b := math.Float32bits(y)
				if y != y {
					b = 0x7fc00000
				}
				h = (h ^ uint64(b&0xff)) * 1099511628211
				h = (h ^ uint64(b>>8&0xff)) * 1099511628211
				h = (h ^ uint64(b>>16&0xff)) * 1099511628211
				h = (h ^ uint64(b>>24)) * 1099511628211
			}
			got[i] = h
		}()
	}
	wg.Wait()
	for i, j := range jobs {
		if got[i] != j.want {
			t.Errorf("%s on [%08x, %09x): fingerprint %016x, glibc %016x", j.name, j.lo, j.hi, got[i], j.want)
		}
	}
}

// TestStdLogMonotonic checks what findKBest relies on to skip logarithms:
// stdLog never decreases as its argument grows from 0 to +Inf.
func TestStdLogMonotonic(t *testing.T) {
	if testing.Short() {
		t.Skip("exhaustive check skipped with -short")
	}
	const parts = 16
	const end = 0x7f800000 // +Inf
	var wg sync.WaitGroup
	bad := make([]int64, parts)
	for p := 0; p < parts; p++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			lo := uint32(uint64(end) * uint64(p) / parts)
			hi := uint32(uint64(end) * uint64(p+1) / parts)
			prev := stdLog(math.Float32frombits(lo))
			bad[p] = -1
			for u := lo + 1; u <= hi; u++ {
				y := stdLog(math.Float32frombits(u))
				if !(y >= prev) {
					bad[p] = int64(u)
					return
				}
				prev = y
			}
		}()
	}
	wg.Wait()
	for _, b := range bad {
		if b >= 0 {
			t.Errorf("stdLog decreases at %08x", b)
		}
	}
}
