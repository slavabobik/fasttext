package fasttext

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// TestHeapMatchesLibstdcxx replays the cases of accuracy/heap/heapref.cc,
// which runs fastText's best-k loop on scores with many ties using the C++
// standard library, and compares the order of the results with the
// fingerprints that program wrote when built with the reference toolchain
// (accuracy/golden/heap.tsv).
func TestHeapMatchesLibstdcxx(t *testing.T) {
	path := filepath.Join(accuracyDir(), "golden", "heap.tsv")
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("no reference fingerprints: %v", err)
	}
	defer f.Close()

	state := uint64(20261120)
	next := func() uint32 {
		state = state*6364136223846793005 + 1442695040888963407
		return uint32(state >> 33)
	}
	cases := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var n, k, v int
		var want uint64
		if _, err := fmt.Sscanf(sc.Text(), "%d\t%d\t%d\t%x", &n, &k, &v, &want); err != nil {
			t.Fatalf("%s: %q: %v", path, sc.Text(), err)
		}
		h := uint64(14695981039346656037)
		for rep := 0; rep < 50; rep++ {
			var heap []scored
			for i := 0; i < n; i++ {
				score := float32(next() % uint32(v))
				if len(heap) == k && score < heap[0].score {
					continue
				}
				heap = append(heap, scored{score, int32(i)})
				pushHeap(heap)
				if len(heap) > k {
					popHeap(heap)
					heap = heap[:len(heap)-1]
				}
			}
			sortHeap(heap)
			for i, p := range heap {
				if i > 0 && p.score > heap[i-1].score {
					t.Fatalf("n=%d k=%d: result not ordered best first", n, k)
				}
				u := uint32(p.label)
				for b := 0; b < 32; b += 8 {
					h = (h ^ uint64(byte(u>>b))) * 1099511628211
				}
			}
		}
		if h != want {
			t.Errorf("n=%d k=%d distinct=%d: fingerprint %016x, libstdc++ %016x", n, k, v, h, want)
		}
		cases++
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if cases != 350 {
		t.Errorf("%d cases, want 350", cases)
	}
}
