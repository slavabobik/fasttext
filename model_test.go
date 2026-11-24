package fasttext

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/slavabobik/fasttext/internal/testdata"
)

// TestModelReader checks the model reader against the reference tool: the
// dictionary (every word and label with its count) and the dense matrices
// must hash to the same value as the output of "refdump --info"
// (accuracy/golden/info.sha256).
func TestModelReader(t *testing.T) {
	f, err := os.Open(filepath.Join(accuracyDir(), "golden", "info.sha256"))
	if err != nil {
		t.Skipf("no reference: %v", err)
	}
	defer f.Close()
	n := 0
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		want, name, ok := strings.Cut(sc.Text(), "  ")
		if !ok {
			t.Fatalf("unexpected line %q", sc.Text())
		}
		n++
		t.Run(name, func(t *testing.T) {
			m, err := Open(testdata.Model(t, name))
			if err != nil {
				t.Fatal(err)
			}
			h := sha256.New()
			if err := m.writeInfo(h); err != nil {
				t.Fatal(err)
			}
			if got := hex.EncodeToString(h.Sum(nil)); got != want {
				t.Errorf("dictionary and matrices hash to %s, reference %s", got, want)
			}
		})
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n != 11 {
		t.Errorf("%d models listed, want 11", n)
	}
}

// TestDamagedModel loads truncated and randomly damaged model files. Loading
// may fail or succeed, and a model that loads may predict anything, but
// nothing may panic or ask for absurd amounts of memory.
func TestDamagedModel(t *testing.T) {
	texts := []string{"", "hello world", "Это предложение написано по-русски.", "a b c d e f g </s> h", strings.Repeat("x", 300)}
	for _, name := range []string{"lid.176.ftz", "t-softmax-d32.ftz", "t-ova-d64.ftz", "t-hs-d20-wn3.ftz", "t-ns-d16.bin"} {
		t.Run(name, func(t *testing.T) {
			data, err := os.ReadFile(testdata.Model(t, name))
			if err != nil {
				t.Skipf("model not available: %v", err)
			}
			try := func(b []byte) {
				m, err := Load(bytes.NewReader(b))
				if err != nil {
					return
				}
				for _, text := range texts {
					for _, k := range []int{1, 3, -1} {
						m.Predict(text, k, 0)
					}
				}
			}

			// Truncation: every length in the header, then a spread.
			for n := 0; n < 200 && n < len(data); n++ {
				if _, err := Load(bytes.NewReader(data[:n])); err == nil {
					t.Fatalf("truncated to %d bytes: loaded", n)
				}
			}
			rng := rand.New(rand.NewPCG(20261120, uint64(len(data))))
			for i := 0; i < 40; i++ {
				n := rng.IntN(len(data))
				if _, err := Load(bytes.NewReader(data[:n])); err == nil {
					t.Fatalf("truncated to %d of %d bytes: loaded", n, len(data))
				}
			}

			// Damage: the header and the dictionary counters byte by byte,
			// then random places.
			b := make([]byte, len(data))
			for pos := 0; pos < 96; pos++ {
				for _, v := range []byte{0x00, 0x01, 0x7f, 0x80, 0xff} {
					copy(b, data)
					b[pos] = v
					try(b)
				}
			}
			for i := 0; i < 300; i++ {
				copy(b, data)
				for j := 0; j < 1+i%4; j++ {
					b[rng.IntN(len(b))] = byte(rng.Uint32())
				}
				try(b)
			}
			// The structures that follow the dictionary: sizes of the pruned
			// index and of the matrices.
			for i := 0; i < 300; i++ {
				copy(b, data)
				pos := len(b) - 1 - rng.IntN(min(len(b), 1<<16))
				b[pos] ^= byte(1 << rng.IntN(8))
				try(b)
			}
		})
	}
}

func TestHash(t *testing.T) {
	// FNV-1a values; the last one shows the sign extension of bytes above
	// 0x7f that fastText keeps for compatibility.
	if got := hash(""); got != 2166136261 {
		t.Errorf("hash(\"\") = %d", got)
	}
	if got := hash("a"); got != 0xe40c292c {
		t.Errorf("hash(\"a\") = %#x", got)
	}
	plain := uint32(2166136261)
	plain = (plain ^ 0xc3) * 16777619
	if got := hash("\xc3"); got == plain {
		t.Errorf("hash of a byte above 0x7f equals plain FNV-1a: no sign extension")
	}
}
