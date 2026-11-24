package fasttext

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/slavabobik/fasttext/internal/testdata"
)

// hideSize hides the length of a reader from Load, as an HTTP body or a gzip
// stream does.
func hideSize(r io.Reader) io.Reader { return struct{ io.Reader }{r} }

// header writes a model file up to and including the dictionary: one
// supervised model of dimension 16 without n-grams, with entries words
// followed by one label, and the pruned index size given.
func header(nwords, nlabels int32, pruneidx int64) *bytes.Buffer {
	b := new(bytes.Buffer)
	put := func(v any) {
		if err := binary.Write(b, binary.LittleEndian, v); err != nil {
			panic(err)
		}
	}
	put(int32(fileMagic))
	put(int32(fileVersion))
	// dim, ws, epoch, minCount, neg, wordNgrams, loss, model, bucket, minn,
	// maxn, lrUpdateRate, t
	put([]int32{16, 5, 5, 1, 5, 1, int32(lossSoftmax), modelSupervised, 0, 0, 0, 100})
	put(1e-4)
	put([]int32{nwords + nlabels, nwords, nlabels})
	put(int64(1)) // tokens
	put(pruneidx)
	return b
}

func putLabel(b *bytes.Buffer) {
	b.WriteString("__label__x\x00")
	binary.Write(b, binary.LittleEndian, int64(1))
	b.WriteByte(typeLabel)
}

// TestLoadClaimsTooMuch loads short files whose sizes claim tens of
// gigabytes, through a reader with a known length and through one without.
// Both must fail, and neither may allocate more than a few megabytes: memory
// has to follow the bytes that arrive, not the sizes in the file. Reported by
// Ian (issue #6): a 129-byte file claiming 2^34 rows made Load allocate
// 64 GiB.
func TestLoadClaimsTooMuch(t *testing.T) {
	le := binary.LittleEndian
	cases := []struct {
		name string
		file func() []byte
	}{
		{"dense matrix of 2^34 rows (64 GiB)", func() []byte {
			b := header(0, 1, -1)
			putLabel(b)
			b.WriteByte(0) // dense
			binary.Write(b, le, []int64{1 << 34, 1})
			return b.Bytes()
		}},
		{"dense matrix, data cut short", func() []byte {
			b := header(0, 1, -1)
			putLabel(b)
			b.WriteByte(0)
			binary.Write(b, le, []int64{1 << 30, 16})
			b.Write(make([]byte, 3<<20)) // 3 MB of the 64 GiB
			return b.Bytes()
		}},
		{"dictionary of 2^31-1 entries (32 GB)", func() []byte {
			return header(math.MaxInt32-1, 1, -1).Bytes()
		}},
		{"pruned index of 2^31-1 pairs (32 GB)", func() []byte {
			b := header(0, 1, math.MaxInt32)
			putLabel(b)
			binary.Write(b, le, []int32{7, 0})
			return b.Bytes()
		}},
		{"quantized codes of 2^31-1 bytes", func() []byte {
			b := header(0, 1, 0)
			putLabel(b)
			b.WriteByte(1) // quantized
			b.WriteByte(0) // no norms
			binary.Write(b, le, []int64{math.MaxInt32, 16})
			binary.Write(b, le, int32(math.MaxInt32))
			return b.Bytes()
		}},
		{"product quantizer of 2^26 dimensions (64 GiB)", func() []byte {
			b := header(0, 1, 0)
			putLabel(b)
			b.WriteByte(1)
			b.WriteByte(0)
			binary.Write(b, le, []int64{1, 1 << 26})
			binary.Write(b, le, int32(1))
			b.WriteByte(0) // the one code
			binary.Write(b, le, []int32{1 << 26, 1, 1 << 26, 1 << 26})
			return b.Bytes()
		}},
	}
	const limit = 16 << 20
	for _, c := range cases {
		data := c.file()
		for _, mode := range []string{"known size", "unknown size"} {
			t.Run(c.name+", "+mode, func(t *testing.T) {
				var r io.Reader = bytes.NewReader(data)
				if mode == "unknown size" {
					r = hideSize(r)
				}
				var before, after runtime.MemStats
				runtime.ReadMemStats(&before)
				_, err := Load(r)
				runtime.ReadMemStats(&after)
				if err == nil {
					t.Fatalf("%d-byte file loaded", len(data))
				}
				if n := after.TotalAlloc - before.TotalAlloc; n > limit {
					t.Errorf("%d-byte file: Load allocated %d MB before failing (%v)", len(data), n>>20, err)
				}
			})
		}
	}
	if n := len(cases[0].file()); n != 129 {
		t.Errorf("the reported file has %d bytes, want 129", n)
	}
}

// TestLoadUnknownSize checks that a model read from a stream, where the
// slices grow as the data arrives, is the same as one read from its file.
func TestLoadUnknownSize(t *testing.T) {
	for _, name := range []string{"lid.176.ftz", "lid.176.bin", "t-softmax-d32.ftz", "t-ns-d16.bin", "t-hs-d20-wn3.ftz"} {
		t.Run(name, func(t *testing.T) {
			path := testdata.Model(t, name)
			want, err := Open(path)
			if err != nil {
				t.Fatal(err)
			}
			f, err := os.Open(path)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			got, err := Load(hideSize(f))
			if err != nil {
				t.Fatal(err)
			}
			for _, field := range []struct {
				name      string
				got, want any
			}{
				{"version", got.version, want.version},
				{"args", got.args, want.args},
				{"dict", &got.dict, &want.dict},
				{"labels", got.labels, want.labels},
				{"quant", got.quant, want.quant},
				{"in", &got.in, &want.in},
				{"qin", &got.qin, &want.qin},
				{"qout", got.qout, want.qout},
				{"out", &got.out, &want.out},
				{"outT", got.outT, want.outT},
				{"qoutM", &got.qoutM, &want.qoutM},
				{"tree", [][]int32{got.left, got.right}, [][]int32{want.left, want.right}},
			} {
				if !reflect.DeepEqual(field.got, field.want) {
					t.Errorf("%s differs", field.name)
				}
			}
		})
	}
}
