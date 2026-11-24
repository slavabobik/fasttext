package fasttext

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"unsafe"
)

// reader reads the little-endian binary values of a fastText model file. The
// first error sticks: after it every read returns zero values.
type reader struct {
	br   *bufio.Reader
	err  error
	left int64 // bytes left in the input, or -1 if unknown
	buf  [8]byte
}

// maxUnknown is what remaining reports when the input size is unknown. It
// only rejects sizes no model can have: what bounds memory then is that
// readSlice and the dictionary grow with the bytes that arrive.
const maxUnknown = 1 << 36

// chunkBytes is the most memory a read of unknown size allocates ahead of the
// bytes that have arrived.
const chunkBytes = 1 << 20

func newReader(r io.Reader, size int64) *reader {
	return &reader{br: bufio.NewReaderSize(r, 1<<16), left: size}
}

func (r *reader) fail(err error) {
	if r.err == nil {
		if errors.Is(err, io.EOF) {
			err = io.ErrUnexpectedEOF
		}
		r.err = fmt.Errorf("fasttext: reading model: %w", err)
	}
}

func (r *reader) consumed(n int) {
	if r.left >= 0 {
		r.left -= int64(n)
	}
}

// remaining returns the bytes left, or a large number if unknown.
func (r *reader) remaining() int64 {
	if r.left < 0 {
		return maxUnknown
	}
	return r.left
}

// sizeKnown reports whether the sizes in the file have been checked against
// the length of the input.
func (r *reader) sizeKnown() bool { return r.left >= 0 }

// capHint returns the capacity to allocate for n values that will be read one
// by one: n when the input size is known (and n has been checked against
// it), at most limit otherwise.
func (r *reader) capHint(n, limit int) int {
	if r.sizeKnown() {
		return n
	}
	return min(n, limit)
}

func (r *reader) fixed(n int) []byte {
	b := r.buf[:n]
	if r.err != nil {
		clear(b)
		return b
	}
	if _, err := io.ReadFull(r.br, b); err != nil {
		r.fail(err)
		clear(b)
		return b
	}
	r.consumed(n)
	return b
}

func (r *reader) int8() int8   { return int8(r.fixed(1)[0]) }
func (r *reader) bool() bool   { return r.fixed(1)[0] != 0 }
func (r *reader) int32() int32 { return int32(binary.LittleEndian.Uint32(r.fixed(4))) }
func (r *reader) int64() int64 { return int64(binary.LittleEndian.Uint64(r.fixed(8))) }

func (r *reader) float64() float64 {
	return math.Float64frombits(binary.LittleEndian.Uint64(r.fixed(8)))
}

// cstring reads a zero-terminated string.
func (r *reader) cstring() string {
	if r.err != nil {
		return ""
	}
	b, err := r.br.ReadBytes(0)
	if err != nil {
		r.fail(err)
		return ""
	}
	r.consumed(len(b))
	return string(b[:len(b)-1])
}

// elements checks the shape of a matrix and returns rows*cols. size is the
// number of bytes per element in the file, or 0 if the elements are not
// stored one by one.
func (r *reader) elements(rows, cols, size int64) (int, error) {
	if rows < 0 || cols < 0 || (cols != 0 && rows > math.MaxInt64/8/cols) {
		return 0, fmt.Errorf("fasttext: invalid matrix size %d×%d", rows, cols)
	}
	n := rows * cols
	if size > 0 && n > r.remaining()/size {
		return 0, fmt.Errorf("fasttext: matrix of %d×%d does not fit in the input", rows, cols)
	}
	if int64(int(n)) != n {
		return 0, fmt.Errorf("fasttext: matrix of %d×%d is too large for this platform", rows, cols)
	}
	return int(n), nil
}

func (r *reader) bytes(n int) []byte { return readSlice[byte](r, n) }

var hostLittleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// float32s reads n floats straight into the memory of the returned slice, so
// a large matrix is not held twice.
func (r *reader) float32s(n int) []float32 {
	f := readSlice[float32](r, n)
	if !hostLittleEndian {
		b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f))), len(f)*4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
	}
	return f
}

// readSlice reads n values of T as they are stored in the file, straight
// into the memory of the returned slice. When the input size is known, n has
// been checked against it and the slice is allocated at once. When it is not
// (Load from a stream), the slice starts at chunkBytes and doubles as the
// data arrives, so a file that claims more than it holds fails after
// allocating at most about three times what it held, not what it claimed.
func readSlice[T byte | float32](r *reader, n int) []T {
	if r.err != nil || n == 0 {
		return nil
	}
	size := int(unsafe.Sizeof(T(0)))
	s := make([]T, r.capHint(n, max(1, chunkBytes/size)))
	done := 0
	for {
		b := unsafe.Slice((*byte)(unsafe.Pointer(&s[done])), (len(s)-done)*size)
		if _, err := io.ReadFull(r.br, b); err != nil {
			r.fail(err)
			return nil
		}
		r.consumed(len(b))
		done = len(s)
		if done == n {
			return s
		}
		next := min(n, 2*done)
		s = slices.Grow(s, next-done)[:next]
	}
}

var errNotModel = errors.New("fasttext: not a fastText model file")
