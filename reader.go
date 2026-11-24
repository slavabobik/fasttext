package fasttext

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
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

// maxUnknown bounds one allocation when the input size is unknown.
const maxUnknown = 1 << 36

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

func (r *reader) bytes(n int) []byte {
	if r.err != nil {
		return nil
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r.br, b); err != nil {
		r.fail(err)
		return nil
	}
	r.consumed(n)
	return b
}

var hostLittleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// float32s reads n floats straight into the memory of the returned slice, so
// a large matrix is not held twice.
func (r *reader) float32s(n int) []float32 {
	if r.err != nil || n == 0 {
		return nil
	}
	f := make([]float32, n)
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(f))), n*4)
	if _, err := io.ReadFull(r.br, b); err != nil {
		r.fail(err)
		return nil
	}
	r.consumed(len(b))
	if !hostLittleEndian {
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
		}
	}
	return f
}

var errNotModel = errors.New("fasttext: not a fastText model file")
