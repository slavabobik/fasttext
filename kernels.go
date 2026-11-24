package fasttext

// The two loops below take most of the time of a prediction with a large
// model: adding input rows into the hidden vector, and the dot products of
// the hidden vector with every output row. Both act on each element
// separately, so a vector instruction that multiplies and then adds (never a
// fused multiply-add) gives the same bits as the plain Go loop. When the
// package is built with GOEXPERIMENT=simd, kernels_amd64.go and
// kernels_arm64.go install such versions here.

// A kernel is one implementation of the two loops.
type kernel struct {
	name string
	// add adds src to dst element by element. len(src) >= len(dst).
	add func(dst, src []float32)
	// add4 adds four rows to dst, element by element and in the order a,
	// b, c, d: dst[i] = (((dst[i] + a[i]) + b[i]) + c[i]) + d[i]. The
	// result is that of four calls of add; dst is read and written once
	// instead of four times.
	add4 func(dst, a, b, c, d []float32)
	// axpy computes y[i] += a*x[i], the product rounded before the
	// addition. len(y) is a multiple of axpyBlock and len(x) >= len(y).
	axpy func(y, x []float32, a float32)
}

// axpyBlock is the multiple to which the label count is padded in the
// transposed output matrix, so that axpy needs no code for a tail.
const axpyBlock = 16

// kernels lists the vector kernels this machine can run, best last. Empty in
// a build without GOEXPERIMENT=simd.
var kernels []kernel

// vec is the kernel in use, or nil for the plain Go loops.
var vec *kernel

func useBestKernel() {
	vec = nil
	if len(kernels) > 0 {
		vec = &kernels[len(kernels)-1]
	}
}

// Kernel names the implementation of the inner loops in use: "go" for plain
// Go, or the vector instruction set chosen for this machine in a build with
// GOEXPERIMENT=simd. Results do not depend on it.
func Kernel() string {
	if vec == nil {
		return "go"
	}
	return vec.name
}

// addTo adds src to dst element by element.
func addTo(dst, src []float32) {
	src = src[:len(dst)]
	j := 0
	for ; j+4 <= len(dst); j += 4 {
		d := dst[j : j+4 : j+4]
		s := src[j : j+4 : j+4]
		d[0] += s[0]
		d[1] += s[1]
		d[2] += s[2]
		d[3] += s[3]
	}
	for ; j < len(dst); j++ {
		dst[j] += src[j]
	}
}

// addTo4 is the plain Go form of kernel.add4.
func addTo4(dst, a, b, c, d []float32) {
	a, b, c, d = a[:len(dst)], b[:len(dst)], c[:len(dst)], d[:len(dst)]
	for j := range dst {
		dst[j] = dst[j] + a[j] + b[j] + c[j] + d[j]
	}
}

// axpy is the plain Go form of kernel.axpy.
func axpy(y, x []float32, a float32) {
	x = x[:len(y)]
	for i := range y {
		y[i] += float32(a * x[i])
	}
}

// dotRow is DenseMatrix::dotRow: the products are added one by one, starting
// from zero. Any other order of additions changes the result.
func dotRow(row, x []float32) float32 {
	x = x[:len(row)]
	var d float32
	for j, v := range row {
		d += float32(v * x[j])
	}
	return d
}

// dotRows fills out[i] with dotRow(row i of w, h) for a matrix w of rows of
// len(h) values. It runs eight rows at a time: each sum still receives its
// products one by one in order, as in dotRow, but the processor no longer
// waits for one addition to finish before it starts the next.
func dotRows(out, w, h []float32) {
	dim := len(h)
	n := len(out)
	i := 0
	for ; i+8 <= n; i += 8 {
		r0 := w[i*dim : (i+1)*dim : (i+1)*dim]
		r1 := w[(i+1)*dim : (i+2)*dim : (i+2)*dim]
		r2 := w[(i+2)*dim : (i+3)*dim : (i+3)*dim]
		r3 := w[(i+3)*dim : (i+4)*dim : (i+4)*dim]
		r4 := w[(i+4)*dim : (i+5)*dim : (i+5)*dim]
		r5 := w[(i+5)*dim : (i+6)*dim : (i+6)*dim]
		r6 := w[(i+6)*dim : (i+7)*dim : (i+7)*dim]
		r7 := w[(i+7)*dim : (i+8)*dim : (i+8)*dim]
		var d0, d1, d2, d3, d4, d5, d6, d7 float32
		for j, hj := range h {
			d0 += float32(r0[j] * hj)
			d1 += float32(r1[j] * hj)
			d2 += float32(r2[j] * hj)
			d3 += float32(r3[j] * hj)
			d4 += float32(r4[j] * hj)
			d5 += float32(r5[j] * hj)
			d6 += float32(r6[j] * hj)
			d7 += float32(r7[j] * hj)
		}
		o := out[i : i+8 : i+8]
		o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7] = d0, d1, d2, d3, d4, d5, d6, d7
	}
	for ; i < n; i++ {
		out[i] = dotRow(w[i*dim:(i+1)*dim], h)
	}
}
