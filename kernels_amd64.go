//go:build goexperiment.simd && amd64

package fasttext

import "simd/archsimd"

func init() {
	if archsimd.X86.AVX2() {
		kernels = append(kernels, kernel{"avx2", addAVX2, add4AVX2, axpyAVX2})
	}
	if archsimd.X86.AVX512() {
		kernels = append(kernels, kernel{"avx512", addAVX512, add4AVX512, axpyAVX512})
	}
	useBestKernel()
}

// Every function ends with ClearAVXUpperBits: Go's own floating-point code
// uses the older instruction encoding, which runs slowly while the upper
// halves of the vector registers hold data.

func addAVX2(dst, src []float32) {
	src = src[:len(dst)]
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		d := dst[i : i+16 : i+16]
		s := src[i : i+16 : i+16]
		archsimd.LoadFloat32x8(d).Add(archsimd.LoadFloat32x8(s)).Store(d)
		archsimd.LoadFloat32x8(d[8:]).Add(archsimd.LoadFloat32x8(s[8:])).Store(d[8:])
	}
	archsimd.ClearAVXUpperBits()
	for ; i < len(dst); i++ {
		dst[i] += src[i]
	}
}

func add4AVX2(dst, a, b, c, d []float32) {
	a, b, c, d = a[:len(dst)], b[:len(dst)], c[:len(dst)], d[:len(dst)]
	i := 0
	for ; i+8 <= len(dst); i += 8 {
		ds := dst[i : i+8 : i+8]
		archsimd.LoadFloat32x8(ds).
			Add(archsimd.LoadFloat32x8(a[i : i+8 : i+8])).
			Add(archsimd.LoadFloat32x8(b[i : i+8 : i+8])).
			Add(archsimd.LoadFloat32x8(c[i : i+8 : i+8])).
			Add(archsimd.LoadFloat32x8(d[i : i+8 : i+8])).Store(ds)
	}
	archsimd.ClearAVXUpperBits()
	for ; i < len(dst); i++ {
		dst[i] = dst[i] + a[i] + b[i] + c[i] + d[i]
	}
}

// axpyAVX2 multiplies and adds in two instructions. A fused multiply-add
// would round once instead of twice and change the result.
func axpyAVX2(y, x []float32, a float32) {
	x = x[:len(y)]
	va := archsimd.BroadcastFloat32x8(a)
	for i := 0; i+16 <= len(y); i += 16 {
		ys := y[i : i+16 : i+16]
		xs := x[i : i+16 : i+16]
		archsimd.LoadFloat32x8(ys).Add(archsimd.LoadFloat32x8(xs).Mul(va)).Store(ys)
		archsimd.LoadFloat32x8(ys[8:]).Add(archsimd.LoadFloat32x8(xs[8:]).Mul(va)).Store(ys[8:])
	}
	archsimd.ClearAVXUpperBits()
}

func addAVX512(dst, src []float32) {
	src = src[:len(dst)]
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		d := dst[i : i+16 : i+16]
		s := src[i : i+16 : i+16]
		archsimd.LoadFloat32x16(d).Add(archsimd.LoadFloat32x16(s)).Store(d)
	}
	archsimd.ClearAVXUpperBits()
	for ; i < len(dst); i++ {
		dst[i] += src[i]
	}
}

func add4AVX512(dst, a, b, c, d []float32) {
	a, b, c, d = a[:len(dst)], b[:len(dst)], c[:len(dst)], d[:len(dst)]
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		ds := dst[i : i+16 : i+16]
		archsimd.LoadFloat32x16(ds).
			Add(archsimd.LoadFloat32x16(a[i : i+16 : i+16])).
			Add(archsimd.LoadFloat32x16(b[i : i+16 : i+16])).
			Add(archsimd.LoadFloat32x16(c[i : i+16 : i+16])).
			Add(archsimd.LoadFloat32x16(d[i : i+16 : i+16])).Store(ds)
	}
	archsimd.ClearAVXUpperBits()
	for ; i < len(dst); i++ {
		dst[i] = dst[i] + a[i] + b[i] + c[i] + d[i]
	}
}

func axpyAVX512(y, x []float32, a float32) {
	x = x[:len(y)]
	va := archsimd.BroadcastFloat32x16(a)
	for i := 0; i+16 <= len(y); i += 16 {
		ys := y[i : i+16 : i+16]
		xs := x[i : i+16 : i+16]
		archsimd.LoadFloat32x16(ys).Add(archsimd.LoadFloat32x16(xs).Mul(va)).Store(ys)
	}
	archsimd.ClearAVXUpperBits()
}
