//go:build goexperiment.simd && arm64

package fasttext

import "simd/archsimd"

func init() {
	kernels = append(kernels, kernel{"neon", addNEON, add4NEON, axpyNEON})
	useBestKernel()
}

func addNEON(dst, src []float32) {
	src = src[:len(dst)]
	i := 0
	for ; i+16 <= len(dst); i += 16 {
		d := dst[i : i+16 : i+16]
		s := src[i : i+16 : i+16]
		archsimd.LoadFloat32x4(d).Add(archsimd.LoadFloat32x4(s)).Store(d)
		archsimd.LoadFloat32x4(d[4:]).Add(archsimd.LoadFloat32x4(s[4:])).Store(d[4:])
		archsimd.LoadFloat32x4(d[8:]).Add(archsimd.LoadFloat32x4(s[8:])).Store(d[8:])
		archsimd.LoadFloat32x4(d[12:]).Add(archsimd.LoadFloat32x4(s[12:])).Store(d[12:])
	}
	for ; i < len(dst); i++ {
		dst[i] += src[i]
	}
}

func add4NEON(dst, a, b, c, d []float32) {
	a, b, c, d = a[:len(dst)], b[:len(dst)], c[:len(dst)], d[:len(dst)]
	i := 0
	for ; i+8 <= len(dst); i += 8 {
		ds := dst[i : i+8 : i+8]
		as, bs, cs, es := a[i:i+8:i+8], b[i:i+8:i+8], c[i:i+8:i+8], d[i:i+8:i+8]
		archsimd.LoadFloat32x4(ds).
			Add(archsimd.LoadFloat32x4(as)).
			Add(archsimd.LoadFloat32x4(bs)).
			Add(archsimd.LoadFloat32x4(cs)).
			Add(archsimd.LoadFloat32x4(es)).Store(ds)
		archsimd.LoadFloat32x4(ds[4:]).
			Add(archsimd.LoadFloat32x4(as[4:])).
			Add(archsimd.LoadFloat32x4(bs[4:])).
			Add(archsimd.LoadFloat32x4(cs[4:])).
			Add(archsimd.LoadFloat32x4(es[4:])).Store(ds[4:])
	}
	for ; i < len(dst); i++ {
		dst[i] = dst[i] + a[i] + b[i] + c[i] + d[i]
	}
}

// axpyNEON multiplies and adds in two instructions. A fused multiply-add
// would round once instead of twice and change the result.
func axpyNEON(y, x []float32, a float32) {
	x = x[:len(y)]
	va := archsimd.BroadcastFloat32x4(a)
	for i := 0; i+16 <= len(y); i += 16 {
		ys := y[i : i+16 : i+16]
		xs := x[i : i+16 : i+16]
		archsimd.LoadFloat32x4(ys).Add(archsimd.LoadFloat32x4(xs).Mul(va)).Store(ys)
		archsimd.LoadFloat32x4(ys[4:]).Add(archsimd.LoadFloat32x4(xs[4:]).Mul(va)).Store(ys[4:])
		archsimd.LoadFloat32x4(ys[8:]).Add(archsimd.LoadFloat32x4(xs[8:]).Mul(va)).Store(ys[8:])
		archsimd.LoadFloat32x4(ys[12:]).Add(archsimd.LoadFloat32x4(xs[12:]).Mul(va)).Store(ys[12:])
	}
}
