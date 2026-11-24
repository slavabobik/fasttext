package fasttext

import "math"

// fastText calls expf, exp and log from the C maths library, and the last bit
// of a probability depends on how that library rounds. The functions below
// are ports of the routines glibc has used since 2.27 and 2.28, which come
// from ARM Optimized Routines (math/expf.c, math/exp.c, math/log.c,
// Copyright (c) 2017-2026 Arm Limited, MIT license; see
// LICENSE-arm-optimized-routines).
//
// They follow the build glibc selects on x86-64 CPUs without FMA: no fused
// multiply-add, and rounding to an integer with the shift trick. Every product
// that feeds an addition is wrapped in an explicit conversion. The conversion
// is what forbids the Go compiler from fusing the two operations (it does so
// on arm64 otherwise), so do not remove them.
//
// accuracy/libm checks the three functions against glibc on every float32
// input.

var (
	exp2fInvLn2Scaled = math.Float64frombits(exp2fInvLn2ScaledBits)
	exp2fShift        = math.Float64frombits(exp2fShiftBits)
	exp2fPolyScaled   = floats3(exp2fPolyScaledBits)

	expInvLn2N   = math.Float64frombits(expInvLn2NBits)
	expNegLn2HiN = math.Float64frombits(expNegLn2HiNBits)
	expNegLn2LoN = math.Float64frombits(expNegLn2LoNBits)
	expShift     = math.Float64frombits(expShiftBits)
	expPoly      = floats4(expPolyBits)

	logLn2Hi = math.Float64frombits(logLn2HiBits)
	logLn2Lo = math.Float64frombits(logLn2LoBits)
	logPoly  = floats5(logPolyBits)
	logPoly1 = floats11(logPoly1Bits)
)

func floats3(b [3]uint64) (f [3]float64) {
	for i := range b {
		f[i] = math.Float64frombits(b[i])
	}
	return
}

func floats4(b [4]uint64) (f [4]float64) {
	for i := range b {
		f[i] = math.Float64frombits(b[i])
	}
	return
}

func floats5(b [5]uint64) (f [5]float64) {
	for i := range b {
		f[i] = math.Float64frombits(b[i])
	}
	return
}

func floats11(b [11]uint64) (f [11]float64) {
	for i := range b {
		f[i] = math.Float64frombits(b[i])
	}
	return
}

// expf is C's expf.
func expf(x float32) float32 {
	ix := math.Float32bits(x)
	xd := float64(x)
	abstop := (ix >> 20) & 0x7ff
	if abstop >= 0x42b { // |x| >= 88 or x is NaN
		if ix == 0xff800000 { // -Inf
			return 0
		}
		if abstop >= 0x7f8 { // +Inf or NaN
			return x + x
		}
		if x > 0x1.62e42ep6 { // x > log(0x1p128)
			return float32(math.Inf(1))
		}
		if x < -0x1.9fe368p6 { // x < log(0x1p-150)
			return 0
		}
		if x < -0x1.9d1d9ep6 { // x < log(0x1p-149)
			return 0x1p-149
		}
	}

	// x*N/ln2 = k + r with r in [-1/2, 1/2] and integer k.
	z := float64(exp2fInvLn2Scaled * xd)
	kd := z + exp2fShift
	ki := math.Float64bits(kd)
	kd -= exp2fShift
	r := z - kd

	// exp(x) = 2^(k/N) * 2^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)
	t := exp2fTab[ki%32]
	t += ki << (52 - 5)
	s := math.Float64frombits(t)
	z = float64(exp2fPolyScaled[0]*r) + exp2fPolyScaled[1]
	r2 := float64(r * r)
	y := float64(exp2fPolyScaled[2]*r) + 1
	y = float64(z*r2) + y
	y = float64(y * s)
	return float32(y)
}

// exp64 is C's exp for a double argument.
func exp64(x float64) float64 {
	ix := math.Float64bits(x)
	abstop := uint32(ix>>52) & 0x7ff
	// 0x3c9 is the exponent field of 0x1p-54, 0x408 that of 512.
	if abstop-0x3c9 >= 0x408-0x3c9 {
		if abstop-0x3c9 >= 0x80000000 {
			// Tiny x.
			return 1.0 + x
		}
		if abstop >= 0x409 { // |x| >= 1024
			if ix == 0xfff0000000000000 { // -Inf
				return 0
			}
			if abstop >= 0x7ff { // +Inf or NaN
				return 1.0 + x
			}
			if ix>>63 != 0 {
				return 0
			}
			return math.Inf(1)
		}
		// Large x is special-cased below.
		abstop = 0
	}

	// exp(x) = 2^(k/N) * exp(r), with exp(r) in [2^(-1/2N), 2^(1/2N)].
	// x = ln2/N*k + r, with integer k and r in [-ln2/2N, ln2/2N].
	z := float64(expInvLn2N * x)
	kd := z + expShift
	ki := math.Float64bits(kd)
	kd -= expShift
	r := x + float64(kd*expNegLn2HiN) + float64(kd*expNegLn2LoN)
	// 2^(k/N) ~= scale * (1 + tail).
	idx := 2 * (ki % 128)
	top := ki << (52 - 7)
	tail := math.Float64frombits(expTab[idx])
	// This is only a valid scale when -1023*N < k < 1024*N.
	sbits := expTab[idx+1] + top
	// exp(x) = 2^(k/N) * exp(r) ~= scale + scale * (tail + exp(r) - 1).
	r2 := float64(r * r)
	tmp := tail + r + float64(r2*(expPoly[0]+float64(r*expPoly[1]))) +
		float64(float64(r2*r2)*(expPoly[2]+float64(r*expPoly[3])))
	if abstop == 0 {
		return expSpecial(tmp, sbits, ki)
	}
	scale := math.Float64frombits(sbits)
	return scale + float64(scale*tmp)
}

// expSpecial handles results that may overflow or underflow: scale*(1+tmp)
// computed without intermediate rounding problems.
func expSpecial(tmp float64, sbits, ki uint64) float64 {
	if ki&0x80000000 == 0 {
		// k > 0, the exponent of scale might have overflowed by <= 460.
		sbits -= 1009 << 52
		scale := math.Float64frombits(sbits)
		return float64(0x1p1009 * (scale + float64(scale*tmp)))
	}
	// k < 0, need special care in the subnormal range.
	sbits += 1022 << 52
	scale := math.Float64frombits(sbits)
	y := scale + float64(scale*tmp)
	if y < 1.0 {
		// Round y to the right precision before scaling it into the
		// subnormal range to avoid double rounding.
		lo := scale - y + float64(scale*tmp)
		hi := 1.0 + y
		lo = 1.0 - hi + y + lo
		y = (hi + lo) - 1.0
		if y == 0.0 {
			y = 0.0 // no negative zero
		}
	}
	return float64(0x1p-1022 * y)
}

// log64 is C's log for a double argument.
func log64(x float64) float64 {
	ix := math.Float64bits(x)
	top := uint32(ix >> 48)

	const (
		lo1 = 0x3fee000000000000 // 1.0 - 0x1p-4
		hi1 = 0x3ff1090000000000 // 1.0 + 0x1.09p-4
	)
	if ix-lo1 < hi1-lo1 {
		// Inputs close to 1.0.
		if ix == 0x3ff0000000000000 {
			return 0
		}
		b := &logPoly1
		r := x - 1.0
		r2 := float64(r * r)
		r3 := float64(r * r2)
		p3 := b[7] + float64(r*b[8]) + float64(r2*b[9]) + float64(r3*b[10])
		p2 := b[4] + float64(r*b[5]) + float64(r2*b[6]) + float64(r3*p3)
		p1 := b[1] + float64(r*b[2]) + float64(r2*b[3]) + float64(r3*p2)
		y := float64(r3 * p1)
		w := float64(r * 0x1p27)
		rhi := r + w - w
		rlo := r - rhi
		w = float64(float64(rhi*rhi) * b[0]) // b[0] == -0.5
		hi := r + w
		lo := r - hi + w
		lo += float64(float64(b[0]*rlo) * (rhi + r))
		y += lo
		y += hi
		return y
	}
	if top-0x0010 >= 0x7ff0-0x0010 {
		// x < 0x1p-1022 or Inf or NaN.
		if ix*2 == 0 {
			return math.Inf(-1)
		}
		if ix == 0x7ff0000000000000 {
			return x
		}
		if top&0x8000 != 0 || top&0x7ff0 == 0x7ff0 {
			return math.NaN()
		}
		// x is subnormal, normalise it.
		ix = math.Float64bits(float64(x * 0x1p52))
		ix -= 52 << 52
	}

	// x = 2^k z, where z is in [OFF, 2*OFF) and exact. The range is split
	// into N subintervals; the ith contains z and c is near its centre.
	const off = 0x3fe6000000000000
	tmp := ix - off
	i := (tmp >> (52 - 7)) % 128
	k := int64(tmp) >> 52
	iz := ix - (tmp & (0xfff << 52))
	invc := math.Float64frombits(logTab[2*i])
	logc := math.Float64frombits(logTab[2*i+1])
	z := math.Float64frombits(iz)

	// log(x) = log1p(z/c-1) + log(c) + k*ln2, with r ~= z/c - 1.
	chi := math.Float64frombits(logTab2[2*i])
	clo := math.Float64frombits(logTab2[2*i+1])
	r := float64((z - chi - clo) * invc)
	kd := float64(k)

	// hi + lo = r + log(c) + k*ln2.
	w := float64(kd*logLn2Hi) + logc
	hi := w + r
	lo := w - hi + r + float64(kd*logLn2Lo)

	// log(x) = lo + (log1p(r) - r) + hi.
	a := &logPoly
	r2 := float64(r * r)
	p := a[1] + float64(r*a[2]) + float64(r2*(a[3]+float64(r*a[4])))
	return lo + float64(r2*a[0]) + float64(float64(r*r2)*p) + hi
}

// stdLog is fastText's std_log: log(x + 1e-5), computed in double and
// rounded to float.
func stdLog(x float32) float32 {
	return float32(log64(float64(x) + 1e-5))
}
