// Compares glibc's two x86-64 builds of the maths routines fastText calls:
// the one picked on CPUs with FMA and the one picked on CPUs without it.
// glibc chooses between them when the program starts, so the same static
// binary could give different results on different machines.
//
// Every float bit pattern in [lo, hi) is tried, in the form fastText uses:
//
//   expf  expf(x)                 FastText::predictLine, sigmoid in the tree
//   logf  logf(x)                 the log table (training only)
//   exp   (float) exp((double)x)  softmax, on libstdc++
//   log   (float) log(x + 1e-5)   std_log
//
// For each routine it prints how many inputs give different results, before
// and after rounding to float, and a fingerprint (FNV-1a over the float
// results of the non-FMA build) that a port can be checked against without
// storing the results.
//
// build: cc -O2 -ffp-contract=off -static variants.c -lm -o variants
// usage: variants <lo> <hi>      (hex, e.g. 0 100000000)
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float __expf_fma(float);
float __expf_sse2(float);
float __logf_fma(float);
float __logf_sse2(float);
double __ieee754_exp_fma(double);
double __ieee754_exp_sse2(double);
double __ieee754_log_fma(double);
double __ieee754_log_sse2(double);

static uint32_t fbits(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return u;
}

static uint64_t dbits(double x) {
  uint64_t u;
  memcpy(&u, &x, 8);
  return u;
}

// All NaNs count as equal: the payload is not part of the contract.
static int fdiff(float a, float b) {
  return (a != a && b != b) ? 0 : fbits(a) != fbits(b);
}

static int ddiff(double a, double b) {
  return (a != a && b != b) ? 0 : dbits(a) != dbits(b);
}

static uint64_t fnv(uint64_t h, float x) {
  uint32_t u = (x != x) ? 0x7fc00000u : fbits(x);
  for (int i = 0; i < 4; i++) {
    h ^= (u >> (8 * i)) & 0xff;
    h *= 1099511628211ULL;
  }
  return h;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: variants <lo> <hi>\n");
    return 2;
  }
  uint64_t lo = strtoull(argv[1], NULL, 16), hi = strtoull(argv[2], NULL, 16);
  uint64_t wide[4] = {0}, narrow[4] = {0};
  uint64_t h[4] = {14695981039346656037ULL, 14695981039346656037ULL,
                   14695981039346656037ULL, 14695981039346656037ULL};
  uint32_t first[4] = {0};
  for (uint64_t u = lo; u < hi; u++) {
    uint32_t b = (uint32_t)u;
    float x;
    memcpy(&x, &b, 4);

    float ef = __expf_fma(x), es = __expf_sse2(x);
    if (fdiff(ef, es)) {
      if (!narrow[0]) first[0] = b;
      wide[0]++, narrow[0]++;
    }
    h[0] = fnv(h[0], es);

    float lf = __logf_fma(x), ls = __logf_sse2(x);
    if (fdiff(lf, ls)) {
      if (!narrow[1]) first[1] = b;
      wide[1]++, narrow[1]++;
    }
    h[1] = fnv(h[1], ls);

    double df = __ieee754_exp_fma((double)x), ds = __ieee754_exp_sse2((double)x);
    wide[2] += ddiff(df, ds);
    if (fdiff((float)df, (float)ds)) {
      if (!narrow[2]) first[2] = b;
      narrow[2]++;
    }
    h[2] = fnv(h[2], (float)ds);

    double arg = x + 1e-5;
    df = __ieee754_log_fma(arg), ds = __ieee754_log_sse2(arg);
    wide[3] += ddiff(df, ds);
    if (fdiff((float)df, (float)ds)) {
      if (!narrow[3]) first[3] = b;
      narrow[3]++;
    }
    h[3] = fnv(h[3], (float)ds);
  }
  const char* names[4] = {"expf", "logf", "exp", "log"};
  for (int i = 0; i < 4; i++) {
    printf("%s\t%08" PRIx64 "\t%09" PRIx64 "\t%" PRIu64 "\t%" PRIu64 "\t%08x\t%016" PRIx64 "\n",
           names[i], lo, hi, wide[i], narrow[i], first[i], h[i]);
  }
  return 0;
}
