// Fingerprints of the maths routines as a program on this machine really
// gets them: the public expf, exp and log of the C library, in the forms
// fastText uses.
//
// variants.c compares glibc's two x86-64 builds by calling them by their
// internal names. This program calls the public names, so it shows which
// results the reference binary computes on the machine it runs on: glibc
// picks the build when the program starts, on x86-64 by CPU features, and on
// arm64 there is a single build that uses fused multiply-add.
//
// The output has the columns of variants.c (the counts are zero), so the
// fingerprints can be compared with results/libm-variants-n4.tsv, which holds
// those of the x86-64 build without FMA: the one the Go port follows.
//
// build: cc -O2 -ffp-contract=off -static public.c -lm -o libm-public
// usage: libm-public <lo> <hi>      (hex, e.g. 0 100000000)
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv(uint64_t h, float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  if (x != x) u = 0x7fc00000u;
  for (int i = 0; i < 4; i++) {
    h ^= (u >> (8 * i)) & 0xff;
    h *= 1099511628211ULL;
  }
  return h;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: libm-public <lo> <hi>\n");
    return 2;
  }
  uint64_t lo = strtoull(argv[1], NULL, 16), hi = strtoull(argv[2], NULL, 16);
  uint64_t h[3] = {14695981039346656037ULL, 14695981039346656037ULL, 14695981039346656037ULL};
  // volatile: the calls must reach the library, not be folded by the compiler.
  volatile float vx;
  for (uint64_t u = lo; u < hi; u++) {
    uint32_t b = (uint32_t)u;
    float x;
    memcpy(&x, &b, 4);
    vx = x;
    h[0] = fnv(h[0], expf(vx));
    h[1] = fnv(h[1], (float)exp((double)vx));
    double arg = vx + 1e-5;
    h[2] = fnv(h[2], (float)log(arg));
  }
  const char* names[3] = {"expf", "exp", "log"};
  for (int i = 0; i < 3; i++) {
    printf("%s\t%08" PRIx64 "\t%09" PRIx64 "\t0\t0\t00000000\t%016" PRIx64 "\n", names[i], lo, hi, h[i]);
  }
  return 0;
}
