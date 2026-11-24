// heapref: fingerprints of the order in which the C++ standard library's heap
// routines return candidates with equal scores.
//
// fastText keeps its best-k list with std::push_heap, std::pop_heap and
// std::sort_heap, comparing scores only. With equal scores the result depends
// on the library (libstdc++ and libc++ differ). The Go port repeats
// libstdc++'s moves; its tests replay the cases below and compare the
// fingerprints with accuracy/golden/heap.tsv, written by this program built
// with the reference toolchain.
//
// build: c++ -O2 -std=c++17 heapref.cc -o heapref
// output: n, k, number of distinct scores, fingerprint; tab-separated.
#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

uint64_t state = 20261120;

// Knuth's MMIX generator; the top bits are the output.
uint32_t next() {
  state = state * 6364136223846793005ULL + 1442695040888963407ULL;
  return static_cast<uint32_t>(state >> 33);
}

bool comparePairs(const std::pair<float, int32_t>& l, const std::pair<float, int32_t>& r) {
  return l.first > r.first;
}

} // namespace

int main() {
  const int ns[] = {1, 2, 3, 4, 5, 8, 16, 50, 323, 2000};
  const int ks[] = {1, 2, 3, 5, 10, 100, 1000000};
  const int vs[] = {1, 2, 3, 7, 1000};
  for (int n : ns) {
    for (int k : ks) {
      for (int v : vs) {
        uint64_t h = 14695981039346656037ULL;
        for (int rep = 0; rep < 50; rep++) {
          // The loop of Loss::findKBest.
          std::vector<std::pair<float, int32_t>> heap;
          for (int32_t i = 0; i < n; i++) {
            float score = static_cast<float>(next() % v);
            if (heap.size() == static_cast<size_t>(k) && score < heap.front().first) {
              continue;
            }
            heap.push_back(std::make_pair(score, i));
            std::push_heap(heap.begin(), heap.end(), comparePairs);
            if (heap.size() > static_cast<size_t>(k)) {
              std::pop_heap(heap.begin(), heap.end(), comparePairs);
              heap.pop_back();
            }
          }
          std::sort_heap(heap.begin(), heap.end(), comparePairs);
          for (const auto& p : heap) {
            uint32_t u = static_cast<uint32_t>(p.second);
            for (int b = 0; b < 4; b++) {
              h ^= (u >> (8 * b)) & 0xff;
              h *= 1099511628211ULL;
            }
          }
        }
        std::printf("%d\t%d\t%d\t%016" PRIx64 "\n", n, k, v, h);
      }
    }
  }
  return 0;
}
