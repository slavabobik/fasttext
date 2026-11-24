// ftbench: prediction speed of fastText itself, measured the way a caller of
// the library sees it. Every text goes through the same calls the Python
// package makes: a string stream over text + "\n" and FastText::predictLine.
//
// fasttext/cmd/ftbench is the same program for the Go port; the two print the
// same columns.
//
// usage: ftbench <model> <corpus> <k> <threads> <seconds>
//
// Output, tab-separated: tool, model, corpus, k, threads, texts per pass,
// bytes per pass, predictions returned in one pass (a check that both tools
// did the same work), load time in ms, ns per text (per thread), texts per
// second (all threads), peak memory in MB.
#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "fasttext.h"

using Clock = std::chrono::steady_clock;

static double seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

static const char* base(const char* p) {
  const char* b = p;
  for (const char* c = p; *c; c++) {
    if (*c == '/') b = c + 1;
  }
  return b;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: ftbench <model> <corpus> <k> <threads> <seconds>\n");
    return 2;
  }
  int k = std::atoi(argv[3]);
  int threads = std::atoi(argv[4]);
  double budget = std::atof(argv[5]);

  auto t0 = Clock::now();
  fasttext::FastText ft;
  ft.loadModel(argv[1]);
  double loadMs = seconds(t0, Clock::now()) * 1e3;

  std::vector<std::string> texts;
  int64_t bytes = 0;
  {
    std::ifstream in(argv[2]);
    std::string line;
    while (std::getline(in, line)) {
      bytes += line.size();
      line += '\n';
      texts.push_back(line);
    }
  }
  if (texts.empty()) {
    std::fprintf(stderr, "ftbench: empty corpus\n");
    return 1;
  }

  // One pass to warm up and to count the predictions.
  int64_t returned = 0;
  {
    std::vector<std::pair<fasttext::real, std::string>> preds;
    for (const std::string& text : texts) {
      std::istringstream in(text);
      preds.clear();
      ft.predictLine(in, preds, k, 0.0);
      returned += preds.size();
    }
  }

  std::vector<double> rate(threads);
  std::vector<std::thread> pool;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  for (int t = 0; t < threads; t++) {
    pool.emplace_back([&, t]() {
      std::vector<std::pair<fasttext::real, std::string>> preds;
      ready++;
      while (!go) {
      }
      auto start = Clock::now();
      int64_t n = 0;
      double elapsed = 0;
      do {
        for (const std::string& text : texts) {
          std::istringstream in(text);
          preds.clear();
          ft.predictLine(in, preds, k, 0.0);
        }
        n += texts.size();
        elapsed = seconds(start, Clock::now());
      } while (elapsed < budget);
      rate[t] = n / elapsed;
    });
  }
  while (ready < threads) {
  }
  go = true;
  for (auto& th : pool) th.join();

  double total = 0;
  for (double r : rate) total += r;
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  double rssMb = ru.ru_maxrss / 1048576.0;
#else
  double rssMb = ru.ru_maxrss / 1024.0;
#endif
  std::printf(
      "%s\t%s\t%s\t%d\t%d\t%zu\t%" PRId64 "\t%" PRId64 "\t%.1f\t%.0f\t%.0f\t%.1f\n",
      "fasttext", base(argv[1]), base(argv[2]), k, threads, texts.size(), bytes, returned,
      loadMs, 1e9 * threads / total, total, rssMb);
  return 0;
}
