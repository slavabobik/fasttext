// refdump: reference dump of fastText predictions, layer by layer.
//
// Links against the unmodified fastText sources. For every input record it
// writes the intermediate values a port has to reproduce, with floats as raw
// IEEE-754 bits, so two builds (or fastText and a port) can be compared
// exactly.
//
// usage: refdump [--ids] [--stream] [--info] <model> [<k>[,<k>...]] [<threshold>]
//
//   <k>        comma-separated list of k values; -1 means all labels.
//              Default "1".
//   --ids      write every word and n-gram id instead of a count and a hash.
//   --stream   read stdin the way the fastText CLI does (one predictLine call
//              after another on the same stream). Without it every input line
//              is one record, passed as `line + "\n"`, which is what the
//              Python package does.
//   --info     write the model header, dictionary and matrix digests and exit.
//
// Output, one line per record, tab-separated:
//
//   ids=<n>:<fnv64 of the ids>      or, with --ids,   ids=<n>:<id>,<id>,...
//   h=<8 hex digits per dimension>  the hidden vector; empty when n is 0
//   k<k>=<label>:<score>:<prob>,... one field per requested k, best first.
//                                   <label> is the label index, <score> the
//                                   log-probability bits, <prob> the bits of
//                                   the value predictLine returns.
//
// Lines that start with '#' describe the model.

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "fasttext.h"

namespace {

using fasttext::real;

uint32_t bits(real x) {
  uint32_t u;
  static_assert(sizeof(u) == sizeof(x), "real must be 32 bits");
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

struct Fnv64 {
  uint64_t h = 14695981039346656037ULL;
  void add(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; i++) {
      h ^= b[i];
      h *= 1099511628211ULL;
    }
  }
};

const char* lossName(fasttext::loss_name l) {
  switch (l) {
    case fasttext::loss_name::hs:
      return "hs";
    case fasttext::loss_name::ns:
      return "ns";
    case fasttext::loss_name::softmax:
      return "softmax";
    case fasttext::loss_name::ova:
      return "ova";
  }
  return "?";
}

// FastText keeps the dictionary and model protected. The subclass only
// exposes them; it changes no behaviour.
class Ref : public fasttext::FastText {
 public:
  void header(FILE* out) const {
    std::fprintf(
        out,
        "#args dim=%d ws=%d epoch=%d minCount=%d neg=%d wordNgrams=%d "
        "loss=%s model=%d bucket=%d minn=%d maxn=%d lrUpdateRate=%d t=%a "
        "qout=%d\n",
        args_->dim, args_->ws, args_->epoch, args_->minCount, args_->neg,
        args_->wordNgrams, lossName(args_->loss), static_cast<int>(args_->model),
        args_->bucket, args_->minn, args_->maxn, args_->lrUpdateRate,
        args_->t, static_cast<int>(args_->qout));
    std::fprintf(
        out, "#dict nwords=%d nlabels=%d ntokens=%" PRId64 " quant=%d\n",
        dict_->nwords(), dict_->nlabels(), dict_->ntokens(),
        static_cast<int>(quant_));
    std::fprintf(out, "#labels");
    for (int32_t i = 0; i < dict_->nlabels(); i++) {
      std::fprintf(out, " %s", dict_->getLabel(i).c_str());
    }
    std::fprintf(out, "\n");
  }

  void info(FILE* out) const {
    header(out);
    std::vector<int64_t> wc = dict_->getCounts(fasttext::entry_type::word);
    std::vector<int64_t> lc = dict_->getCounts(fasttext::entry_type::label);
    for (int32_t i = 0; i < dict_->nwords(); i++) {
      std::fprintf(
          out, "word\t%d\t%" PRId64 "\t%s\n", i, wc[i],
          dict_->getWord(i).c_str());
    }
    for (int32_t i = 0; i < dict_->nlabels(); i++) {
      std::fprintf(
          out, "label\t%d\t%" PRId64 "\t%s\n", i, lc[i],
          dict_->getLabel(i).c_str());
    }
    if (!quant_) {
      digest(out, "input", *getInputMatrix());
    }
    if (!quant_ || !args_->qout) {
      // The output matrix stays dense unless the model was quantised with
      // -qout.
      auto dense = std::dynamic_pointer_cast<fasttext::DenseMatrix>(output_);
      if (dense) {
        digest(out, "output", *dense);
      }
    }
  }

  static void digest(FILE* out, const char* name, const fasttext::DenseMatrix& m) {
    Fnv64 f;
    f.add(m.data(), sizeof(real) * m.rows() * m.cols());
    std::fprintf(
        out, "matrix\t%s\t%" PRId64 "\t%" PRId64 "\t%016" PRIx64 "\n", name,
        m.rows(), m.cols(), f.h);
  }

  // One record. `in` is positioned at the start of the record. Returns false
  // at end of input, like predictLine.
  bool record(
      std::istream& in,
      const std::vector<int32_t>& ks,
      real threshold,
      bool fullIds,
      std::string& line,
      std::vector<std::pair<int32_t, uint32_t>>* firstK = nullptr) const {
    if (in.peek() == EOF) {
      return false;
    }
    std::vector<int32_t> words, labels;
    dict_->getLine(in, words, labels);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "ids=%zu:", words.size());
    line += buf;
    if (fullIds) {
      for (size_t i = 0; i < words.size(); i++) {
        std::snprintf(buf, sizeof(buf), i ? ",%d" : "%d", words[i]);
        line += buf;
      }
    } else {
      Fnv64 f;
      f.add(words.data(), sizeof(int32_t) * words.size());
      std::snprintf(buf, sizeof(buf), "%016" PRIx64, f.h);
      line += buf;
    }

    line += "\th=";
    bool first = true;
    for (int32_t k : ks) {
      // Same calls as FastText::predict, with a fresh state per call.
      fasttext::Predictions preds;
      fasttext::Model::State state(args_->dim, dict_->nlabels(), 0);
      if (!words.empty()) {
        model_->predict(words, k, threshold, preds, state);
      }
      if (first) {
        if (!words.empty()) {
          for (int64_t i = 0; i < state.hidden.size(); i++) {
            std::snprintf(buf, sizeof(buf), "%08x", bits(state.hidden[i]));
            line += buf;
          }
        }
      }
      std::snprintf(buf, sizeof(buf), "\tk%d=", k);
      line += buf;
      for (size_t i = 0; i < preds.size(); i++) {
        // Same expression as FastText::predictLine.
        real prob = std::exp(preds[i].first);
        std::snprintf(
            buf, sizeof(buf), i ? ",%d:%08x:%08x" : "%d:%08x:%08x",
            preds[i].second, bits(preds[i].first), bits(prob));
        line += buf;
        if (first && firstK) {
          firstK->emplace_back(preds[i].second, bits(prob));
        }
      }
      first = false;
    }
    return true;
  }

  // Runs the public predictLine on the same text and checks that it returns
  // the labels and probabilities record() wrote for the first k. Guards
  // against the tool drifting from the real API.
  void selfCheck(
      const std::string& text,
      int32_t k,
      real threshold,
      const std::vector<std::pair<int32_t, uint32_t>>& wrote,
      int64_t recordNo) const {
    std::istringstream in(text);
    std::vector<std::pair<real, std::string>> preds;
    predictLine(in, preds, k, threshold);
    bool ok = preds.size() == wrote.size();
    for (size_t i = 0; ok && i < preds.size(); i++) {
      int32_t id = dict_->getId(preds[i].second) - dict_->nwords();
      ok = id == wrote[i].first && bits(preds[i].first) == wrote[i].second;
    }
    if (!ok) {
      std::fprintf(
          stderr, "refdump: self-check against predictLine failed at record %" PRId64 "\n",
          recordNo);
      std::exit(3);
    }
  }
};

void usage() {
  std::fprintf(
      stderr,
      "usage: refdump [--ids] [--stream] [--info] <model> [<k>[,<k>...]] "
      "[<threshold>]\n");
  std::exit(2);
}

} // namespace

int main(int argc, char** argv) {
  bool fullIds = false, stream = false, info = false;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--ids") {
      fullIds = true;
    } else if (a == "--stream") {
      stream = true;
    } else if (a == "--info") {
      info = true;
    } else {
      pos.push_back(a);
    }
  }
  if (pos.empty() || pos.size() > 3) {
    usage();
  }
  std::vector<int32_t> ks;
  {
    std::stringstream ss(pos.size() > 1 ? pos[1] : "1");
    std::string item;
    while (std::getline(ss, item, ',')) {
      ks.push_back(std::atoi(item.c_str()));
    }
  }
  real threshold = pos.size() > 2 ? std::strtof(pos[2].c_str(), nullptr) : 0.0;

  Ref ft;
  ft.loadModel(pos[0]);

  if (info) {
    ft.info(stdout);
    return 0;
  }

  ft.header(stdout);
  std::fprintf(stdout, "#run threshold=%a mode=%s k=", threshold, stream ? "stream" : "line");
  for (size_t i = 0; i < ks.size(); i++) {
    std::fprintf(stdout, i ? ",%d" : "%d", ks[i]);
  }
  std::fprintf(stdout, "\n");

  std::ios::sync_with_stdio(false);
  std::string out;
  int64_t n = 0;
  if (stream) {
    while (true) {
      out.clear();
      if (!ft.record(std::cin, ks, threshold, fullIds, out)) {
        break;
      }
      out += '\n';
      std::fwrite(out.data(), 1, out.size(), stdout);
      n++;
    }
  } else {
    std::string text;
    std::vector<std::pair<int32_t, uint32_t>> wrote;
    while (std::getline(std::cin, text)) {
      text += '\n';
      std::istringstream in(text);
      out.clear();
      wrote.clear();
      ft.record(in, ks, threshold, fullIds, out, &wrote);
      ft.selfCheck(text, ks[0], threshold, wrote, n);
      out += '\n';
      std::fwrite(out.data(), 1, out.size(), stdout);
      n++;
    }
  }
  std::fprintf(stderr, "refdump: %" PRId64 " records\n", n);
  return 0;
}
