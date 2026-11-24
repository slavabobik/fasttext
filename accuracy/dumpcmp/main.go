// dumpcmp compares two refdump outputs record by record and layer by layer.
//
// usage: dumpcmp [-label text] [-tsv] [-show n] <dump A> <dump B>
//
// A dump is a file or "-" for standard input. A is the baseline.
//
// For every layer it counts the records that differ: ids, hidden vector, and
// for each k the labels, the score bits and the probability bits. It also
// reports the largest probability difference, and for records whose top label
// differs, how close the top two probabilities of A were.
//
// The exit status is 0 when the dumps are identical, 1 when they differ and 2
// on a usage or format error.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
)

type pred struct {
	label int32
	score uint32
	prob  uint32
}

type record struct {
	ids    string
	hidden string
	ks     []string // "k5"
	preds  [][]pred
}

type kStats struct {
	name        string
	countDiff   int64 // number of predictions differs
	top1Diff    int64 // best label differs
	orderDiff   int64 // same number of predictions, some label differs
	scoreDiff   int64 // records with a differing score, same labels
	probDiff    int64 // records with a differing probability, same labels
	maxProbAbs  float64
	maxProbUlp  int64
	maxScoreUlp int64
	// For records whose best label differs: the largest gap in A between its
	// best and second-best probability. Flips with a large gap are real
	// disagreements; flips with a gap near zero are ties.
	maxFlipGap float64
	flipNoGap  int64 // flips where A had fewer than two predictions
	tiesA      int64 // records where A's top two probabilities are bit-equal
}

type stats struct {
	records     int64
	headerDiff  []string
	idsDiff     int64
	hiddenDiff  int64
	hiddenUlp   int64
	ks          []*kStats
	examples    []string
	maxExamples int
}

func (s *stats) example(n int64, format string, args ...any) {
	if len(s.examples) < s.maxExamples {
		s.examples = append(s.examples, fmt.Sprintf("record %d: ", n)+fmt.Sprintf(format, args...))
	}
}

func parseHex32(s string) (uint32, error) {
	v, err := strconv.ParseUint(s, 16, 32)
	return uint32(v), err
}

func parseRecord(line string, r *record) error {
	fields := strings.Split(line, "\t")
	if len(fields) < 2 || !strings.HasPrefix(fields[0], "ids=") || !strings.HasPrefix(fields[1], "h=") {
		return fmt.Errorf("not a record: %.60q", line)
	}
	r.ids = fields[0]
	r.hidden = fields[1][2:]
	r.ks = r.ks[:0]
	r.preds = r.preds[:0]
	for _, f := range fields[2:] {
		name, list, ok := strings.Cut(f, "=")
		if !ok {
			return fmt.Errorf("bad field %.40q", f)
		}
		r.ks = append(r.ks, name)
		var ps []pred
		for len(list) > 0 {
			var item string
			item, list, _ = strings.Cut(list, ",")
			parts := strings.Split(item, ":")
			if len(parts) != 3 {
				return fmt.Errorf("bad prediction %q", item)
			}
			label, err := strconv.ParseInt(parts[0], 10, 32)
			if err != nil {
				return err
			}
			score, err := parseHex32(parts[1])
			if err != nil {
				return err
			}
			prob, err := parseHex32(parts[2])
			if err != nil {
				return err
			}
			ps = append(ps, pred{int32(label), score, prob})
		}
		r.preds = append(r.preds, ps)
	}
	return nil
}

// ulpDistance is the number of representable float32 values between a and b.
// NaNs and opposite infinities count as the maximum distance.
func ulpDistance(a, b uint32) int64 {
	fa, fb := math.Float32frombits(a), math.Float32frombits(b)
	if fa != fa || fb != fb {
		if a == b {
			return 0
		}
		return math.MaxInt32
	}
	order := func(u uint32) int64 {
		if u&0x80000000 != 0 {
			return -int64(u & 0x7fffffff)
		}
		return int64(u)
	}
	d := order(a) - order(b)
	if d < 0 {
		d = -d
	}
	return d
}

func compareHidden(a, b string) (int64, error) {
	if len(a) != len(b) || len(a)%8 != 0 {
		return math.MaxInt32, nil
	}
	var worst int64
	for i := 0; i < len(a); i += 8 {
		if a[i:i+8] == b[i:i+8] {
			continue
		}
		x, err := parseHex32(a[i : i+8])
		if err != nil {
			return 0, err
		}
		y, err := parseHex32(b[i : i+8])
		if err != nil {
			return 0, err
		}
		// +0 and -0 are different bit patterns at distance 0; report 1.
		worst = max(worst, max(ulpDistance(x, y), 1))
	}
	return worst, nil
}

func compareK(st *kStats, s *stats, n int64, a, b []pred) {
	if len(a) >= 2 && a[0].prob == a[1].prob {
		st.tiesA++
	}
	if len(a) != len(b) {
		st.countDiff++
		s.example(n, "%s: %d predictions vs %d", st.name, len(a), len(b))
	}
	if len(a) > 0 && len(b) > 0 && a[0].label != b[0].label {
		st.top1Diff++
		if len(a) >= 2 {
			gap := float64(math.Float32frombits(a[0].prob)) - float64(math.Float32frombits(a[1].prob))
			st.maxFlipGap = max(st.maxFlipGap, gap)
			s.example(n, "%s: best label %d vs %d, gap in A %.3g", st.name, a[0].label, b[0].label, gap)
		} else {
			st.flipNoGap++
			s.example(n, "%s: best label %d vs %d", st.name, a[0].label, b[0].label)
		}
	}
	sameLabels := len(a) == len(b)
	if sameLabels {
		for i := range a {
			if a[i].label != b[i].label {
				sameLabels = false
				break
			}
		}
		if !sameLabels {
			st.orderDiff++
		}
	}
	// Compare values label by label, so a swap of two near-equal labels is
	// measured as a small difference and not as two large ones.
	var byLabel map[int32]pred
	if !sameLabels {
		byLabel = make(map[int32]pred, len(b))
		for _, p := range b {
			byLabel[p.label] = p
		}
	}
	scoreDiff, probDiff := false, false
	for i, p := range a {
		var q pred
		if sameLabels {
			q = b[i]
		} else {
			var ok bool
			if q, ok = byLabel[p.label]; !ok {
				continue
			}
		}
		if p.score != q.score {
			scoreDiff = true
			st.maxScoreUlp = max(st.maxScoreUlp, ulpDistance(p.score, q.score))
		}
		if p.prob != q.prob {
			probDiff = true
			st.maxProbUlp = max(st.maxProbUlp, ulpDistance(p.prob, q.prob))
			d := math.Abs(float64(math.Float32frombits(p.prob)) - float64(math.Float32frombits(q.prob)))
			if d == d {
				st.maxProbAbs = max(st.maxProbAbs, d)
			}
		}
	}
	if scoreDiff {
		st.scoreDiff++
	}
	if probDiff {
		st.probDiff++
		if !scoreDiff {
			s.example(n, "%s: probability differs with equal scores", st.name)
		}
	}
}

func open(name string) (io.ReadCloser, error) {
	if name == "-" {
		return os.Stdin, nil
	}
	return os.Open(name)
}

func run(pathA, pathB string, show int) (*stats, error) {
	fa, err := open(pathA)
	if err != nil {
		return nil, err
	}
	defer fa.Close()
	fb, err := open(pathB)
	if err != nil {
		return nil, err
	}
	defer fb.Close()

	const maxLine = 64 << 20
	sa := bufio.NewScanner(fa)
	sa.Buffer(make([]byte, 1<<20), maxLine)
	sb := bufio.NewScanner(fb)
	sb.Buffer(make([]byte, 1<<20), maxLine)

	s := &stats{maxExamples: show}
	var ra, rb record
	for {
		okA, okB := sa.Scan(), sb.Scan()
		if !okA || !okB {
			if err := sa.Err(); err != nil {
				return nil, fmt.Errorf("%s: %w", pathA, err)
			}
			if err := sb.Err(); err != nil {
				return nil, fmt.Errorf("%s: %w", pathB, err)
			}
			if okA != okB {
				return nil, fmt.Errorf("dumps have different lengths after %d records", s.records)
			}
			break
		}
		la, lb := sa.Text(), sb.Text()
		if strings.HasPrefix(la, "#") || strings.HasPrefix(lb, "#") {
			if la != lb {
				tag, _, _ := strings.Cut(la, " ")
				s.headerDiff = append(s.headerDiff, tag)
			}
			continue
		}
		n := s.records
		s.records++
		if la == lb {
			// Identical records still count towards the tie statistics.
			if err := parseRecord(la, &ra); err != nil {
				return nil, err
			}
			s.ensureKs(ra.ks)
			for i, ps := range ra.preds {
				if len(ps) >= 2 && ps[0].prob == ps[1].prob {
					s.ks[i].tiesA++
				}
			}
			continue
		}
		if err := parseRecord(la, &ra); err != nil {
			return nil, fmt.Errorf("%s: %w", pathA, err)
		}
		if err := parseRecord(lb, &rb); err != nil {
			return nil, fmt.Errorf("%s: %w", pathB, err)
		}
		if len(ra.ks) != len(rb.ks) {
			return nil, fmt.Errorf("record %d: different k lists", n)
		}
		s.ensureKs(ra.ks)
		if ra.ids != rb.ids {
			s.idsDiff++
			s.example(n, "ids %s vs %s", ra.ids, rb.ids)
		}
		if ra.hidden != rb.hidden {
			s.hiddenDiff++
			ulp, err := compareHidden(ra.hidden, rb.hidden)
			if err != nil {
				return nil, err
			}
			s.hiddenUlp = max(s.hiddenUlp, ulp)
		}
		for i := range ra.preds {
			compareK(s.ks[i], s, n, ra.preds[i], rb.preds[i])
		}
	}
	return s, nil
}

func (s *stats) ensureKs(names []string) {
	for len(s.ks) < len(names) {
		s.ks = append(s.ks, &kStats{name: names[len(s.ks)]})
	}
}

func (s *stats) identical() bool {
	if len(s.headerDiff) > 0 || s.idsDiff > 0 || s.hiddenDiff > 0 {
		return false
	}
	for _, k := range s.ks {
		if k.countDiff+k.top1Diff+k.orderDiff+k.scoreDiff+k.probDiff > 0 {
			return false
		}
	}
	return true
}

func main() {
	label := flag.String("label", "", "text to print in front of the result")
	tsv := flag.Bool("tsv", false, "write one tab-separated line per k instead of a report")
	show := flag.Int("show", 5, "number of differing records to describe")
	flag.Parse()
	if flag.NArg() != 2 {
		fmt.Fprintln(os.Stderr, "usage: dumpcmp [-label text] [-tsv] [-show n] <dump A> <dump B>")
		os.Exit(2)
	}
	s, err := run(flag.Arg(0), flag.Arg(1), *show)
	if err != nil {
		fmt.Fprintln(os.Stderr, "dumpcmp:", err)
		os.Exit(2)
	}

	if *tsv {
		// label, k, records, header, ids, hidden, hidden ulp, count, top1,
		// order, score, score ulp, prob, prob ulp, max prob diff, max flip
		// gap, ties in A
		for _, k := range s.ks {
			fmt.Printf("%s\t%s\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3g\t%.3g\t%d\n",
				*label, k.name, s.records, strings.Join(s.headerDiff, ","), s.idsDiff, s.hiddenDiff, s.hiddenUlp,
				k.countDiff, k.top1Diff, k.orderDiff, k.scoreDiff, k.maxScoreUlp, k.probDiff, k.maxProbUlp,
				k.maxProbAbs, k.maxFlipGap, k.tiesA)
		}
	} else {
		if *label != "" {
			fmt.Println(*label)
		}
		fmt.Printf("  records            %d\n", s.records)
		if len(s.headerDiff) > 0 {
			fmt.Printf("  header differs     %s\n", strings.Join(s.headerDiff, " "))
		}
		fmt.Printf("  ids differ         %d\n", s.idsDiff)
		fmt.Printf("  hidden differs     %d (largest %d ulp)\n", s.hiddenDiff, s.hiddenUlp)
		for _, k := range s.ks {
			fmt.Printf("  %s\n", k.name)
			fmt.Printf("    best label differs   %d (largest gap in A %.3g)\n", k.top1Diff, k.maxFlipGap)
			fmt.Printf("    label order differs  %d\n", k.orderDiff)
			fmt.Printf("    count differs        %d\n", k.countDiff)
			fmt.Printf("    scores differ        %d (largest %d ulp)\n", k.scoreDiff, k.maxScoreUlp)
			fmt.Printf("    probs differ         %d (largest %d ulp, %.3g absolute)\n", k.probDiff, k.maxProbUlp, k.maxProbAbs)
			fmt.Printf("    exact ties in A      %d\n", k.tiesA)
		}
		for _, e := range s.examples {
			fmt.Println("  " + e)
		}
	}
	if !s.identical() {
		os.Exit(1)
	}
}
