package langid_test

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/slavabobik/fasttext/internal/testdata"
	"github.com/slavabobik/fasttext/langid"
)

var update = flag.Bool("update", false, "rewrite testdata/tatoeba-lid176.tsv and docs/LANGUAGES.md")

// tatoebaLanguages maps the ISO 639-3 codes of Tatoeba to the labels of
// lid.176, for every Tatoeba language that lid.176 knows. Where Tatoeba
// splits a language that lid.176 does not (Swahili, Malay), both map to the
// one label.
var tatoebaLanguages = []struct{ tatoeba, code, name string }{
	{"afr", "af", "Afrikaans"},
	{"amh", "am", "Amharic"},
	{"ara", "ar", "Arabic"},
	{"arg", "an", "Aragonese"},
	{"arz", "arz", "Egyptian Arabic"},
	{"asm", "as", "Assamese"},
	{"ast", "ast", "Asturian"},
	{"ava", "av", "Avar"},
	{"aze", "az", "Azerbaijani"},
	{"bak", "ba", "Bashkir"},
	{"bar", "bar", "Bavarian"},
	{"bel", "be", "Belarusian"},
	{"ben", "bn", "Bengali"},
	{"bod", "bo", "Tibetan"},
	{"bos", "bs", "Bosnian"},
	{"bre", "br", "Breton"},
	{"bua", "bxr", "Buryat"},
	{"bul", "bg", "Bulgarian"},
	{"cat", "ca", "Catalan"},
	{"cbk", "cbk", "Chavacano"},
	{"ceb", "ceb", "Cebuano"},
	{"ces", "cs", "Czech"},
	{"che", "ce", "Chechen"},
	{"chv", "cv", "Chuvash"},
	{"ckb", "ckb", "Central Kurdish"},
	{"cmn", "zh", "Chinese (Mandarin)"},
	{"cor", "kw", "Cornish"},
	{"cos", "co", "Corsican"},
	{"cym", "cy", "Welsh"},
	{"dan", "da", "Danish"},
	{"deu", "de", "German"},
	{"div", "dv", "Dhivehi"},
	{"dsb", "dsb", "Lower Sorbian"},
	{"ell", "el", "Greek"},
	{"eng", "en", "English"},
	{"epo", "eo", "Esperanto"},
	{"est", "et", "Estonian"},
	{"eus", "eu", "Basque"},
	{"fin", "fi", "Finnish"},
	{"fra", "fr", "French"},
	{"frr", "frr", "North Frisian"},
	{"fry", "fy", "West Frisian"},
	{"gla", "gd", "Scottish Gaelic"},
	{"gle", "ga", "Irish"},
	{"glg", "gl", "Galician"},
	{"glv", "gv", "Manx"},
	{"grn", "gn", "Guarani"},
	{"gsw", "als", "Swiss German"},
	{"guj", "gu", "Gujarati"},
	{"hat", "ht", "Haitian Creole"},
	{"heb", "he", "Hebrew"},
	{"hin", "hi", "Hindi"},
	{"hrv", "hr", "Croatian"},
	{"hsb", "hsb", "Upper Sorbian"},
	{"hun", "hu", "Hungarian"},
	{"hye", "hy", "Armenian"},
	{"ido", "io", "Ido"},
	{"ile", "ie", "Interlingue"},
	{"ilo", "ilo", "Ilocano"},
	{"ina", "ia", "Interlingua"},
	{"ind", "id", "Indonesian"},
	{"isl", "is", "Icelandic"},
	{"ita", "it", "Italian"},
	{"jav", "jv", "Javanese"},
	{"jbo", "jbo", "Lojban"},
	{"jpn", "ja", "Japanese"},
	{"kan", "kn", "Kannada"},
	{"kat", "ka", "Georgian"},
	{"kaz", "kk", "Kazakh"},
	{"khm", "km", "Khmer"},
	{"kir", "ky", "Kyrgyz"},
	{"kmr", "ku", "Kurdish (Kurmanji)"},
	{"kor", "ko", "Korean"},
	{"kpv", "kv", "Komi"},
	{"krc", "krc", "Karachay-Balkar"},
	{"lao", "lo", "Lao"},
	{"lat", "la", "Latin"},
	{"lez", "lez", "Lezgian"},
	{"lim", "li", "Limburgish"},
	{"lit", "lt", "Lithuanian"},
	{"lmo", "lmo", "Lombard"},
	{"ltz", "lb", "Luxembourgish"},
	{"lvs", "lv", "Latvian"},
	{"mal", "ml", "Malayalam"},
	{"mar", "mr", "Marathi"},
	{"mhr", "mhr", "Meadow Mari"},
	{"min", "min", "Minangkabau"},
	{"mkd", "mk", "Macedonian"},
	{"mlg", "mg", "Malagasy"},
	{"mlt", "mt", "Maltese"},
	{"mon", "mn", "Mongolian"},
	{"mrj", "mrj", "Hill Mari"},
	{"mya", "my", "Burmese"},
	{"myv", "myv", "Erzya"},
	{"nah", "nah", "Nahuatl"},
	{"nap", "nap", "Neapolitan"},
	{"nds", "nds", "Low German"},
	{"new", "new", "Newar"},
	{"nld", "nl", "Dutch"},
	{"nno", "nn", "Norwegian Nynorsk"},
	{"nob", "no", "Norwegian Bokmål"},
	{"npi", "ne", "Nepali"},
	{"oci", "oc", "Occitan"},
	{"ori", "or", "Odia"},
	{"oss", "os", "Ossetian"},
	{"pam", "pam", "Kapampangan"},
	{"pes", "fa", "Persian"},
	{"pms", "pms", "Piedmontese"},
	{"pol", "pl", "Polish"},
	{"por", "pt", "Portuguese"},
	{"que", "qu", "Quechua"},
	{"roh", "rm", "Romansh"},
	{"ron", "ro", "Romanian"},
	{"rue", "rue", "Rusyn"},
	{"rus", "ru", "Russian"},
	{"sah", "sah", "Yakut"},
	{"san", "sa", "Sanskrit"},
	{"sco", "sco", "Scots"},
	{"slk", "sk", "Slovak"},
	{"slv", "sl", "Slovene"},
	{"som", "so", "Somali"},
	{"spa", "es", "Spanish"},
	{"sqi", "sq", "Albanian"},
	{"srp", "sr", "Serbian"},
	{"sun", "su", "Sundanese"},
	{"swc", "sw", "Swahili (Congo)"},
	{"swe", "sv", "Swedish"},
	{"swh", "sw", "Swahili"},
	{"tam", "ta", "Tamil"},
	{"tat", "tt", "Tatar"},
	{"tel", "te", "Telugu"},
	{"tgk", "tg", "Tajik"},
	{"tgl", "tl", "Tagalog"},
	{"tha", "th", "Thai"},
	{"tuk", "tk", "Turkmen"},
	{"tur", "tr", "Turkish"},
	{"tyv", "tyv", "Tuvan"},
	{"uig", "ug", "Uyghur"},
	{"ukr", "uk", "Ukrainian"},
	{"urd", "ur", "Urdu"},
	{"uzb", "uz", "Uzbek"},
	{"vec", "vec", "Venetian"},
	{"vep", "vep", "Veps"},
	{"vie", "vi", "Vietnamese"},
	{"vol", "vo", "Volapük"},
	{"war", "war", "Waray"},
	{"wuu", "wuu", "Wu Chinese"},
	{"xal", "xal", "Kalmyk"},
	{"xmf", "xmf", "Mingrelian"},
	{"yid", "yi", "Yiddish"},
	{"yor", "yo", "Yoruba"},
	{"yue", "yue", "Cantonese"},
	{"zlm", "ms", "Malay"},
	{"zsm", "ms", "Malay (Standard)"},
	{"zza", "diq", "Zazaki"},
}

// tatoebaModels are the models scored, in column order.
var tatoebaModels = []string{"lid.176.bin", "lid.176.ftz"}

type langScore struct {
	tatoeba, code, name string
	texts               int
	correct             []int          // per model
	wrong               map[string]int // wrong answers of the first model
}

type lengthBucket struct {
	name    string
	max     int // characters
	texts   int
	correct []int // per model
}

type tatoebaResult struct {
	langs      []*langScore // most texts first
	buckets    []lengthBucket
	texts      int
	chars      int
	correct    []int // per model
	otherTexts int   // texts in languages lid.176 does not know
	otherLangs int
}

// TestTatoeba runs both lid.176 models over the Tatoeba test sentences
// (accuracy/corpus/tatoeba-test.tsv, Git LFS) and compares the number of
// correct answers per language with testdata/tatoeba-lid176.tsv. The port
// gives the same bits as fastText, so the counts are fastText's own and must
// not change. -update rewrites the file and docs/LANGUAGES.md.
func TestTatoeba(t *testing.T) {
	if testing.Short() {
		t.Skip("-short: 2 × 200,000 predictions")
	}
	path := testdata.Path(t, "corpus/tatoeba-test.tsv")
	detectors := make([]*langid.Detector, len(tatoebaModels))
	for i, m := range tatoebaModels {
		detectors[i] = open(t, m)
	}
	r := scoreTatoeba(t, path, detectors)

	got := tatoebaTSV(r)
	golden := filepath.Join("testdata", "tatoeba-lid176.tsv")
	if *update {
		if err := os.WriteFile(golden, got, 0o644); err != nil {
			t.Fatal(err)
		}
		root, err := testdata.Root()
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "docs", "LANGUAGES.md"), languagesDoc(r), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		gotLines, wantLines := strings.Split(string(got), "\n"), strings.Split(string(want), "\n")
		for i := 0; i < len(gotLines) || i < len(wantLines); i++ {
			var g, w string
			if i < len(gotLines) {
				g = gotLines[i]
			}
			if i < len(wantLines) {
				w = wantLines[i]
			}
			if g != w {
				t.Errorf("line %d:\n got  %s\n want %s", i+1, g, w)
			}
		}
		t.Fatalf("results differ from %s (go test -run TestTatoeba -update rewrites it)", golden)
	}
	for i, m := range tatoebaModels {
		t.Logf("%s: %d of %d texts correct in %d languages (%.2f%%)",
			m, r.correct[i], r.texts, len(r.langs), pct(r.correct[i], r.texts))
	}
}

func scoreTatoeba(t *testing.T, path string, detectors []*langid.Detector) tatoebaResult {
	known := map[string]bool{}
	for _, c := range detectors[0].Languages() {
		known[c] = true
	}
	scores := map[string]*langScore{}
	for _, l := range tatoebaLanguages {
		if !known[l.code] {
			t.Fatalf("%s: lid.176 has no label %q", l.tatoeba, l.code)
		}
		if scores[l.tatoeba] != nil {
			t.Fatalf("%s listed twice", l.tatoeba)
		}
		scores[l.tatoeba] = &langScore{tatoeba: l.tatoeba, code: l.code, name: l.name,
			correct: make([]int, len(detectors)), wrong: map[string]int{}}
	}
	r := tatoebaResult{correct: make([]int, len(detectors))}
	for _, b := range []struct {
		name string
		max  int
	}{{"1 to 20", 20}, {"21 to 40", 40}, {"41 to 80", 80}, {"81 and more", 1 << 30}} {
		r.buckets = append(r.buckets, lengthBucket{name: b.name, max: b.max, correct: make([]int, len(detectors))})
	}
	others := map[string]bool{}

	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		fields := strings.SplitN(sc.Text(), "\t", 3)
		if len(fields) != 3 {
			t.Fatalf("bad line %q", sc.Text())
		}
		s := scores[fields[1]]
		if s == nil {
			r.otherTexts++
			others[fields[1]] = true
			continue
		}
		text := fields[2]
		n := utf8.RuneCountInString(text)
		bk := &r.buckets[0]
		for i := 0; n > r.buckets[i].max; i++ {
			bk = &r.buckets[i+1]
		}
		s.texts++
		bk.texts++
		r.texts++
		r.chars += n
		for i, d := range detectors {
			got, err := d.Detect(text)
			if err != nil {
				got.Code = "(" + err.Error() + ")"
			}
			if got.Code == s.code {
				s.correct[i]++
				bk.correct[i]++
				r.correct[i]++
			} else if i == 0 {
				s.wrong[got.Code]++
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	r.otherLangs = len(others)

	for _, s := range scores {
		if s.texts > 0 {
			r.langs = append(r.langs, s)
		}
	}
	sort.Slice(r.langs, func(i, j int) bool {
		if r.langs[i].texts != r.langs[j].texts {
			return r.langs[i].texts > r.langs[j].texts
		}
		return r.langs[i].tatoeba < r.langs[j].tatoeba
	})
	return r
}

func tatoebaTSV(r tatoebaResult) []byte {
	var b bytes.Buffer
	b.WriteString("tatoeba\tcode\ttexts")
	for _, m := range tatoebaModels {
		b.WriteString("\t" + m)
	}
	b.WriteString("\twrong answers of lid.176.bin\n")
	for _, s := range r.langs {
		fmt.Fprintf(&b, "%s\t%s\t%d", s.tatoeba, s.code, s.texts)
		for _, c := range s.correct {
			fmt.Fprintf(&b, "\t%d", c)
		}
		fmt.Fprintf(&b, "\t%s\n", topWrong(s.wrong, 3))
	}
	return b.Bytes()
}

// topWrong lists the n most frequent wrong answers as "code count".
func topWrong(wrong map[string]int, n int) string {
	type kv struct {
		code  string
		count int
	}
	var all []kv
	for c, k := range wrong {
		all = append(all, kv{c, k})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].count != all[j].count {
			return all[i].count > all[j].count
		}
		return all[i].code < all[j].code
	})
	var parts []string
	for i := 0; i < len(all) && i < n; i++ {
		parts = append(parts, fmt.Sprintf("%s %d", all[i].code, all[i].count))
	}
	return strings.Join(parts, ", ")
}

// minTexts is the number of texts a language needs for its own row in
// docs/LANGUAGES.md.
const minTexts = 20

func languagesDoc(r tatoebaResult) []byte {
	var b bytes.Buffer
	rows := 0
	macro := make([]float64, len(tatoebaModels))
	for _, s := range r.langs {
		if s.texts >= minTexts {
			rows++
			for i := range macro {
				macro[i] += pct(s.correct[i], s.texts)
			}
		}
	}
	code := func(s string) string { return "`" + s + "`" }

	fmt.Fprintf(&b, `# Accuracy per language

How often the lid.176 models name the right language for a sentence from [Tatoeba](https://tatoeba.org), for each of the %d Tatoeba languages that lid.176 knows. The Go port returns the same bits as fastText, so these are fastText's own numbers: any program that runs these models, in Go or C++, gets the same answers.

- **Texts:** the %s sentences of %s (every 64th sentence of a Tatoeba export) in a language that lid.176 knows, %.0f characters long on average. The other %s sentences, in %d languages that lid.176 does not know, are left out.
- **Correct** means the top answer of %s is the language Tatoeba gives. No threshold is applied.
- **Caution:** fastText trained lid.176 partly on Tatoeba, so many of these sentences, or ones close to them, were seen in training. Expect lower numbers on other text. The numbers are best used to compare languages and the two models.
- **Reproduce:** %s. The test fails if any count changes, and %s rewrites this page and [%s](../langid/testdata/tatoeba-lid176.tsv), which also lists the languages with fewer than %d texts.

## Summary

| | lid.176.bin | lid.176.ftz |
|---|---:|---:|
| Texts correct, of %s | %s | %s |
| Mean accuracy of the %d languages with %d or more texts | %.1f%% | %.1f%% |

## By length

A few words carry little evidence, and close languages share most of them, so short texts are the hard case.

| Characters | Texts | lid.176.bin | lid.176.ftz |
|---|---:|---:|---:|
`, len(r.langs),
		thousands(r.texts), code("accuracy/corpus/tatoeba-test.tsv"), float64(r.chars)/float64(r.texts),
		thousands(r.otherTexts), r.otherLangs,
		code("Detector.Detect"),
		code("git lfs pull && go test ./langid -run TestTatoeba -v"), code("-update"),
		code("langid/testdata/tatoeba-lid176.tsv"), minTexts,
		thousands(r.texts), pctString(r.correct[0], r.texts), pctString(r.correct[1], r.texts),
		rows, minTexts, macro[0]/float64(rows), macro[1]/float64(rows))
	for _, bk := range r.buckets {
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", bk.name, thousands(bk.texts),
			pctString(bk.correct[0], bk.texts), pctString(bk.correct[1], bk.texts))
	}

	fmt.Fprintf(&b, `
## By language

Languages with %d or more texts, most texts first. The last column lists what lid.176.bin answered instead, the three most frequent wrong answers with their counts.

| Language | Tatoeba | Code | Texts | lid.176.bin | lid.176.ftz | Wrong answers of lid.176.bin |
|---|---|---|---:|---:|---:|---|
`, minTexts)
	for _, s := range r.langs {
		if s.texts < minTexts {
			continue
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s | %s |\n", s.name, code(s.tatoeba), code(s.code),
			thousands(s.texts), pctString(s.correct[0], s.texts), pctString(s.correct[1], s.texts),
			topWrong(s.wrong, 3))
	}
	return b.Bytes()
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

func pctString(a, b int) string {
	if b == 0 {
		return "–"
	}
	return fmt.Sprintf("%.1f%%", pct(a, b))
}

func thousands(n int) string {
	s := fmt.Sprint(n)
	for i := len(s) - 3; i > 0; i -= 3 {
		s = s[:i] + "," + s[i:]
	}
	return s
}
