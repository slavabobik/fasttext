package fasttext

import (
	"fmt"
	"math"
	"strings"
)

const (
	eosToken    = "</s>"
	labelPrefix = "__label__"

	typeWord  = 0
	typeLabel = 1
)

// isSeparator marks the bytes fastText's readWord splits on.
var isSeparator = [256]bool{' ': true, '\n': true, '\r': true, '\t': true, '\v': true, '\f': true, 0: true}

// dictionary is fastText's Dictionary: the vocabulary, the labels and the
// rules that turn a line of text into row numbers of the input matrix.
type dictionary struct {
	entries []string // words first, then labels
	counts  []int64
	types   []int8

	nwords       int32
	nlabels      int32
	ntokens      int64
	pruneidxSize int64 // -1: not pruned; 0: no n-gram rows kept
	prune        pruneTable

	// Open-addressing table from token to entry number, keyed by the
	// fastText hash of the token.
	table []int32
	mask  uint32

	// Character n-gram settings. minn and maxn are compared as unsigned
	// 64-bit values, as the C++ code does (int against size_t).
	minn       uint64
	maxn       uint64
	useNgrams  bool // maxn > 0
	bucket     uint32
	bucket64   uint64
	wordNgrams int

	// Precomputed ids for every vocabulary word when useNgrams is set: the
	// word's own id followed by the ids of its character n-grams. The ids of
	// word i are subIDs[subOff[i]:subOff[i+1]].
	subOff []uint32
	subIDs []int32
}

// hash is fastText's Dictionary::hash: 32-bit FNV-1a in which every byte is
// sign-extended before the xor. The sign extension is a bug upstream keeps
// for compatibility with published models.
func hash(s string) uint32 {
	h := uint32(2166136261)
	for i := 0; i < len(s); i++ {
		h = (h ^ uint32(int8(s[i]))) * 16777619
	}
	return h
}

func (d *dictionary) load(r *reader, a *args) error {
	size := r.int32()
	d.nwords = r.int32()
	d.nlabels = r.int32()
	d.ntokens = r.int64()
	d.pruneidxSize = r.int64()
	if r.err != nil {
		return r.err
	}
	if size < 0 || d.nwords < 0 || d.nlabels < 0 || int64(d.nwords)+int64(d.nlabels) != int64(size) {
		return fmt.Errorf("fasttext: invalid dictionary header (size %d, words %d, labels %d)", size, d.nwords, d.nlabels)
	}
	if d.pruneidxSize > math.MaxInt32 {
		return fmt.Errorf("fasttext: invalid pruned index size %d", d.pruneidxSize)
	}

	// An entry takes at least ten bytes in the file, a pruned index pair
	// eight: sizes that cannot fit are refused before anything is allocated.
	if int64(size) > r.remaining()/10 || d.pruneidxSize > r.remaining()/8 {
		return fmt.Errorf("fasttext: dictionary of %d entries does not fit in the input", size)
	}

	// When the input size is unknown, the slices grow with the entries that
	// arrive instead of trusting size.
	hint := r.capHint(int(size), 1<<16)
	d.entries = make([]string, 0, hint)
	d.counts = make([]int64, 0, hint)
	d.types = make([]int8, 0, hint)
	for i := int32(0); i < size; i++ {
		entry := r.cstring()
		count := r.int64()
		typ := r.int8()
		if r.err != nil {
			return r.err
		}
		// fastText sorts the dictionary: words first, then labels.
		want := int8(typeWord)
		if i >= d.nwords {
			want = typeLabel
		}
		if typ != want {
			return fmt.Errorf("fasttext: dictionary entry %d has type %d, want %d", i, typ, want)
		}
		d.entries = append(d.entries, entry)
		d.counts = append(d.counts, count)
		d.types = append(d.types, typ)
	}

	if d.pruneidxSize > 0 {
		// The pairs are read before the table is sized, for the same reason.
		pairs := make([][2]int32, 0, r.capHint(int(d.pruneidxSize), 1<<16))
		for i := int64(0); i < d.pruneidxSize; i++ {
			first := r.int32()
			second := r.int32()
			if r.err != nil {
				return r.err
			}
			if second < 0 {
				return fmt.Errorf("fasttext: invalid pruned index entry %d: %d", first, second)
			}
			pairs = append(pairs, [2]int32{first, second})
		}
		d.prune.init(len(pairs))
		for _, p := range pairs {
			d.prune.put(p[0], p[1])
		}
	}

	d.minn = uint64(int64(a.minn))
	d.maxn = uint64(int64(a.maxn))
	d.useNgrams = a.maxn > 0
	d.bucket = uint32(a.bucket)
	d.bucket64 = uint64(int64(a.bucket))
	d.wordNgrams = int(a.wordNgrams)
	if a.bucket <= 0 && (a.maxn != 0 || d.wordNgrams > 1) {
		// The C++ code would divide by zero.
		return fmt.Errorf("fasttext: model uses n-grams with bucket %d", a.bucket)
	}

	// Token table. fastText sizes it at size/0.7; a power of two at most
	// half full gives the same answers with cheaper probing.
	n := 16
	for n < 2*int(size) {
		n *= 2
	}
	d.table = make([]int32, n)
	for i := range d.table {
		d.table[i] = -1
	}
	d.mask = uint32(n - 1)
	for i, w := range d.entries {
		slot := hash(w) & d.mask
		for d.table[slot] >= 0 && d.entries[d.table[slot]] != w {
			slot = (slot + 1) & d.mask
		}
		// A repeated entry replaces the earlier one, as in fastText.
		d.table[slot] = int32(i)
	}

	if d.useNgrams {
		d.subOff = make([]uint32, d.nwords+1)
		var buf []byte
		for i := int32(0); i < d.nwords; i++ {
			d.subIDs = append(d.subIDs, i)
			if w := d.entries[i]; w != eosToken {
				buf = append(append(append(buf[:0], '<'), w...), '>')
				d.subIDs = d.computeSubwords(buf, d.subIDs)
			}
			d.subOff[i+1] = uint32(len(d.subIDs))
		}
	}
	return nil
}

// find returns the entry number of token, or -1.
func (d *dictionary) find(token string, h uint32) int32 {
	slot := h & d.mask
	for {
		id := d.table[slot]
		if id < 0 || d.entries[id] == token {
			return id
		}
		slot = (slot + 1) & d.mask
	}
}

// pushHash is Dictionary::pushHash: it appends the input row for n-gram
// bucket id, or nothing if the model was pruned and dropped that bucket.
func (d *dictionary) pushHash(ids []int32, id int32) []int32 {
	if d.pruneidxSize == 0 || id < 0 {
		return ids
	}
	if d.pruneidxSize > 0 {
		v, ok := d.prune.get(id)
		if !ok {
			return ids
		}
		id = v
	}
	return append(ids, d.nwords+id)
}

// computeSubwords is Dictionary::computeSubwords: it appends the ids of the
// character n-grams of word, which already carries the < and > markers.
// N-grams are cut on UTF-8 character boundaries, where every byte that is
// not a continuation byte starts a character.
func (d *dictionary) computeSubwords(word []byte, ids []int32) []int32 {
	n := len(word)
	for i := 0; i < n; i++ {
		if word[i]&0xC0 == 0x80 {
			continue
		}
		// The hash of an n-gram extends the hash of its prefix.
		h := uint32(2166136261)
		j := i
		for c := uint64(1); j < n && c <= d.maxn; c++ {
			h = (h ^ uint32(int8(word[j]))) * 16777619
			j++
			for j < n && word[j]&0xC0 == 0x80 {
				h = (h ^ uint32(int8(word[j]))) * 16777619
				j++
			}
			if c >= d.minn && !(c == 1 && (i == 0 || j == n)) {
				ids = d.pushHash(ids, int32(h%d.bucket))
			}
		}
	}
	return ids
}

// addToken handles one token as Dictionary::getLine does and reports whether
// it was the end-of-sentence token, which ends the line.
func (d *dictionary) addToken(token string, s *scratch) bool {
	h := hash(token)
	id := d.find(token, h)
	switch {
	case id < 0:
		if strings.HasPrefix(token, labelPrefix) {
			// An unknown label. Ignored.
			break
		}
		// An unknown word contributes only its character n-grams.
		if token != eosToken && d.maxn != 0 {
			s.word = append(append(append(s.word[:0], '<'), token...), '>')
			s.ids = d.computeSubwords(s.word, s.ids)
		}
		s.hashes = append(s.hashes, int32(h))
	case d.types[id] == typeWord:
		if d.useNgrams {
			s.ids = append(s.ids, d.subIDs[d.subOff[id]:d.subOff[id+1]]...)
		} else {
			s.ids = append(s.ids, id)
		}
		s.hashes = append(s.hashes, int32(h))
	}
	return token == eosToken
}

// line is Dictionary::getLine on the stream text + "\n": it fills s.ids with
// the input rows for the first line of text.
func (d *dictionary) line(text string, s *scratch) {
	s.ids = s.ids[:0]
	s.hashes = s.hashes[:0]
	ended := false
	for i, n := 0, len(text); i < n; {
		c := text[i]
		if c == '\n' {
			break
		}
		if isSeparator[c] {
			i++
			continue
		}
		j := i + 1
		for j < n && !isSeparator[text[j]] {
			j++
		}
		if d.addToken(text[i:j], s) {
			ended = true
			break
		}
		i = j
	}
	if !ended {
		// The newline that ends the line is itself a token.
		d.addToken(eosToken, s)
	}
	d.addWordNgrams(s)
}

// addWordNgrams is Dictionary::addWordNgrams. The token hashes are stored as
// int32 and widened with sign extension, as upstream does.
func (d *dictionary) addWordNgrams(s *scratch) {
	if d.wordNgrams <= 1 {
		return
	}
	hashes := s.hashes
	for i := range hashes {
		h := uint64(int64(hashes[i]))
		for j := i + 1; j < len(hashes) && j < i+d.wordNgrams; j++ {
			h = h*116049371 + uint64(int64(hashes[j]))
			s.ids = d.pushHash(s.ids, int32(h%d.bucket64))
		}
	}
}

// pruneTable maps n-gram buckets to rows in a pruned model. It is an
// open-addressing table; fastText uses std::unordered_map.
type pruneTable struct {
	keys []int32 // -1: empty
	vals []int32
	mask uint32
}

func (t *pruneTable) init(n int) {
	size := 16
	for size < 2*n {
		size *= 2
	}
	t.keys = make([]int32, size)
	for i := range t.keys {
		t.keys[i] = -1
	}
	t.vals = make([]int32, size)
	t.mask = uint32(size - 1)
}

func (t *pruneTable) slot(key int32) uint32 {
	return (uint32(key) * 2654435761) >> 7 & t.mask
}

func (t *pruneTable) put(key, val int32) {
	if key < 0 {
		// pushHash never looks up a negative id.
		return
	}
	s := t.slot(key)
	for t.keys[s] >= 0 && t.keys[s] != key {
		s = (s + 1) & t.mask
	}
	t.keys[s] = key
	t.vals[s] = val
}

func (t *pruneTable) get(key int32) (int32, bool) {
	s := t.slot(key)
	for {
		k := t.keys[s]
		if k == key {
			return t.vals[s], true
		}
		if k < 0 {
			return 0, false
		}
		s = (s + 1) & t.mask
	}
}
