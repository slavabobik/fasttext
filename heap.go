package fasttext

// scored is one candidate label: fastText's std::pair<real, int32_t>.
type scored struct {
	score float32 // log-probability
	label int32
}

// fastText keeps the best k candidates in a heap ordered by score alone, with
// std::push_heap, std::pop_heap and std::sort_heap. When two labels have
// exactly the same score, which one survives and in what order they come out
// depends on the moves those routines make. The functions below repeat the
// moves of libstdc++ (bits/stl_heap.h), which is what the Linux reference
// build uses.
//
// The comparison is fastText's comparePairs: comp(l, r) = l.score > r.score,
// which makes the heap a min-heap: h[0] is the worst candidate kept.

// siftUp is libstdc++'s __push_heap: it moves value up from hole.
func siftUp(h []scored, hole, top int, value scored) {
	parent := (hole - 1) / 2
	for hole > top && h[parent].score > value.score {
		h[hole] = h[parent]
		hole = parent
		parent = (hole - 1) / 2
	}
	h[hole] = value
}

// pushHeap is std::push_heap(h.begin(), h.end(), comparePairs): the last
// element is the new one.
func pushHeap(h []scored) {
	siftUp(h, len(h)-1, 0, h[len(h)-1])
}

// adjustHeap is libstdc++'s __adjust_heap over h[:n]: it moves the hole down
// to a leaf along the preferred children and then sifts value up from there.
func adjustHeap(h []scored, hole, n int, value scored) {
	top := hole
	child := hole
	for child < (n-1)/2 {
		child = 2 * (child + 1)
		if h[child].score > h[child-1].score {
			child--
		}
		h[hole] = h[child]
		hole = child
	}
	if n&1 == 0 && child == (n-2)/2 {
		child = 2 * (child + 1)
		h[hole] = h[child-1]
		hole = child - 1
	}
	siftUp(h, hole, top, value)
}

// popHeap is std::pop_heap(h.begin(), h.end(), comparePairs): it moves h[0]
// to the end and restores the heap in h[:len(h)-1].
func popHeap(h []scored) {
	if len(h) > 1 {
		last := len(h) - 1
		value := h[last]
		h[last] = h[0]
		adjustHeap(h, 0, last, value)
	}
}

// sortHeap is std::sort_heap(h.begin(), h.end(), comparePairs): it leaves h
// ordered best first.
func sortHeap(h []scored) {
	for n := len(h); n > 1; n-- {
		popHeap(h[:n])
	}
}
