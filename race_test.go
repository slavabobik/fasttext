//go:build race

package fasttext_test

// The race detector makes sync.Pool drop items at random, so allocation
// counts mean nothing under it.
const raceEnabled = true
