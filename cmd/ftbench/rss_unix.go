//go:build unix

package main

import (
	"runtime"
	"syscall"
)

// peakRSSMegabytes returns the peak resident set size of the process.
func peakRSSMegabytes() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	mb := float64(ru.Maxrss) / 1024 // kilobytes on Linux
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		mb /= 1024 // bytes on Darwin
	}
	return mb
}
