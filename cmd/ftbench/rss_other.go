//go:build !unix

package main

// peakRSSMegabytes is not measured on this platform.
func peakRSSMegabytes() float64 { return 0 }
