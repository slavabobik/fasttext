package fasttext

// KernelNames lists the implementations of the inner loops that this build
// can run on this machine: "go" and, with GOEXPERIMENT=simd, the vector ones.
func KernelNames() []string {
	names := []string{"go"}
	for _, k := range kernels {
		names = append(names, k.name)
	}
	return names
}

// UseKernel selects an implementation by name for models loaded afterwards,
// and returns a function that restores the default.
func UseKernel(name string) (restore func()) {
	vec = nil
	for i := range kernels {
		if kernels[i].name == name {
			vec = &kernels[i]
		}
	}
	return useBestKernel
}
