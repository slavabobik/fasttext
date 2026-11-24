// Package testdata finds the reference material the tests compare against:
// models, corpora, golden dumps and results under accuracy/ at the root of
// the repository.
//
// Large models are stored in Git LFS. When a file is only an LFS pointer
// (the repository was cloned without git-lfs, or "git lfs pull" was not run)
// the tests that need it are skipped instead of failing.
//
// lid.176.bin is not stored under accuracy/ at all: it is embedded by the
// lid176 module in three parts, and Model assembles it once into the user's
// cache directory.
package testdata

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// Lid176SHA256 is the SHA-256 of lid.176.bin as published by fastText.
const Lid176SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"

// Lid176Parts are the files that make up lid.176.bin, relative to the
// repository root.
var Lid176Parts = []string{
	"lid176/model/lid.176.bin.0",
	"lid176/model/lid.176.bin.1",
	"lid176/model/lid.176.bin.2",
}

var lfsPrefix = []byte("version https://git-lfs")

// Root returns the root of the repository: the nearest directory above the
// working directory that contains accuracy/ and lid176/.
func Root() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if isDir(filepath.Join(dir, "accuracy")) && isDir(filepath.Join(dir, "lid176")) {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", errors.New("testdata: repository root not found")
		}
		dir = parent
	}
}

// Dir returns the accuracy directory, or the one named by
// FASTTEXT_ACCURACY_DIR.
func Dir() string {
	if d := os.Getenv("FASTTEXT_ACCURACY_DIR"); d != "" {
		return d
	}
	root, err := Root()
	if err != nil {
		return "accuracy"
	}
	return filepath.Join(root, "accuracy")
}

// Path returns the path of a file under the accuracy directory, such as
// "corpus/edge-cases.txt". It skips the test if the file is missing or is an
// LFS pointer.
func Path(tb testing.TB, rel string) string {
	tb.Helper()
	p := filepath.Join(Dir(), filepath.FromSlash(rel))
	if err := usable(p); err != nil {
		tb.Skipf("test data not available: %v", err)
	}
	return p
}

// Model returns the path of a test model by file name, such as
// "t-softmax-d32.ftz" or "lid.176.bin". It skips the test if the model is not
// available.
func Model(tb testing.TB, name string) string {
	tb.Helper()
	p := filepath.Join(Dir(), "models", name)
	err := usable(p)
	if err == nil {
		return p
	}
	if name == "lid.176.bin" && errors.Is(err, os.ErrNotExist) {
		p, err := lid176()
		if err != nil {
			tb.Skipf("lid.176.bin not available: %v", err)
		}
		return p
	}
	tb.Skipf("model not available: %v", err)
	return ""
}

// usable reports why the file at p cannot be used, or nil.
func usable(p string) error {
	f, err := os.Open(p)
	if err != nil {
		return err
	}
	defer f.Close()
	head := make([]byte, len(lfsPrefix))
	n, _ := io.ReadFull(f, head)
	if bytes.Equal(head[:n], lfsPrefix) {
		return fmt.Errorf("%s is a Git LFS pointer; run \"git lfs pull\"", p)
	}
	return nil
}

var (
	lidOnce sync.Once
	lidPath string
	lidErr  error
)

// lid176 assembles lid.176.bin from the parts in lid176/model into the user
// cache directory, once, and checks its SHA-256.
func lid176() (string, error) {
	lidOnce.Do(func() {
		lidPath, lidErr = assembleLid176()
	})
	return lidPath, lidErr
}

func assembleLid176() (string, error) {
	root, err := Root()
	if err != nil {
		return "", err
	}
	cache, err := os.UserCacheDir()
	if err != nil {
		cache = os.TempDir()
	}
	dir := filepath.Join(cache, "fasttext-go-testdata")
	out := filepath.Join(dir, "lid.176.bin")
	if sum, err := fileSHA256(out); err == nil && sum == Lid176SHA256 {
		return out, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.CreateTemp(dir, "lid.176.bin.*")
	if err != nil {
		return "", err
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	for _, part := range Lid176Parts {
		p := filepath.Join(root, filepath.FromSlash(part))
		if err := usable(p); err != nil {
			tmp.Close()
			return "", err
		}
		f, err := os.Open(p)
		if err != nil {
			tmp.Close()
			return "", err
		}
		_, err = io.Copy(io.MultiWriter(tmp, h), f)
		f.Close()
		if err != nil {
			tmp.Close()
			return "", err
		}
	}
	if err := tmp.Close(); err != nil {
		return "", err
	}
	if sum := hex.EncodeToString(h.Sum(nil)); sum != Lid176SHA256 {
		return "", fmt.Errorf("assembled lid.176.bin has SHA-256 %s, want %s", sum, Lid176SHA256)
	}
	if err := os.Rename(tmp.Name(), out); err != nil {
		return "", err
	}
	return out, nil
}

func fileSHA256(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}
