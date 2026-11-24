# lid176

Language detection with fastText's `lid.176.bin` model built in: 176 languages, pure Go, no cgo, nothing to download.

```sh
go get github.com/slavabobik/fasttext/lid176
```

```go
lang, err := lid176.Detect("Où est ma commande ?")
// lang.Code == "fr"
```

The command-line tool:

```sh
go install github.com/slavabobik/fasttext/lid176/cmd/lid@latest
echo "Где мой заказ?" | lid        # ru	0.99…
```

This is a separate Go module so that the 131 MB model is downloaded only by programs that import it. The model is stored in three parts (GitHub rejects files over 100 MB) and embedded with `go:embed`; `TestEmbeddedChecksum` checks that the parts add up to the published file.

The model is © Facebook, Inc., under CC BY-SA 3.0 ([LICENSE-lid176](LICENSE-lid176)). See the [main README](../README.md) for the API, accuracy and benchmarks.
