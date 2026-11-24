# Security policy

## Supported versions

Security fixes go into the latest minor release of each module.

## Model files are untrusted input

`fasttext.Open` and `fasttext.Load` parse a binary format. They are written to fail safely on truncated or damaged files:

- Sizes are checked against the length of the input before allocating.
- When the input has no known length (`Load` from a stream), memory grows with the bytes that arrive, not with the sizes the file claims.
- `TestDamagedModel` loads truncated files and files with random bytes changed, and predicts with whatever loads. `TestLoadClaimsTooMuch` loads short files that claim tens of gigabytes. `FuzzLoad` loads mutated files with and without a known length.

A damaged model may still load and give meaningless predictions. If you load models from sources you don't control, check their SHA-256 first.

## Reporting a vulnerability

Please don't open a public issue. Report it privately through GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository, and include:

- a model file or a program that shows the problem;
- the module version, Go version and platform.

You can expect a first answer within a week.
