# The reference environment: the toolchain that wrote golden/ (see
# golden/BUILD: gcc 14.2 from Debian, static glibc 2.41, Linux) plus Go for
# the port. accuracy/verify.sh runs inside it.
#
#   docker build -t fasttext-reference accuracy/reference
#   docker run --rm -v "$PWD:/src" -w /src fasttext-reference accuracy/verify.sh
#
# Debian trixie ships gcc 14.2 and glibc 2.41, the versions of the reference.
FROM golang:1.27-trixie

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ git ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*

# The repository is mounted at /src and owned by another user.
RUN git config --global --add safe.directory '*'
ENV GOFLAGS=-buildvcs=false
