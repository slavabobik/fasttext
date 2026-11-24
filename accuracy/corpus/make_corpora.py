#!/usr/bin/env python3
"""Writes the generated test corpora next to this script.

  edge-cases.txt   hand-written inputs that exercise tokenising and hashing
  random.txt       seeded random inputs: raw bytes, separator-heavy bytes,
                   random Unicode, and damaged Tatoeba sentences

Every corpus is a byte file with one record per line ("\n"). Records are
bytes, not text: several are deliberately not valid UTF-8.

The output is deterministic. `random.Random(seed)` with `getrandbits`,
`randrange`, `choice` and `sample` on a fixed Python major version gives the
same stream on every platform; the files are still checked in by checksum
(see README.md) so a change shows up.
"""

import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))


def edge_cases():
    out = []
    add = out.append

    # Empty and separator-only records.
    add(b"")
    add(b" ")
    add(b"   ")
    add(b"\t")
    add(b"\r")
    add(b"\v")
    add(b"\f")
    add(b"\0")
    add(b" \t\r\v\f\0 ")

    # Plain words and spacing.
    add(b"hello")
    add(b"Hello World")
    add(b"HELLO WORLD")
    add(b" leading space")
    add(b"trailing space ")
    add(b"double  space")
    add(b"windows line ending\r")
    add(b"a\tb\vc\fd\0e\rf g")
    add(b"a")
    add(b"I")
    add(b"ok")
    add(b"ok.")
    add(b"lol")
    add(b"12345")
    add(b"3.14 2,71 1e10 -5")
    add(b"!!!")
    add(b"...")
    add(b"?")
    add(b"http://example.com/path?x=1&y=2#frag")
    add(b"user@example.com")
    add(b"@user check this #hashtag")
    add(b"don't can't it's")
    add(b"semi;colon,comma.period:colon")

    # Label prefix. Tokens that start with __label__ are not words.
    add(b"__label__en hello")
    add(b"hello __label__en")
    add(b"__label__")
    add(b"__label__nosuchlabel")
    add(b"__label__en")
    add(b"__label__en __label__fr")
    add(b"x__label__en hello")
    add(b"__label hello")
    add(b"__LABEL__en hello")

    # The end-of-sentence token written out. getLine stops at it.
    add(b"</s>")
    add(b"</s> </s>")
    add(b"hello </s> world")
    add(b"hello</s>world")
    add(b"</s> hello")

    # Word boundary markers used for character n-grams.
    add(b"<")
    add(b">")
    add(b"<>")
    add(b"<hello>")
    add(b"<hello")
    add(b"hello>")
    add(b"< >")

    # Non-ASCII: bytes >= 0x80 hit the sign-extension in the hash.
    for s in [
        "é",
        "ÿ",
        "ß",
        "中",
        "ééé",
        "café",
        "naïve façade",
        "Привет, мир!",
        "你好，世界",
        "こんにちは世界",
        "안녕하세요",
        "مرحبا بالعالم",
        "שלום עולם",
        "नमस्ते दुनिया",
        "สวัสดีชาวโลก",
        "Γειά σου Κόσμε",
        "hello привет 你好 مرحبا",
        "Tiếng Việt có dấu",
        "İstanbul ığdır",
        "e\u0301",  # e + combining acute
        "\u00a0",  # no-break space: not a separator for fastText
        "a\u00a0b",
        "a\u3000b",  # ideographic space
        "a\u2028b",  # line separator
        "a\u200bb",  # zero-width space
        "\ufeffhello",  # byte-order mark
        "\ufffd",
        "\U0001f600",
        "I \u2764\ufe0f you \U0001f44d\U0001f3fd",
        "\U0001f468\u200d\U0001f469\u200d\U0001f467\u200d\U0001f466",
        "haha \U0001f602\U0001f602\U0001f602",
        "\U0001f1fa\U0001f1e6",
        "\U00010348",  # four-byte, outside emoji
        "\U0010ffff",
    ]:
        add(s.encode("utf-8"))

    # Invalid UTF-8.
    add(b"\x80")
    add(b"\x80\x80\x80")
    add(b"\xbf")
    add(b"\xc3")  # lead byte, nothing after
    add(b"\xe4\xbd")  # truncated three-byte sequence
    add(b"\xf0\x9f\x98")  # truncated four-byte sequence
    add(b"\xc0\xaf")  # overlong
    add(b"\xed\xa0\x80")  # surrogate
    add(b"\xff")
    add(b"\xff\xfe")
    add(b"\xfe\xff\x00h\x00i")
    add(b"\xc3 \xa9")  # lead and continuation split by a space
    add(b"a\x80b")
    add(b"\x80abc")
    add(b"abc\xc3")
    add(b"caf\xe9")  # Latin-1
    add(b"\xcf\xf0\xe8\xe2\xe5\xf2")  # Windows-1251
    add(bytes(range(1, 256)).replace(b"\n", b""))
    add(bytes(range(128, 256)))
    add(bytes(range(128, 192)) * 3)
    add(bytes(range(192, 256)) * 3)

    # Control characters that are not separators.
    add(b"a\x01b\x02c")
    add(b"\x1b[31mred\x1b[0m")
    add(b"a\x7fb")

    # Length limits. MAX_LINE_SIZE is 1024 tokens.
    add(b"a" * 10000)
    add("é".encode() * 3000)
    add("中".encode() * 3000)
    for n in [1022, 1023, 1024, 1025, 1026, 2048, 5000]:
        add(b" ".join(b"w%d" % i for i in range(n)))
    add(b" ".join([b"the"] * 3000))
    add(b"x " * 1500 + b"__label__en")
    add(b"the quick brown fox jumps over the lazy dog " * 200)

    # Short chat-style messages.
    for s in [
        "brb",
        "jajaja",
        "kkkkk",
        "mdr",
        "xD",
        ":)",
        "ok ok ok",
        "yes",
        "no",
        "si",
        "да",
        "нет",
        "はい",
        "thx!!",
        "gm",
        "wtf???",
        "omg lol",
        "+1",
        "k",
        "👍",
        "🙏🙏",
        "https://t.co/abc123",
        "LOOOOOOOL",
        "hahahahahahahahahaha",
        "Hello, how are you?",
        "Bonjour, comment ça va ?",
        "Hola, ¿cómo estás?",
        "Wie geht es dir?",
        "Ciao, come stai?",
        "Olá, tudo bem?",
        "Hoe gaat het met je?",
        "Jak się masz?",
    ]:
        add(s.encode("utf-8"))

    for r in out:
        assert b"\n" not in r
    return out


def random_corpus(tatoeba_path):
    rng = random.Random(20261120)
    out = []

    def rbytes(n):
        return bytes(rng.getrandbits(8) for _ in range(n))

    # 1. Raw random bytes. A newline ends a record, so drop it.
    for _ in range(5000):
        out.append(rbytes(rng.randrange(0, 200)).replace(b"\n", b""))

    # 2. Separator-heavy: short tokens from a small alphabet.
    alphabet = [b" ", b" ", b" ", b"\t", b"\r", b"\v", b"\f", b"\0", b"<", b">",
                b"_", b"/", b"s", b"l", b"a", b"b", b"e", b"\xc3", b"\xa9",
                b"\xe4", b"\xb8", b"\xad", b"\xf0", b"\x9f", b"\x98", b"\x80",
                b"__label__", b"</s>"]
    for _ in range(5000):
        n = rng.randrange(0, 60)
        out.append(b"".join(rng.choice(alphabet) for _ in range(n)))

    # 3. Random valid Unicode, biased to a few blocks.
    blocks = [(0x20, 0x7F), (0xA0, 0x250), (0x370, 0x530), (0x590, 0x700),
              (0x900, 0xE00), (0x3040, 0x3100), (0x4E00, 0x9FFF),
              (0xAC00, 0xD7A4), (0x1F300, 0x1F700), (0x20, 0x10FFFF)]
    for _ in range(5000):
        n = rng.randrange(0, 80)
        chars = []
        for _ in range(n):
            lo, hi = rng.choice(blocks)
            c = rng.randrange(lo, hi)
            if 0xD800 <= c <= 0xDFFF or c == 0x0A:
                c = 0x20
            chars.append(chr(c))
        out.append("".join(chars).encode("utf-8"))

    # 4. Damaged natural text: real sentences hit known words and n-grams, the
    # damage hits the boundaries.
    with open(tatoeba_path, "rb") as f:
        sentences = [line.rstrip(b"\n").split(b"\t", 2)[2] for line in f]
    picks = rng.sample(range(len(sentences)), 5000)
    for i in picks:
        s = bytearray(sentences[i])
        kind = rng.randrange(6)
        if not s:
            out.append(bytes(s))
            continue
        if kind == 0:  # cut anywhere, possibly inside a character
            s = s[: rng.randrange(len(s) + 1)]
        elif kind == 1:  # drop one byte
            del s[rng.randrange(len(s))]
        elif kind == 2:  # flip one bit
            p = rng.randrange(len(s))
            s[p] ^= 1 << rng.randrange(8)
        elif kind == 3:  # insert a random byte
            s.insert(rng.randrange(len(s) + 1), rng.getrandbits(8))
        elif kind == 4:  # replace a space with another separator
            p = s.find(b" ")
            if p >= 0:
                s[p : p + 1] = rng.choice([b"\t", b"\r", b"\v", b"\f", b"\0", b"  "])
        else:  # join with another sentence
            s += b" " + sentences[rng.randrange(len(sentences))]
        out.append(bytes(s).replace(b"\n", b""))

    return out


def write(name, records):
    path = os.path.join(HERE, name)
    with open(path, "wb") as f:
        for r in records:
            f.write(r + b"\n")
    print(f"{name}: {len(records)} records")


if __name__ == "__main__":
    write("edge-cases.txt", edge_cases())
    write("random.txt", random_corpus(os.path.join(HERE, "tatoeba-test.tsv")))
