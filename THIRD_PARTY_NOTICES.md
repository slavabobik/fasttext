# Third-party notices

This repository contains or is derived from the following third-party works.

| Component | Where | Licence | Copyright |
| --- | --- | --- | --- |
| fastText (prediction code, ported to Go) | `*.go` in the root package | MIT, see [LICENSE-fastText](LICENSE-fastText) | © 2016-present, Facebook, Inc. |
| ARM Optimized Routines (`expf`, `exp`, `log`, ported to Go; also the algorithms in glibc) | `libm.go`, `libm_tables.go` | MIT (chosen from MIT OR Apache-2.0 WITH LLVM-exception), see [LICENSE-arm-optimized-routines](LICENSE-arm-optimized-routines) | © 1999-2022, Arm Limited |
| `lid.176.bin` language-identification model | `lid176/model/` (embedded), `accuracy/` | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), see [lid176/LICENSE-lid176](lid176/LICENSE-lid176) | © Facebook, Inc. |
| `lid.176.ftz` language-identification model | `accuracy/models/` | CC BY-SA 3.0 | © Facebook, Inc. |
| Tatoeba sentences | `accuracy/corpus/`, and the training data of the `t-*` test models | [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/) | © the Tatoeba contributors, <https://tatoeba.org> |

The C++ tools in `accuracy/` (`refdump.cc`, `ftbench.cc`, `heapref.cc`, `libm/*.c`) are part of this project, under its MIT licence. They link against fastText's sources, which `accuracy/refdump/fetch.sh` downloads from <https://github.com/facebookresearch/fastText> and which are not included here.

The `t-*` test models in `accuracy/models` were trained for this project with fastText on Tatoeba sentences, and are distributed under CC BY 2.0 FR.
