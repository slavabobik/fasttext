# Accuracy per language

How often the lid.176 models name the right language for a sentence from [Tatoeba](https://tatoeba.org), for each of the 155 Tatoeba languages that lid.176 knows. The Go port returns the same bits as fastText, so these are fastText's own numbers: any program that runs these models, in Go or C++, gets the same answers.

- **Texts:** the 182,958 sentences of `accuracy/corpus/tatoeba-test.tsv` (every 64th sentence of a Tatoeba export) in a language that lid.176 knows, 36 characters long on average. The other 30,267 sentences, in 168 languages that lid.176 does not know, are left out.
- **Correct** means the top answer of `Detector.Detect` is the language Tatoeba gives. No threshold is applied.
- **Caution:** fastText trained lid.176 partly on Tatoeba, so many of these sentences, or ones close to them, were seen in training. Expect lower numbers on other text. The numbers are best used to compare languages and the two models.
- **Reproduce:** `git lfs pull && go test ./langid -run TestTatoeba -v`. The test fails if any count changes, and `-update` rewrites this page and [`langid/testdata/tatoeba-lid176.tsv`](../langid/testdata/tatoeba-lid176.tsv), which also lists the languages with fewer than 20 texts.

## Summary

| | lid.176.bin | lid.176.ftz |
|---|---:|---:|
| Texts correct, of 182,958 | 97.3% | 95.6% |
| Mean accuracy of the 102 languages with 20 or more texts | 75.3% | 66.8% |

## By length

A few words carry little evidence, and close languages share most of them, so short texts are the hard case.

| Characters | Texts | lid.176.bin | lid.176.ftz |
|---|---:|---:|---:|
| 1 to 20 | 32,237 | 93.2% | 89.0% |
| 21 to 40 | 99,655 | 97.9% | 96.5% |
| 41 to 80 | 44,249 | 98.8% | 98.1% |
| 81 and more | 6,817 | 97.2% | 96.9% |

## By language

Languages with 20 or more texts, most texts first. The last column lists what lid.176.bin answered instead, the three most frequent wrong answers with their counts.

| Language | Tatoeba | Code | Texts | lid.176.bin | lid.176.ftz | Wrong answers of lid.176.bin |
|---|---|---|---:|---:|---:|---|
| English | `eng` | `en` | 31,930 | 99.9% | 99.6% | it 9, es 5, nl 3 |
| Russian | `rus` | `ru` | 19,194 | 99.9% | 99.6% | uk 12, bg 7, mk 4 |
| Italian | `ita` | `it` | 15,499 | 99.6% | 98.6% | es 20, en 16, fr 7 |
| Esperanto | `epo` | `eo` | 13,025 | 99.6% | 99.2% | es 13, it 11, de 3 |
| German | `deu` | `de` | 12,256 | 99.7% | 99.4% | en 13, it 4, eo 3 |
| Turkish | `tur` | `tr` | 11,782 | 99.7% | 99.4% | en 9, de 3, es 3 |
| French | `fra` | `fr` | 11,338 | 99.4% | 98.9% | en 20, it 13, es 10 |
| Portuguese | `por` | `pt` | 6,960 | 98.3% | 96.3% | es 66, it 14, gl 10 |
| Hungarian | `hun` | `hu` | 6,888 | 99.4% | 98.0% | pt 7, en 6, it 4 |
| Spanish | `spa` | `es` | 6,881 | 98.6% | 97.4% | pt 33, it 22, eo 9 |
| Japanese | `jpn` | `ja` | 3,883 | 99.9% | 99.7% | zh 2 |
| Hebrew | `heb` | `he` | 3,343 | 100.0% | 99.9% |  |
| Dutch | `nld` | `nl` | 3,139 | 95.5% | 92.6% | en 57, de 22, nds 13 |
| Ukrainian | `ukr` | `uk` | 2,934 | 97.6% | 92.0% | ru 60, bg 5, mk 3 |
| Lithuanian | `lit` | `lt` | 2,530 | 96.0% | 91.3% | eo 11, sr 9, es 8 |
| Finnish | `fin` | `fi` | 2,461 | 98.0% | 96.0% | en 10, it 8, de 5 |
| Polish | `pol` | `pl` | 2,118 | 99.0% | 97.7% | en 6, cs 2, es 2 |
| Tatar | `tat` | `tt` | 1,589 | 79.6% | 67.8% | tr 91, az 65, fi 45 |
| Chinese (Mandarin) | `cmn` | `zh` | 1,444 | 99.5% | 92.0% | yue 3, ja 2, ko 1 |
| Marathi | `mar` | `mr` | 1,376 | 99.9% | 99.0% | hi 1, new 1 |
| Czech | `ces` | `cs` | 1,337 | 95.4% | 90.9% | sk 8, sl 7, pl 6 |
| Tagalog | `tgl` | `tl` | 1,241 | 93.2% | 88.6% | ceb 22, en 16, hr 7 |
| Macedonian | `mkd` | `mk` | 1,224 | 98.9% | 93.6% | ru 6, bg 5, uk 2 |
| Arabic | `ara` | `ar` | 1,098 | 97.6% | 96.9% | arz 10, fa 5, mzn 3 |
| Danish | `dan` | `da` | 1,012 | 90.9% | 87.3% | no 40, sv 23, en 6 |
| Latin | `lat` | `la` | 914 | 87.0% | 74.5% | it 29, en 20, fr 15 |
| Swedish | `swe` | `sv` | 904 | 94.0% | 91.9% | da 17, no 9, ca 3 |
| Serbian | `srp` | `sr` | 831 | 80.9% | 68.4% | mk 59, ru 33, hr 25 |
| Armenian | `hye` | `hy` | 777 | 100.0% | 100.0% |  |
| Greek | `ell` | `el` | 679 | 100.0% | 100.0% |  |
| Romanian | `ron` | `ro` | 674 | 93.9% | 90.4% | es 9, en 7, fr 5 |
| Interlingua | `ina` | `ia` | 661 | 69.7% | 53.9% | it 72, es 59, fr 22 |
| Slovak | `slk` | `sk` | 520 | 63.8% | 50.6% | cs 98, pl 16, no 8 |
| Persian | `pes` | `fa` | 517 | 99.8% | 99.4% | ar 1 |
| Vietnamese | `vie` | `vi` | 484 | 99.0% | 99.2% | en 1, hr 1, it 1 |
| Indonesian | `ind` | `id` | 461 | 91.3% | 87.2% | ms 20, it 5, en 2 |
| Yiddish | `yid` | `yi` | 413 | 92.5% | 79.2% | he 28, es 1, mn 1 |
| Bulgarian | `bul` | `bg` | 404 | 96.3% | 81.4% | mk 9, ru 4, sr 1 |
| Occitan | `oci` | `oc` | 403 | 47.1% | 33.5% | es 67, ca 47, fr 34 |
| Swahili (Congo) | `swc` | `sw` | 346 | 54.0% | 41.9% | eo 21, en 20, tl 18 |
| Latvian | `lvs` | `lv` | 341 | 87.1% | 84.5% | lt 13, en 4, fr 4 |
| Cantonese | `yue` | `yue` | 327 | 32.4% | 30.9% | zh 220, ja 1 |
| Low German | `nds` | `nds` | 286 | 88.1% | 82.2% | de 12, nl 10, en 6 |
| Norwegian Bokmål | `nob` | `no` | 284 | 38.4% | 33.5% | da 137, sv 11, nn 10 |
| Lojban | `jbo` | `jbo` | 270 | 95.6% | 89.6% | bs 2, cbk 1, cs 1 |
| Korean | `kor` | `ko` | 266 | 99.6% | 98.1% | tt 1 |
| Hindi | `hin` | `hi` | 249 | 96.8% | 94.8% | mr 5, en 1, sa 1 |
| Interlingue | `ile` | `ie` | 248 | 44.8% | 28.2% | es 49, it 21, en 14 |
| Bengali | `ben` | `bn` | 242 | 100.0% | 99.6% |  |
| Belarusian | `bel` | `be` | 238 | 87.8% | 72.7% | ru 15, uk 13, mk 1 |
| Ido | `ido` | `io` | 234 | 42.7% | 32.5% | eo 96, es 12, it 8 |
| Central Kurdish | `ckb` | `ckb` | 215 | 95.3% | 93.0% | fa 6, ur 2, ar 1 |
| Assamese | `asm` | `as` | 208 | 69.7% | 58.7% | bn 60, bcl 1, bpy 1 |
| Azerbaijani | `aze` | `az` | 203 | 77.8% | 67.5% | tr 41, en 1, io 1 |
| Icelandic | `isl` | `is` | 203 | 93.6% | 93.6% | da 3, et 2, no 2 |
| Norwegian Nynorsk | `nno` | `nn` | 193 | 34.2% | 32.6% | no 39, da 38, de 12 |
| Catalan | `cat` | `ca` | 171 | 78.4% | 66.7% | es 18, pt 7, fr 2 |
| Kurdish (Kurmanji) | `kmr` | `ku` | 159 | 71.7% | 58.5% | tr 8, en 7, diq 4 |
| Uyghur | `uig` | `ug` | 151 | 100.0% | 96.7% |  |
| Ossetian | `oss` | `os` | 131 | 81.7% | 67.9% | ru 12, id 2, rm 2 |
| Breton | `bre` | `br` | 129 | 86.0% | 72.9% | en 7, fr 2, la 2 |
| Georgian | `kat` | `ka` | 129 | 100.0% | 100.0% |  |
| Croatian | `hrv` | `hr` | 121 | 44.6% | 39.7% | sr 58, pl 3, bs 2 |
| Turkmen | `tuk` | `tk` | 118 | 49.2% | 36.4% | tr 32, uz 6, en 5 |
| Galician | `glg` | `gl` | 114 | 54.4% | 42.1% | pt 35, es 17 |
| Basque | `eus` | `eu` | 109 | 90.8% | 81.7% | eo 2, it 2, es 1 |
| Volapük | `vol` | `vo` | 106 | 73.6% | 41.5% | en 4, de 3, eu 3 |
| Malay (Standard) | `zsm` | `ms` | 105 | 26.7% | 26.7% | id 73, ar 2, en 1 |
| Thai | `tha` | `th` | 103 | 100.0% | 99.0% |  |
| Estonian | `est` | `et` | 100 | 78.0% | 66.0% | fi 6, en 5, es 2 |
| Wu Chinese | `wuu` | `wuu` | 86 | 82.6% | 57.0% | zh 14, ja 1 |
| Swahili | `swh` | `sw` | 80 | 62.5% | 51.2% | en 10, id 3, tl 3 |
| Kazakh | `kaz` | `kk` | 78 | 92.3% | 82.1% | ru 2, ba 1, de 1 |
| Afrikaans | `afr` | `af` | 75 | 73.3% | 64.0% | nl 12, en 5, fi 1 |
| Cornish | `cor` | `kw` | 73 | 82.2% | 72.6% | cy 2, pl 2, als 1 |
| Nepali | `npi` | `ne` | 72 | 76.4% | 34.7% | hi 14, sa 2, new 1 |
| Chavacano | `cbk` | `cbk` | 63 | 28.6% | 9.5% | es 35, it 3, pt 2 |
| Meadow Mari | `mhr` | `mhr` | 58 | 94.8% | 75.9% | ru 2, bg 1 |
| Slovene | `slv` | `sl` | 58 | 46.6% | 37.9% | sr 14, pl 4, eo 2 |
| Urdu | `urd` | `ur` | 56 | 96.4% | 94.6% | fa 1, pnb 1 |
| Guarani | `grn` | `gn` | 53 | 41.5% | 26.4% | es 8, la 3, cs 2 |
| North Frisian | `frr` | `frr` | 45 | 0.0% | 0.0% | nds 13, nl 6, de 4 |
| Mongolian | `mon` | `mn` | 43 | 93.0% | 86.0% | bg 1, bxr 1, ru 1 |
| Albanian | `sqi` | `sq` | 41 | 80.5% | 78.0% | hr 2, eo 1, pl 1 |
| Khmer | `khm` | `km` | 40 | 100.0% | 97.5% |  |
| Malay | `zlm` | `ms` | 39 | 5.1% | 5.1% | id 30, ar 1, hr 1 |
| Irish | `gle` | `ga` | 37 | 91.9% | 70.3% | en 1, es 1, su 1 |
| Ilocano | `ilo` | `ilo` | 36 | 86.1% | 69.4% | tl 2, es 1, hr 1 |
| Zazaki | `zza` | `diq` | 35 | 2.9% | 0.0% | ku 15, tr 10, hu 3 |
| Chuvash | `chv` | `cv` | 34 | 91.2% | 67.6% | ru 2, bg 1 |
| Cebuano | `ceb` | `ceb` | 30 | 56.7% | 30.0% | tl 9, eo 2, en 1 |
| Welsh | `cym` | `cy` | 30 | 63.3% | 40.0% | en 3, es 3, de 1 |
| Bosnian | `bos` | `bs` | 29 | 0.0% | 0.0% | sr 17, hr 8, eo 2 |
| Uzbek | `uzb` | `uz` | 29 | 48.3% | 24.1% | tr 6, en 4, de 1 |
| Lower Sorbian | `dsb` | `dsb` | 28 | 53.6% | 28.6% | pl 5, hu 2, sr 2 |
| Waray | `war` | `war` | 27 | 55.6% | 40.7% | ceb 3, tl 2, de 1 |
| Mingrelian | `xmf` | `xmf` | 26 | 0.0% | 0.0% | it 7, tr 3, en 2 |
| Upper Sorbian | `hsb` | `hsb` | 25 | 40.0% | 16.0% | pl 7, sr 4, cs 1 |
| Buryat | `bua` | `bxr` | 24 | 8.3% | 4.2% | mn 13, ru 2, sah 2 |
| Kapampangan | `pam` | `pam` | 24 | 66.7% | 29.2% | ca 1, en 1, ga 1 |
| Avar | `ava` | `av` | 23 | 17.4% | 4.3% | ru 5, lez 4, tt 3 |
| Javanese | `jav` | `jv` | 20 | 20.0% | 10.0% | id 6, en 4, ko 2 |
