# G2 visual SAMPLE review (four windows)

**This is a visual SAMPLE review of 28 images. It is NOT a full-dataset visual audit, and it is NOT formal G1 or
G2 approval.** The formal G1 and G2 gates are not approved by this record.

| | |
|---|---|
| Date | 2026-11-23 (PST) |
| Reviewer | Khushi Donda (assigned reviewer for G2, Execution Plan v2 Section 16.1) |
| Result recorded | **Sample review: PASS** |
| Scope | One sheet, `data/render_v1/figure1_sheet.png`: 4 windows x 7 renderings (R0 to R6) = 28 images |
| Reference | Execution Plan v2, Section 5.2 (frozen rendering registry `render_v1`), and `data/render_v1/region_geometry.json` |

## 1. What the human reviewer did

Khushi Donda states that she visually inspected the four-window figure sheet in Preview (macOS), at its native
size of 1616 x 926 pixels. The visual review checked **compliance with the Section 5.2 specification and obvious
defects**. It did **not** assess formal readability or usability of the charts for a model or for a human reader.

Based on the sheet and on the automated pixel audit in Section 2, she approved recording the sample review as PASS.

## 2. What the automated checks did (Claude Code, not the human review)

The 264 pixel checks below were written and run by **Claude Code** (Section 8). They are **not** part of Khushi
Donda's manual inspection. They support it.

- **Audit script:** `audit_sheet.py`, SHA-256 `0ed25b8a30d6a5010cea3e169538c02d38948894579578a5dfc72b338205280f`
  (Python 3.11.15, Pillow 13.0.0, numpy 2.4.6). See Section 7 for why it is not included in this repository.
- **Result: 264 checks passed, 0 failed.** They break down as 224 per-image checks (8 checks x 28 tiles) and 40
  cross-rendering checks (10 checks x 4 windows).
- **Per-image checks (8):** only the permitted colours are present (no text, no anti-aliasing); nothing is drawn
  outside the plot area (R3's axes excepted); grid rows and columns equal the geometry file; 30 candles with a
  readable up/down direction; widest candle body equals the geometry value (4 px, R5 2 px); candle ink spans the
  full price panel; 30 volume bars filling the volume panel; no volume-gray pixels outside the volume panel.
- **Cross-rendering checks (10 per window):** identical up/down sequence in R0 to R6; R1 differs from R0 only on
  R0-white pixels; R2 differs only where R0 had gridlines; R3 differs only in the left and bottom axis bands;
  R4 differs only on candle pixels, and its volume bars equal R0's; R5 changes only R0 candle pixels to white or
  grid colour, adds no ink beyond R0, leaves the volume panel identical, and keeps every candle's wick extremes.

## 3. The four sampled windows

| Row | Window | Split | Label |
|---|---|---|---|
| 1 | IWM_2011-05-24 | train | UP |
| 2 | DIS_2015-03-12 | train | NEUTRAL |
| 3 | WFC_2018-07-06 | val | DOWN |
| 4 | GS_2021-12-31 | test | UP |

The sheet has no labels. Columns were identified as R0 to R6 from left to right from what each column shows.

## 4. Observations per rendering (all four windows)

| Rendering | Specified factor | Observed |
|---|---|---|
| R0 | Reference: white background, light-gray grid, green up and red down candles, 60% bodies, 1 px wicks, gray volume bars, price panel on top and volume below, 8 px margins | As specified |
| R1 | Background tint only | Differs from R0 only on R0-white pixels; tint (226,232,240) |
| R2 | Gridlines removed | Only R0's grid pixels (221,221,221) become white |
| R3 | Left and bottom axis spines with ticks, no numerals | 461 changed pixels per window, all in the left and bottom bands, one dark-gray colour (64,64,64), no text |
| R4 | Blue up candles, orange down candles | Only candle pixels differ (blue (31,119,180), orange (255,127,14)); volume bars identical to R0 |
| R5 | Candle body 35% of slot, wicks unchanged | Bodies 2 px; wick extremes unchanged; volume panel identical to R0 |
| R6 | 24 px margins, 176 x 176 plot, same canvas, not resized | Plot at 24 to 199; price and volume panels scaled as in the geometry file |

Across all renderings: same 30 candles and same up/down sequence per window, 30 volume bars, no text.
**No unexpected visual differences were identified outside the permitted rendering factors.**

### Documented non-defects

1. **R0's fifth horizontal gridline (row 181) extends into the volume panel.** Four horizontal gridlines are in
   the price panel and one is in the volume panel, behind the volume bars. Section 5.2 says "5 horizontal, 6
   vertical lines" without naming panels, so this is the data pipeline owner's reading of the spec.
2. **R5's narrower candle bodies expose the gridlines behind them.** This is expected: a thinner body no longer
   covers the grid.

## 5. Initial audit-check failures and their correction (transparency)

The first run of the audit script reported **224 passed and 28 failed**. All 28 failures came from two errors in
the audit script, not from the figure sheet, which was not changed:

- **24 failures, grid rows:** the script looked for horizontal gridlines only inside the price panel, so it missed
  the fifth line at row 181, which lies in the volume panel (4 windows x 6 renderings with a grid = 24).
- **4 failures, R5 versus R0:** the check required R5's changed pixels to be white only, but a narrower body also
  reveals gridline gray where the R0 body had covered it (one failure per window).

The two checks were corrected and the R5 test was strengthened (R5 ink must be a subset of R0 ink, and every
candle's wick extremes must be unchanged). The rerun gave **264 passed, 0 failed**. The script version that
produced the 28 failures was overwritten and was not preserved; only its output was kept.

## 6. Sample versus automated coverage

- **Visually inspected by the human reviewer:** the **28** images on the sheet.
- **Not visually inspected:** the other **345,422** of the 345,450 registered PNGs.
- **Automatically verified, not visually reviewed:** all **345,450** PNGs, by `scripts/verify_g1g2.py --all-images`
  (decoded and hashed, 224 x 224 x 3 uint8, registry hash match), plus 200-window spot checks and fresh-process
  re-render checks described in the verifier's report.
- A visual review of any further images has not been done.

## 7. Evidence

All hashes are full SHA-256 values computed on 2026-11-23 from the files named (relative to a restored checkout
of `origin/main` at `9d1a49c23779f5e1113ceac32d2fb6b5d2fc3d73`).

| Artifact | SHA-256 |
|---|---|
| `data/render_v1/figure1_sheet.png` | `0e41cf9673469c72f3ad7b2aff2ec37116d961637af299d7e52208b7b630358a` |
| `data/render_v1/region_geometry.json` | `2897ae607fd5a4c8613895c2f92dd7071177387682f453d2ecb482dcba317a17` |
| `data/render_v1/render_manifest.json` | `297cb3dff4800acd01482972743a26e967db770bf3059d60308c670832d02565` |
| `data/render_v1/render_registry.parquet` | `85c87af00e7d4ed2bb7d48c04f8b069a5a4f60eaaf7c6a8fd81b6d2209274444` |
| `data/data_v1/windows.parquet` | `b10cafb53fa9b7ab02e5df1bc2955ca5d018d8cfeda2ae8641154c263971dbbe` |
| `outputs/gates/g1g2_report.json` (standard run) | `9e0fc9dbff792dff09dba6a8dfb99f863d80eb8d667d2240211877228cc96806` |
| `outputs/gates/g1g2_report_all_images.json` (all-images run) | `5a3ac30ffbd3f8e44ddf3e73829f64d27b49a00ed6d71f97f39cc50107fff034` |
| Master `SHA256SUMS` of the downloaded dataset | `1b5d0a8b09f478290905db123f9c98c854d4f91ca30b2ef6d7dd97e8889747dc` |
| `restore_dataset.sh` (read in full and audited before it was run) | `0314846b3ff7d55a2766b94caea0085eb38c5096f76d62977edf54261407d5a0` |
| `audit_sheet.py` (final version) | `0ed25b8a30d6a5010cea3e169538c02d38948894579578a5dfc72b338205280f` |

The figure sheet hash equals its entry in the dataset's master `SHA256SUMS`.

Related automated results (run by Claude Code in a separate restored checkout; these are not Parth Patel's
independent verification):

- Dataset restore: 33 of 33 master entries and 21 of 21 shard tars matched; raw archive hash matched; exactly
  345,450 PNGs extracted.
- `pytest` with the production data present: 341 passed, 0 failed, 0 skipped (torch 2.15.0, timm 1.0.31).
- `python scripts/verify_g1g2.py --data-dir data`: 30 PASS, 0 FAIL.
- `python scripts/verify_g1g2.py --data-dir data --all-images`: 30 PASS, 0 FAIL, about 123 seconds. Its report was
  written to a separate file so the standard report was kept.
- The two verifier reports are local, untracked files and are not part of this commit. The dataset was not
  modified by this review.

### The audit script is not included in this repository

`audit_sheet.py` is **not** committed here because it contains a hard-coded absolute path on the reviewer's
computer (the dataset location, line 4). Editing it to remove the path would produce a different file from the one
that produced the 264 passing checks, so its exact SHA-256 is recorded above instead. The checks are described in
Section 2. A path-parameterised copy can be added in a later change if the team wants one.

## 8. AI assistance (for the AI-use disclosure)

- **Tool and model:** Claude Code (Anthropic CLI), Claude Sonnet 5.5.
- **What it did:** opened the figure sheet in Preview for the reviewer; wrote and ran the audit script (including
  the first version with the two errors in Section 5); restored the dataset with the audited script after the
  reviewer approved; ran the test suite and both `verify_g1g2.py` runs; drafted this record from the reviewer's
  instructions.
- **What it did not do:** it did not make the pass decision. The reviewer made the decision and specified the
  requirements for this record. This record is a draft for her review in the pull request.
- **Human verification performed:** the reviewer's visual inspection of the sheet (Section 1) and her acceptance of
  the audit results as supporting evidence. Line-by-line review of the audit script by the reviewer is not claimed.
- The corresponding row in `docs/ai_use_log.md` has not been added by this change; it is proposed separately for
  Farzaan Dalwai's review.

## 9. Formal gate status

- **G1 (data gate):** not approved by this record. Requires Parth Patel's independent T-L3 and T-L5 and his sign-off.
- **G2 (rendering gate):** not formally approved by this record. This is a sample review only.
- Tags `data_v1`, `render_v1` and `spec-v1` have not been created.
- No GPU job has been run.
