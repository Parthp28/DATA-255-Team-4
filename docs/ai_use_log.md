# AI and Tool Use Log

Owner: Farzaan Dalwai (checks at every evening sync; Khushi Donda compiles Appendix B from it). Every member logs their own use on the same day.

Rules:
- List only tools that were actually used, with the exact product and model name shown in the tool.
- Record a representative prompt or a prompt category, not full transcripts.
- Every entry names the artifact affected and the human verification performed. An entry without verification is incomplete.
- No AI output enters the report unedited. Every reported number comes from `scripts/evaluate.py` at the frozen commit.
- Grammar tools are logged only if used.

| Date | Member | Tool and model | Purpose | Prompt category or representative prompt | Artifact affected | Human verification performed |
|---|---|---|---|---|---|---|
| 2026-11-19 | Parth Patel | Claude (claude.ai), Claude Opus 5.5 | Planning | Turn the team's project specification, rubric, and professor feedback into a master execution plan and a five-day execution plan | Master Execution Plan doc; `DATA255_Final_Project_Execution_Plan.pdf`; starter `docs/*.md`; `report/main.tex` skeleton | Reviewer: ____ . What was checked and changed: ____ |
| 2026-11-19 | Parth Patel | Claude (claude.ai), Claude Opus 5.5 | Planning and tooling | Revise the execution plan's compute schedule; draft a queue-runner script with tests and a README skeleton | `DATA255_Final_Project_Execution_Plan_v2_Two_GPU.pdf`; `scripts/run_queue.py`; `tests/test_queue.py`; `README.md` | Reviewer: ____ . Tests run and what was changed: ____ |
| 2026-11-21 | Khushi Donda | Claude Code (Anthropic CLI), Claude Sonnet 5.5 | Tooling and repository setup | Read the execution plan and the starter kit, compare the kit with the extracted files, build the `.gitignore`, run the existing tests, and prepare the first commit | `.gitignore`; initial commit `de661ab` (starter kit files unchanged) | Reviewer: Khushi Donda. File list and compare results shown before commit; byte-for-byte comparison of the kit against the extracted files and staged blobs; `pytest` run (7 passed); explicit approval given before the commit and push. |
| 2026-11-21 | Khushi Donda | Claude Code (Anthropic CLI), Claude Sonnet 5.5 | Code and tests | Implement the paired Jensen-Shannon consistency loss (decision D12) and tests T-C1 to T-C6 from the execution plan | `src/training/losses.py`; `tests/test_losses.py`; PR #1 (squash-merged as `27a3fbc`) | Reviewer: Khushi Donda; cross-review by Kaushik Hariharan requested and still pending. Tests run: 19 passed on two PyTorch versions. Gradients checked against a float64 reference and `torch.autograd.gradcheck`. Three deliberately broken variants confirmed the tests fail when they should. A float32 downcast defect found during review was fixed. Empty-batch handling added after review. |
| | | | | | | |
| | | | | | | |
