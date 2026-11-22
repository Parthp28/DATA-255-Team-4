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
| | | | | | | |
| | | | | | | |
