# Log

A few lines per session: what was done, what broke, what's next.

## 2026-10-28

- Brought this log up to date (Oct 20 entry below).
- Corrected the week 2 notebook's Status line and "Findings so far": the pilot had 558 valid labels before the week 3 re-runs, not 561 (597 records minus 39 invalid), with 176 compliant (32%, not 31%). It was an arithmetic slip in the write-up; the labels themselves and the final week 3 numbers (584 valid, 198 compliant) don't change.
- Started the hand-check of the parser's labels on the 46 outputs in `data/pilot_v1_review.txt`, compared against the final labels in `data/pilot_labels.jsonl`.

## 2026-10-20

- Wrote the midterm progress report as a working draft of the thesis, in the department's LaTeX template (Overleaf): introduction, background, methodology, results, related work, conclusions and future work, plus an example pilot prompt in the appendix.
- Results chapter: ASR by framing (2% for `role_play` to 91% for `completion`), the final pilot labels (584 valid, 198 compliant), and the probe table and figure: AUROC 0.87, 0.91 and 0.94 at layers 9, 18 and 27; 0.86 and 0.90 within `fake_system` and `urgency` at layer 27; family-only 0.85 and TF-IDF 0.86 overall.
- Checked all 20 references against their sources; four sentences were reworded to match what the papers say.
- Limits stated in the report: one model, 100 cases from one benchmark, only two framings in the 20–80% band, three layers fixed in advance, and no hand-check of the parser's labels yet.
- Sent the report to Timos for approval before the Canvas submission; the email also gives him the go/no-go result. Answered the midterm questionnaire (milestones so far, goals for the remaining weeks).
- Schedule unchanged: full corpus Oct 26–Nov 6, baselines and held-out families Nov 9–19, steering (stretch) Nov 30–Dec 6, writeup Dec 7–11.

## 2026-10-14

- Started `notebooks/week3_probes.ipynb`: section 1 finishes the pilot labels, sections 2–5 train the pilot probes and baselines, and the last cell records the go/no-go.
- Re-ran the 39 invalid outputs with a 320-token cap instead of 160, and generated the 3 missing records. All 39 re-runs start with their old output. 21 of them are compliant, mostly tool calls the old cap had cut off; 2 are refusals; 16 stay invalid (15 call the user's tool again, 1 calls a tool that doesn't exist).
- Final pilot: 584 of 600 prompts have valid labels, 198 compliant (34%). `fake_system` (0.43) and `urgency` (0.39) are the two families in the 20–80% band. Labels are in `data/pilot_labels.jsonl`.
- Probes (L2 logistic regression on final-prompt-token activations, 5-fold CV grouped by case, C chosen by nested CV): AUROC 0.87, 0.91 and 0.94 at layers 9, 18 and 27.
- Template check: the family alone scores 0.85, but with the template fixed, layer 27 still scores 0.86 within `fake_system` and 0.90 within `urgency`. Layer 9 drops to 0.68 and 0.62 there.
- TF-IDF on the prompt text: 0.86 overall, 0.72 and 0.74 within the two families. Layer 27 is higher in every column.
- **Go/no-go: go** with the core plan: full corpus, per-layer sweep, the three baselines, and held-out attack families.
- Saved all 584 activations to one file, `cache/pilot_acts.npz`, so later notebooks can skip the 8-minute load of 584 separate files.

## 2026-10-07

- Pilot v1 on InjecAgent: 200 prompts (100 cases × base and enhanced), labeled by InjecAgent's parser. 19 complied, 169 resisted, 12 invalid, so about 10% compliance among valid outputs (6–13% per cell), well below the 20–80% band. The defensive system prompt idea is dropped, since the problem is the opposite.
- Pilot v2: four new framings on the same 100 cases (fake system message, urgency, role play, fake completion), with generation stopped at the first `Observation:`. ASR now ranges from 2% to 90% depending only on the framing, with `fake_system` and `urgency` in the band. Activations cached for all 600 prompts.

## 2026-09-25

- GitHub repo created. Week-1 notebook cleaned up: sections 7–9 cover the toy run, and the summary is at the end.
- Confirmed there are no other CPSC 4900 deadlines; the proposal's schedule (go/no-go Oct 19, writeup Dec 7–11) is self-set.
- Requested access to Llama-3.2-3B-Instruct on Hugging Face.
- Moved the reusable code into `src/pipeline.py` on Drive, so a new runtime needs one setup cell instead of rerunning the week-1 notebook.
- Toy run reproduces exactly on a second pass. Reading the responses gave the first labeling rules: complied means the injected instruction was carried out, even if the task was also done; quoting or summarizing the injection as content counts as resisted.
- First end-to-end run on the 20 toy prompts: 8 of 10 injections complied, 2 resisted. With this much compliance, the pilot needs a defensive system prompt to get balanced labels.
- 20 toy prompts: 10 injected/clean pairs built by one helper, so each clean prompt differs from its injected twin only by the injected line. Tasks: summarize, translate, question answering, extraction; injections at the start, middle, and end, in six styles.
- Notebook annotated with section headers and findings; transformers pinned to 5.17.0.

## 2026-09-24

- Qwen3-4B runs on a Colab T4 in float16. Greedy decoding is deterministic.
- Hook extraction matches the model's own `hidden_states` exactly: layer ℓ = `hidden_states[ℓ+1]`; the last layer matches only after the final RMSNorm.
- Activation cache saves to Google Drive and reloads identically, with model, dtype, GPU, and library versions recorded in each file.

## 2026-09-18

- Proposal finalized and submitted: three pages, 14 references.

**Next:** finish the hand-check and record the agreement number · start the full corpus (Phase 4, Oct 26–Nov 6): about 2,000 prompts from InjecAgent, AgentDojo and BIPIA, with the LLM judge for the last two, and the sweep over all 36 layers.
