# Log

A few lines per session: what was done, what broke, what's next.

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

**Next:** hand-check the parser's labels (`data/pilot_v1_review.txt`) · start the full corpus (Phase 4).
