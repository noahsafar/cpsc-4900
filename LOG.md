# Log

A few lines per session: what was done, what broke, what's next.

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

**Next:** verify `pipeline.py` reproduces `toy_v1.pt` in a fresh runtime · update Timos · start pilot data (Phase 2).
