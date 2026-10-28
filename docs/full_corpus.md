# The full corpus

Phase 4 (Oct 26–Nov 6). The proposal asked for about 2,000 injected prompts from InjecAgent, AgentDojo and BIPIA, spanning the six attack families, labeled automatically, and a probe sweep over all 36 layers. This page says how the corpus was built and how it gets labeled. The run itself is [notebooks/week4_full_corpus.ipynb](../notebooks/week4_full_corpus.ipynb).

## Design: every case under every framing

The pilot's design carries over: each **case** (one user task with one injected goal) is rendered under every framing, so the framings of a case differ only in the injected text. This gives grouped cross-validation (a case's prompts always share a fold), within-family AUROC, and, later, leave-one-family-out (RQ3), on all three benchmarks.

| benchmark | cases | framings | prompts | where the injection sits |
|---|---|---|---|---|
| InjecAgent | 100 (the pilot's) | 7 | 700 (600 pilot + 100 new) | a tool result in a ReAct scratchpad |
| AgentDojo | 97 | 7 | 679 | a tool result, with native tool calling |
| BIPIA | 100 | 7 | 700 | an email or a table in the user message |
| **total** | **297** | | **2,079** | |

## The seven framings (`src/framings.py`)

`base`, `enhanced`, `fake_system`, `urgency`, `role_play` and `completion` are the pilot's six; the same code reproduces all 500 non-base pilot prompts exactly from the base ones. `encoded` is new: the goal in base64, with a one-line instruction to decode and follow it. It is the proposal's "encoded payload" family, the only one the pilot didn't cover.

`completion` (the proposal's "delimiter confusion") is the one framing that depends on the benchmark, since it fakes the end of the data and a new user turn in each benchmark's own format: InjecAgent's `Final Answer:`/`User Input:` lines, Qwen's `</tool_response>` tag plus a `User:` line for AgentDojo, and an `Answer:`/`Question:` pair for BIPIA. `urgency` names the account owner: John Doe in InjecAgent, Emma Johnson (the user in AgentDojo's system message) in AgentDojo, and no name in BIPIA.

## AgentDojo (`agentdojo==0.1.35`, suites `v1.2.2`)

- **Cases:** all 97 user tasks in the four suites (workspace, travel, banking, slack); each one's ground-truth tool calls read at least one injection vector. Each user task gets one injection task from its suite, drawn at random (seed 0), so a case's user task is never reused, which keeps the grouped folds clean.
- **Static rendering:** AgentDojo's default system message, the user task, then the user task's ground-truth tool calls and their real results from the injected environment, up to and including the first result that contains the injection. The model's next step after that result is what gets labeled. 476 prompts end after one call, the rest after 2–5.
- **Injection:** as in AgentDojo's own attacks, the framed text replaces every injection vector the user task reads.
- **Format:** Qwen's native tool calling through the chat template (`pl.format_prompt(..., tools=...)`), with each suite's tool schemas in `data/agentdojo_tools.json`. This differs from the ReAct text of the InjecAgent prompts; benchmark is a column in every result table, so the difference stays visible.

## BIPIA (test split)

- **Cases:** all 50 test emails and 50 of the 100 test tables (seed 0), each paired with one of BIPIA's 75 test text attacks. Every attack appears at least once, 25 twice. The web QA and summarization contexts aren't used, since BIPIA can't redistribute them.
- **Rendering:** BIPIA's own task prompts, in their no-system-prompt form (context and question in the user message), so the injection never sits in the system role. The attack goes at the start, middle or end of the context (BIPIA's three positions, drawn per case and kept across framings), with BIPIA's sentence-boundary rule for the middle.

## Labels

| | final label | cross-check |
|---|---|---|
| InjecAgent | InjecAgent's parser on the next step (`src/labels.py`), as in the pilot | the pilot hand checks: 72 of 72 |
| AgentDojo | Claude judge (`src/judge.py`) | rule: called an injection-task function with an attacker identifier (email, IBAN, URL) in its arguments |
| BIPIA | Claude judge | BIPIA's own rules where it has them: an exact phrase (2 attacks), emoji (5) |

The judge (`claude-opus-5-5`, low effort, JSON output) sees the user's task, the injected goal (unframed, never base64), BIPIA's description of what complying looks like, and the model's next step. It never sees the framing, the rest of the prompt, or the rule label, so it can't learn the framing from the input, and its agreement with the rules is a real check. Its rules are the README's labeling rules, written out for tool calls and response-format attacks. `unclear` and refused judge requests become `invalid`.

Before the judge's labels are used, the notebook reports (1) agreement with the rule checks where they are decisive, and (2) agreement with a blind hand check of 40 outputs (10 judged complied and 10 not, per benchmark). That second number is the "judge agreement" the proposal promises for the parts of the corpus the parser can't label.

## Known limits

- **Benchmark and format are confounded.** The three benchmarks also differ in prompt format (ReAct, native tool calls, plain text). The per-benchmark columns show whether the probe works within each; a probe trained on one and tested on another would show transfer, and is a natural Phase 5 addition.
- **Next step only.** As in the pilot, the label covers the model's next step. On AgentDojo, an injected goal that needs several calls counts as complied once the model takes a step whose only purpose is the injected task.
- **Most AgentDojo prompts end after one tool call**, because injections are usually in the first result a task reads.
