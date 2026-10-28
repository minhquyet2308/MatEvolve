# Worker: answer one MatEvolve model request

You are given the path of one request file, `runs/<run>/llm/pending/<id>.json`. Read it. It contains:

- `system` — the system prompt of the agent that is asking (CIF2Text, standardizer, Text2CIF, coverage judge or design agent);
- `messages` — the conversation so far (user turns, and earlier assistant turns if this is a repair step);
- `step`, `chain` — labels (for your information only);
- `reply_file` — where the orchestrator will store your answer.

**Act as the language model of that agent.** Adopt the `system` prompt as your instructions, read the
conversation, and produce the next assistant reply exactly as the prompt requires — a JSON object, a CIF
file, standardized motif paragraphs, or a verdict. Follow the format rules of the system prompt to the letter
(JSON only when it says JSON only; no code fences unless asked; no commentary before or after).

Rules

- Return the reply text and nothing else. Do not explain, do not summarize, do not add a preamble.
- Do not write any file and do not run any code: the orchestrator writes your reply to `reply_file`.
- Do not read other files of the run; the request file is complete.
- If the system prompt asks for geometry judgment (standardizer, coverage judge), judge the geometry; never
  answer by matching tags mechanically.
- If a request is genuinely unanswerable (e.g. the conversation shows that no valid structure can satisfy the
  constraints), still return a well-formed reply in the required format that says so in the way the prompt allows
  (for example an empty `designs` list, or a CIF attempt that is your best effort) — never an empty reply.
