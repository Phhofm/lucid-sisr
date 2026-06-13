# Don't Replace Tools. Compose With Them.

*A short essay on using LLMs where they help, and not where they don't.*

## The blockchain parable

I was a student when blockchains became *the* next thing. Every conference had a blockchain track. Universities built courses on crypto, tokens, and consensus algorithms. People were proposing decentralized this and decentralized that — decentralized companies, decentralized databases, decentralized storage. It was the hammer, and everything looked like a nail.

I had two recurring frustrations with that moment. The first was efficiency. Bitcoin's proof-of-work was — and still is — an astonishingly wasteful way to agree on who owns what. The protocol had to *artificially* make the puzzle harder, just so the government-sponsored mining farms and their power-hungry ASICs couldn't outrun everyone else. Money and electricity burned to roll a die. Ethereum's move to proof-of-stake (Cardano had it from the start, as I recall) made the point obvious in retrospect: the same consensus goal, almost none of the waste.

The second frustration was the substitution. We already had distributed databases. We already had consensus protocols at smaller scales (Raft, Paxos) that work fine for the actual problems people were trying to solve. Blockchains are genuinely good at *one* thing: a public, append-only, censorship-resistant ledger of who owns what. They're not a general-purpose replacement for existing infrastructure. The thing that was sold as a revolution was, in most of the use cases being pitched, a much worse version of something that already existed.

I feel the same way now about LLMs. They are remarkable. They are the most useful general-purpose tool I have added to my workflow in years. But every new technology goes through this cycle: it's the new thing, it gets used for everything, and only over time do we figure out *what it is actually for*. The point of this post is to argue for doing that figuring-out sooner rather than later — and one concrete way to do it, demonstrated on a real codebase.

## The temptation

When I sat down to clean up [LUCID](https://github.com/Phhofm/lucid) before the v3.0 release — a 5,200-line Python pipeline for filtering Single Image Super-Resolution training datasets, with a Tkinter GUI, a Tk-style dark theme, and three orchestrated stages (signal filter → ICNet scoring → ResNet18 dedupe) — the easy move was to dump the whole codebase into an LLM and ask "what's wrong with this?" The harder, more boring move was to set up a static analysis stack first, then bring in the LLM to read *its* output.

I did the harder move. Here's what I found.

## The strategy

The shape of the approach is a tooling pyramid. The base of the pyramid is the cheap, deterministic, locally-runnable tools that have been around for years. The apex is the LLM, doing the only thing it's genuinely better at than the tools below it: semantic judgment about a small, focused input.

```
                          ┌─────────────┐
                          │     LLM     │   ← reads tool output,
                          │   (judge)   │     validates, decides
                          └─────────────┘
                       ┌───────────────────┐
                       │  SonarQube /       │   ← duplication,
                       │  SonarCloud        │     complexity, code
                       │                    │     smells, trends
                       └───────────────────┘
                  ┌───────────────────────────┐
                  │  Semgrep / Opengrep        │   ← security,
                  │                            │     semantic patterns
                  └───────────────────────────┘
             ┌──────────────────────────────────┐
             │  ruff │ mypy │ bandit │ gitleaks │   ← fast, free,
             │                                    │     no server
             └──────────────────────────────────┘
```

The point is not that any of these is better than the others. It's that each is good at a different thing, and they're cheap enough to run all of them. Total runtime on a 5,200-line Python repo: under a minute. One-time install: ten minutes. Triage is the only step that costs human (or LLM) time, and triage is bounded by the number of findings — usually a few dozen, often fewer.

## The numbers

For LUCID specifically, here's the baseline I started with:

| Tool | Findings | Time |
|---|---|---|
| `ruff check` | 1 | 0.1s |
| `mypy lucid/` | 42 (38 env noise + 4 real) | 5s |
| `semgrep` (8 community rulesets) | 0 | 12s |
| `sonar-scanner` (SonarQube 9.9 CE, local Docker) | 14 | 7s |
| **Total raw findings** | **57** | **~25s** |

Then I — the LLM — read each finding, fetched the file context, and produced a verdict: real bug, real smell, false positive, or environment. Of the 57:

- **1 latent type bug** hidden behind a string annotation: `lucid/core/filtering.py` uses `"np.ndarray"` in two type hints but `numpy` was never imported. The `# noqa: F821` comment hid it from ruff. mypy caught it. Runtime would have been fine (forward-reference string), but the annotation was misleading and would have broken a future type-narrowing pass.
- **3 trivial mypy annotations** (typed dict, type for `record`, an assertion to narrow a `None` union).
- **1 redundant exception class** in `platform.py`: `except (AttributeError, PermissionError, OSError)` — `PermissionError` is a subclass of `OSError`, so the catch is duplicated. Sonar caught it.
- **3 string-literal duplications** (S1192): `"Sub.TLabel"` × 3, `"Secondary.TButton"` × 6, `"*.png"` × 3. Hoisted to module constants.
- **2 cognitive-complexity reductions** (S3776): `_score_tiles_gpu` (17→under 15) and `resolve_device` (19→under 15). Both are well-isolated functions; one extract each.
- **1 nested-conditional extraction** (S3358): the `batch_size if batch_size else recommend_batch_size(...)` was unpacked to a local.
- **8 findings deferred** with reasoning: 7 of them are `run_ingest`, `extract_tiles_to_staging`, `run_finalize`, `run_pipeline`, `compute_observations`, `run_dedupe` — large functions where a complexity refactor would touch dozens of test cases. The 8th is `run_pipeline` having 17 parameters (S107), which would require a `PipelineConfig` dataclass.
- **38 mypy env noise**: missing stubs for torch, torchvision, cv2, numpy, xxhash, platformdirs in the linting venv. Not a code defect; documented in `reports/DEFERRED.md`.

After the conservative pass:

| Tool | Findings | Delta |
|---|---|---|
| `ruff check` | 0 | −1 |
| `mypy lucid/` (real) | 0 | −4 |
| `mypy` (env noise) | 34 | −4 (mostly from `# type: ignore` adds) |
| `semgrep` | 0 | 0 |
| `sonar-scanner` | 7 (all deferred) | −7 |
| **Total** | **41** | **−16** |

What got fixed in the conservative pass was small but high-signal:

1. **`lucid/core/filtering.py` had two `np.ndarray` string annotations** with no `import numpy as np` in the file. The annotations worked at runtime (they were forward-references) but were a footgun for any future maintainer who tried to add a real `np.ndarray` check — and the `# noqa: F821` comment was actively hiding the issue from ruff. Adding the import fixed it.

2. **`lucid/gui.py` declared `self.process = None` and then assigned `subprocess.Popen` to it** — mypy correctly rejected the assignment because `Popen[str]` is not `None`. Typing the field as `Popen[str] | None` made both the type checker and the IDE happy.

3. **`lucid/core/filtering.py:run_ingest` calls `_score_staged_tiles(model_ctx, ...)` inside a dry-run guard** — but mypy can't see that the call only happens when `model_ctx` is not `None`, because the guard is on `image_stage and num_passed > 0`, not on `model_ctx is not None`. An `assert model_ctx is not None` makes the intent explicit and silences the checker.

4. **`lucid/platform.py:set_process_priority` caught `(AttributeError, PermissionError, OSError)`** — `PermissionError` is a strict subclass of `OSError`, so the explicit `PermissionError` is redundant. Sonar flagged it; dropping the redundant class tightens the contract.

The cumulative effect: the four functions that callers actually use (`run_ingest`, `run_dedupe`, `run_finalize`, the GUI's `_open_calibrate`) all became type-checker-clean, and the GUI now has named constants for its ttk styles so renaming one place is one edit instead of nine.

## What the tools *don't* see

The honest part of this post: the static analyzers do not, and structurally cannot, catch the most interesting bugs in a pipeline like this. They will not tell you:

- Whether the filtering heuristic actually implements the algorithm from the paper you cited.
- Whether your train/val/test split leaks samples.
- Whether the ICNet complexity threshold is calibrated for the dataset's distribution.
- Whether 16-bit TIFFs and 8-bit JPEGs are being normalized the same way.
- Whether the augmentation pipeline is deterministic given the same seed.

These are semantic correctness questions. Static analyzers see *patterns*. They see `range(desc=...)` because that's a pattern they have a rule for. They do not see "this Laplacian-variance threshold of 80.0 was tuned for ImageNet and probably over-prunes noisy natural images."

This is the LLM's actual job in the workflow. Not the spray-and-pray "review my whole codebase" prompt. The LLM should be reading the *filtering logic*, the *score normalization*, the *dataset splits*. That's the layer static tools literally cannot see.

## The token math

A "spray and pray" review of a 5,200-line Python repo requires ~50k–150k input tokens just to *read* the code, and the model still has to rediscover the patterns that ruff already encodes in 50 lines of regex. The static tools produce a structured SARIF or JSON report of <10k tokens. Reading and triaging *that* is a few thousand tokens, and the model isn't hallucinating bugs — it's evaluating reported ones against file context.

The ratio, on this repo, is roughly 10:1 in favor of the static-first path, and the answers are also more accurate. The static findings are deterministic and reproducible; the LLM hallucinations are not.

## What this looks like in practice

If you want to do the same thing on your own codebase, the minimum viable setup is:

```bash
# the fast, free, no-server tier
pip install ruff mypy semgrep
ruff check .
mypy . --strict
semgrep scan --config p/python --config p/security-audit --config p/owasp-top-ten
```

That alone catches most of what you're going to fix in a conservative pass. SonarQube via Docker adds code-quality trends and duplication detection at the cost of a one-time server setup:

```bash
docker run -d --name sonarqube -p 9000:9000 sonarqube:lts-community
# log in, create a project, generate a token, then:
docker run --rm --network host \
  -v "$(pwd):/usr/src" -w /usr/src \
  -e SONAR_TOKEN=<your-token> \
  sonarsource/sonar-scanner-cli:latest
```

After that, the loop is:

1. Run the tools. Save raw output to `reports/`.
2. Read the findings. For each one: real bug? real smell? false positive? environment?
3. Apply the high-confidence fixes. Document the deferred ones with reasoning.
4. Re-run. Show the delta.
5. *Then* bring in the LLM for the semantic review of the actual pipeline logic.

## Don't replace tools. Compose with them.

The pattern I keep seeing, in myself and in others, is to over-trust whatever is newest. Blockchains were the right answer to one specific problem and the wrong answer to most of the problems people tried to use them for. LLMs are the right answer to *some* problems, and I think we'll spend a few years figuring out which ones — but the answer is not "everything."

What I think the answer *is*, for software engineering specifically: **the LLM is a judge, not a scanner.** It's a thing you read the world through once you've already narrowed the world down. The world-narrowing is the part static tools have been doing well for fifteen years and will keep doing well, because it's the part that benefits most from being deterministic, fast, and reproducible.

So when you reach for the new thing, ask: *what existing tool am I about to replace?* If the answer is "all of them, because the new thing is more general" — be skeptical. The new thing is usually more general in some dimensions and much worse in others. The job is to figure out which dimensions you actually need, and to use the right tool for each.

That's the boring move. It is also, in my experience, the move that ships working software.

---

*Tools used in this post: ruff 0.15, mypy 2.1, semgrep 1.165, SonarQube 9.9 LTS Community, Docker. Worked example: [lucid-sisr](https://github.com/lucid-sisr/lucid-sisr), a 1,400-line Python SISR dataset filtering pipeline. Full triage report in `reports/TRIAGE.md`; deferred items in `reports/DEFERRED.md`.*
