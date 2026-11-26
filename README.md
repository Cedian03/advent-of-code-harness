# harness

An Advent of Code solving harness. It archives past puzzles, scaffolds a
solution, tests it against the examples embedded in the statement, and runs it
on your real input — with a small enough loop to iterate on a puzzle quickly.

## Layout

```
crates/
├── aoc-archive/   # library: HTTP client, HTML→Markdown, puzzle model, scraper
├── llm-http/      # library: minimal OpenAI-compatible chat client
└── harness/       # main binary: archive + solve workflow
.aoc/              # local puzzle cache (gitignored)
solutions/         # one directory per solved day (gitignored, yours to keep)
```

## Architecture

Three seams keep the loop independent of where puzzles live and which model
runs it:

- **`Aoc` trait** (`crates/harness/src/aoc.rs`) — `tasks`, `load`, `evaluate`,
  `run`, `record`, `open_part`, `submit`. `ArchiveAoc` implements it from local
  files and only records answers; its `submit` refuses. `RemoteAoc` wraps it,
  upgrades the archive with a Session when one is present, and overrides
  `submit` only for a submit-enabled invocation.
- **`Model` trait** (`crates/harness/src/model.rs`) — one streaming turn:
  text deltas plus any tool calls. `OpenAiModel` is a thin adapter over the
  `llm-http` crate; `ReplayModel` serves canned replies from a local JSON file
  for offline runs and tests.
- **`Agent` loop** (`crates/harness/src/agent.rs`) — a Pi-style tool-calling
  loop: stream a model turn, execute the tools it asks for, append the results,
  repeat until it stops; then verify, run the input, and record the answers.

The loop only sees the traits, so a live AoC client or a different model can be
dropped in without touching it.

`llm-http` covers exactly one endpoint, `POST {base_url}/chat/completions`,
with typed `ChatRequest`, `Message`, `ToolDefinition`, `ToolCall`, and `Usage`.
Provider error bodies (401/404/429/...) are preserved in `Error::Http`, and
non-2xx responses are not turned into opaque transport errors. Tool calls are
assembled from streamed deltas and executed by the harness.

## Data policy

Advent of Code is free to use, not free to copy. Puzzle text and inputs are
Eric Wastl's work, and the site asks that they not be redistributed. This repo
therefore:

- writes everything it downloads to a **gitignored** `.aoc/` directory;
- keeps your inputs in `.aoc/` too, never in `solutions/`;
- makes at most one request per second (configurable upward, never below 1);
- caches raw responses so repeated runs do not hit the server.

Do not commit `.aoc/` or use it to publish puzzle text or inputs.

## Quick start

```sh
# Archive every past event (2015–2025; includes the current year in December)
cargo run -p harness -- scrape

# Archive a single Day, or several Days/Events in one go
cargo run -p harness -- scrape 2025/1
cargo run -p harness -- scrape 2025 2024/3

# Spans are the shell's job — there is no range syntax
cargo run -p harness -- scrape 20{15..25}
cargo run -p harness -- scrape 2025/{1..25}

# One row per Event, or one row per Day when a Selection is given
cargo run -p harness -- list
cargo run -p harness -- list 2024 2025/1
```

A Day is always fully qualified (`2025/1`), an Event is `2024`, and repeating
addresses builds a **Selection**: duplicates collapse and Days run in ascending
order. `scrape 2015-2025` is not an address; the error names the brace form
above. PowerShell has no brace expansion, so write the span out there (or
generate the addresses with a loop).

### Session cookie (optional)

Public pages only carry Part One: AoC reveals Part Two after you solve Part One
while logged in. A session cookie also unlocks your puzzle inputs and the
answers you have already submitted.

Get the value from DevTools → Application/Storage → Cookies →
`adventofcode.com` → `session`. Put it in `.env` (gitignored) or the
environment, then scrape the day:

```sh
# .env supports both "AOC_SESSION=..." and "export AOC_SESSION=..."
echo 'AOC_SESSION=...' >> .env

cargo run -p harness -- scrape 2025/1
```

The cookie is read at runtime and never written to disk. It also adds Part Two
statements for days that account has solved, and previously submitted answers.

### Adding your real input

Inputs are personal to your AoC account, so without a session they cannot be
fetched. Import yours instead — it goes to `.aoc/<year>/<day>/input.txt`
(gitignored, never copied into `solutions/`):

```sh
# From a file
cargo run -p harness -- input 2025/1 --file ~/aoc-inputs/2025-01.txt

# Or paste it and press Ctrl-D
cargo run -p harness -- input 2025/1

# One step while scaffolding a solution
cargo run -p harness -- init 2025/1 --input ~/aoc-inputs/2025-01.txt
```

Import the bytes exactly as AoC gave them (keep the trailing newline). Verify
with `harness list 2025`; the `Inputs` column should say `yes`. After that,
`run --examples` checks the examples and `solve`/`run` execute on the real input
and compare against recorded answers.

The full input is not pasted into the model prompt automatically: the generated
program reads it from stdin, and the model can inspect it with `read_input`.
That tool sends the input to your configured model provider, so treat the
solve step like any other LLM interaction.

## Solving loop

```sh
# 1. Scaffold solutions/2024/01/solve.rs
cargo run -p harness -- init 2024/1

# 2. Read the statement (or grab the JSON task bundle)
cargo run -p harness -- show 2024/1
cargo run -p harness -- show 2024/1 --json
cargo run -p harness -- show 2024/1 --part 2

# 3. Edit solve.rs, then check it against the detected examples
cargo run -p harness -- run 2024/1 --examples

# 4. Run it on the real input
cargo run -p harness -- run 2024/1
cargo run -p harness -- run 2024/1 --part all
```

Every command addresses the same unit — the Day — as `YYYY/D`. A Part is a
filter, not part of the address: `--part 1|2|all` (`run` defaults to part 1).
Asking for a Part that is not available is an error: the Day is Unreleased,
Part Two is Sealed until Part One is accepted, or the Part is simply not in the
archive.

`run` also compares the answer with the one AoC recorded for you, when the
archive was scraped with a session.

### Recording vs submitting

There are no local/remote modes. `solve` reuses the archive, fetches only what is
absent, and treats submission as a separate, opt-in capability.

```sh
harness solve 2025/1              # record answers locally (the default)
harness solve 2025/1 --submit     # also post them to adventofcode.com
harness solve 2025/1 --no-submit  # explicit record-only, even if the default changes
```

`--submit` requires `AOC_SESSION`. Without a Session, `solve` still works when the
Day is already Captured and the input is present; it warns that Part Two cannot
be unsealed. It errors when the input is missing and cannot be fetched.

Wrong submissions follow AoC's cooldown; a `TooSoon` response is returned to the
agent as a tool result with the remaining wait, and the loop continues. After a
correct Part One the day page is refetched and Part Two is revealed **into the
same agent session**: `solve.rs` and the conversation are kept, and the loop
continues with a delta prompt instead of starting over. With `--submit`, the
agent may call `submit` or `run_and_submit`; nothing is posted from a previous
run's `answers.json`, rejected answers are remembered in `submissions.jsonl`,
and the Star is verified on the calendar.

`solve` covers both Parts by default (`--part all`). `--part 1` stops after Part
One; `--part 2` requires Part Two to be available. A no-submit run that records
Part One and stops prints `part 2 is sealed; re-run with --submit`.

`--wait[=DUR]` holds until the Day is released (default ceiling 24h, e.g.
`--wait 90m`). `scrape` accepts it too; both require exactly one Day.

`--replay` implies `--no-submit` and never fetches, so the Day must already be
Captured.

Inspect cached Star state with `list`; `--stars` refreshes it from the site:

```sh
harness list                 # Stars column from remote.json, no network
harness list 2025/1          # one Day
harness list --stars         # refresh the calendar first; requires AOC_SESSION
```

Remote state is cached at `.aoc/<year>/<day>/remote.json`:

```json
{
  "year": 2025,
  "day": 1,
  "stars": 1,
  "part1": {"solved": true, "answer": "1078"},
  "part2": {"solved": false, "answer": null},
  "part2_unlocked": true,
  "fetched_at": 1795646252
}
```

### Let a model solve it

```sh
# Next unsolved day (prefers days with a real input)
cargo run -p harness -- next

# .env in the repo root is loaded automatically (see .env.example)
cargo run -p harness -- solve 2024/1 --no-submit   # record locally
cargo run -p harness -- solve 2024/1 --submit      # post to AoC (needs AOC_SESSION)

# Offline: replay canned assistant replies from a local JSON file
cargo run -p harness -- solve 2024/1 --replay /tmp/replies.json
```

Each turn the model can call tools instead of dumping a whole file:

| Tool | Purpose |
| --- | --- |
| `read_file` / `list_files` | inspect the solution workspace |
| `write_file` / `edit_file` | create and refine `solve.rs` |
| `read_input` | inspect your personal input (truncated; sent to the model) |
| `run_tests` | compile `solve.rs` and run the detected examples |
| `run_part` | run on the real input and get the answer |
| `submit` / `run_and_submit` | with `--submit`: post an answer, or run and post if the run produced one |

The loop streams the model, executes tool calls, appends each result as a tool
message, and asks again. Paths are confined to the day's solution directory; there is no shell tool.
When the model stops calling tools, the harness compiles and tests the file one
last time; on success it runs the real input for every captured part, compares
against answers already recorded for the account, and writes the answer to
`.aoc/<year>/<day>/answers.json`. Nothing is submitted to adventofcode.com unless
the model calls a submit tool.

A full Rust block in the final chat message is still accepted as a fallback for
models that ignore the tool instructions.

`--turns` caps model turns (default 12). Every lifecycle step is an event
(`turn_start`, `model_completed`,
`tool_started`, `tool_finished`, `evaluated`, `finished`, …). Print them as JSON
lines on stderr with `--events`:

```sh
cargo run -p harness -- solve 2024/1 --no-submit --events --quiet 2> events.jsonl
```

### Model configuration

`.env` (gitignored) is read from the repo root; plain `export KEY=value`
lines work. The harness uses an OpenAI-compatible protocol:

| Variable | Meaning | Default |
| --- | --- | --- |
| `HARNESS_MODEL` | provider model id | required (`--model` overrides) |
| `HARNESS_API_KEY` | provider API key | required; `OPENAI_API_KEY` also accepted |
| `HARNESS_BASE_URL` | OpenAI-compatible base URL | `https://api.openai.com/v1` |

`.env` does not override variables already exported in your shell. The harness
warns when that happens, and prints the effective model and endpoint at the
start of every solve. Common base-URL mistakes are corrected automatically:
`https://openrouter.ai/api` and `https://openrouter.ai` become
`https://openrouter.ai/api/v1`, and a pasted `/chat/completions` suffix is
removed.

The default base URL is **OpenAI**, so a key from OpenRouter, Groq, Together,
Ollama, LM Studio, etc. returns **HTTP 401 Unauthorized** until you set the
matching base URL:

```sh
HARNESS_BASE_URL=https://openrouter.ai/api/v1
HARNESS_MODEL=anthropic/claude-sonnet-4
```

Model calls have a per-request timeout (`--timeout`, default 180s) and are
streamed, so a slow or queued provider is visible instead of looking hung:

```
model nvidia/nemotron-3.5-super-vl:free via https://openrouter.ai/api/v1 (timeout 180s, max tokens: provider default)
turn 1: asking nvidia/nemotron-3.5-super-vl:free (4.7 KB prompt)
turn 1: waiting for nvidia/nemotron-3.5-super-vl:free... 10s
turn 1: first token after 12.4s
turn 1: generating... 1.8 KB
turn 1: model replied in 48.9s (2.4 KB, 1 tool call(s))
turn 1: tool write_file
```

Without streaming, the HTTP response only arrives after the whole completion,
so a free/queued model looks dead for minutes. There is no artificial output
cap by default: the request omits `max_tokens` and lets the model use its own
limit. `--max-tokens N` adds a cap if you want to bound latency or cost; a
truncated reply fails with a clear message.

Free OpenRouter variants (`:free`) are shared, low-priority test endpoints
(50 requests/day, 20/min). For faster AoC runs prefer a small text/code model,
or append `:nitro` to a model slug to route by throughput (paid endpoints).

The error includes the provider's response body, which usually says exactly what
is wrong:

- `401 {"error":{"message":"User not found."}}` from OpenRouter means the key is
  invalid or revoked — create a new one at openrouter.ai/keys.
- `404 Not Found` usually means the base URL is missing `/v1`, or the model id
  does not exist.
- `401 Incorrect API key` from OpenAI means `HARNESS_API_KEY` is not an OpenAI
  key; set `HARNESS_BASE_URL` to the key's provider.

### Logging

Logs go to stderr, so stdout stays clean for answers and JSON:

```sh
cargo run -p harness -- --log-level debug solve 2024/1 --no-submit
RUST_LOG=aoc_archive=trace,harness=debug cargo run -p harness -- scrape
```

`--log-level`, `HARNESS_LOG`, then `RUST_LOG` are checked in that order;
the default is `warn`. Debug logs include HTTP URLs/statuses, cache hits,
model configuration (never the key), token/message sizes, compile and execution
timings, and each agent turn.

Every model turn is also recorded structurally. At `--log-level info`:

```
INFO model turn turn=1 ttft_ms=12400 duration_ms=48900 reply_bytes=2410
```

In `--json` output it appears in `history`:

```json
{"turn": 1, "ttft_ms": 12400, "duration_ms": 18900,
 "reply_bytes": 2410, "tool_calls": 1, "tool_errors": 0,
 "tests_passed": 0, "tests_total": 1, "total_tokens": 830}
```

The human report prints the same table after a solve.

Replay JSON is simply an array of text reply strings. Replay is a text-only
model, so it cannot issue tool calls; use it for offline smoke tests.

```json
[
  "```rust\nfn main() { println!(\"0\"); }\n```",
  "```rust\nfn main() { println!(\"11\"); }\n```"
]
```

### Solution contract

A solution lives in `solutions/<year>/<day>/` and is one of:

- `solve` — an executable;
- `solve.py` — run with `python3`;
- `solve.rs` — compiled with `rustc -O` into `target/aoc/` when it changes.

It receives the puzzle input on **stdin**, writes the answer on **stdout**, and
reads the part number (`1` or `2`) from the `AOC_PART` environment variable.
`AOC_YEAR` and `AOC_DAY` are set too.

A different command can be used for any language with the manual commands:

```sh
cargo run -p harness -- run 2024/1 --examples --cmd 'python3 mysolver.py'
```

The agent loop itself writes Rust `solve.rs` through its tools.

## Example tests

Code blocks introduced as examples are captured, and the following emphasized
answer is recorded when one can be found:

```json
{"input": "3   4\n4   3\n...", "expected": "11"}
```

Extraction is heuristic: only examples with a detected answer are used as
tests, and many days have none detectable (especially earlier years where
examples live in prose). The full statement always remains available.

## Archive layout

```
.aoc/
├── index.json                # all years, coverage summary
├── 2024/
│   ├── index.json            # calendar days, titles, parts, tests, inputs
│   ├── 01/
│   │   ├── puzzle.json       # structured: parts, markdown, examples, answers
│   │   ├── puzzle.md         # statement as Markdown
│   │   ├── answers.json      # answers recorded by the harness (parts -> answer)
│   │   ├── submissions.jsonl # append-only Submit log (accepted, rejected, …)
│   │   ├── remote.json       # live stars/parts/answers snapshot
│   │   └── input.txt         # only with AOC_SESSION or --input
│   └── ...
└── raw/                      # cached HTTP responses (resumable scrapes)
```

`puzzle.md` is generated from `puzzle.json`; `puzzle.json` keeps the raw HTML of
each part so parsing can improve without refetching (`scrape --reparse`).

## MVP scope

Working end to end:

- scrape/list/show/input/init/run/solve/next — eight verbs, with Day and Event Selections, `--reparse`, and `--wait`
- manual solve loop: `init` → `run --examples` → `run`
- `--part 1|2|all` as a filter over one Day, with Unreleased/Sealed/Missing errors
- model loop: tools, streaming, timeout, heartbeats, events, answer recording
- opt-in submission with `--submit`, cached Star state, `list --stars`, and two-phase Part One → Part Two in one session
- a local archive of 262 days and 100 detected example tests (not committed)

Deliberately not in scope:

- session persistence, branching, or context compaction
- non-Rust agent solutions (manual `solve.py` / executable still works)
- reliable example extraction for prose-heavy early years

## Next stage

The loop follows Pi's core shape: small tool loop, event stream, mechanism over
policy. The next useful pieces, in rough priority order:

- **Sessions on disk** (JSONL transcript per day, resumable runs, branchable).
- **Context management** — compaction once tool results fill the window.
- **Steering/follow-ups** — inject a message while a turn is running.
- **Retries and timeouts per tool**, plus per-tool approval policies.
- **Broader example-test extraction**, since prose-only examples (common in
  2015–2017) leave some days unverified.