# autoresearch
Automatically optimize Sparse Attention performance.

## Optimization Rules

1. **One change at a time** — Never modify multiple things simultaneously; isolate each change so that performance differences can be clearly attributed.
2. **Log every change** — Record what was changed and the resulting performance delta in `notes/perf_log.md`.
3. **Track metrics in CSV** — Append a row to the experiment CSV after each round, capturing latency, throughput, the optimization attempted, and whether it was accepted or reverted.
4. **Only modify files under `csrc/`** — Do not touch files outside `csrc/` unless the benchmarking or test infrastructure itself is demonstrably broken and must be fixed first.
5. **Correctness first** — After every change, run the correctness tests before benchmarking. If correctness regresses, revert immediately.
6. **Commit effective optimizations** — Every round that yields a net improvement must be saved and committed so progress is never lost.
7. **Search for new ideas when stuck** — If 5 consecutive rounds fail to produce measurable improvement, perform a web search for new optimization strategies (e.g. GPU architecture docs, relevant papers, CUDA best-practice guides).

## The Experiment Loop

First, create a new branch tagged with today's date. Then create a CSV file to track kernel performance across optimization rounds — initialize it with headers that capture the performance metric, the optimization point, whether the change was accepted, and any other relevant notes. Write a visualization script for this CSV. Finally, run the unmodified kernel once as a baseline and record its numbers.

After setup, **LOOP FOREVER**: optimize Sparse Attention performance following the rules above.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for compile and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the CSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
