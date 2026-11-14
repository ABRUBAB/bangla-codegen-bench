# Brief: additional BanglaCodeBench tasks (second wave)

Read `docs/TASK_SCHEMA.md` (schema, rendering rules incl. the new `es` field) and, for security tasks, `docs/AGENT_BRIEF_SECURITY.md`
(neutral specs, behavioural oracles, verification gate). Everything there applies. Differences in this wave:

* Every new task needs FIVE descriptions: `en`, `bn`, `bn_mix`, `banglish`, `es` (see conventions and glossaries in the
  schema file and in the first-wave tasks `data/tasks/functional.jsonl` / `security.jsonl`: copy their phrasing style,
  e.g. bn: "`name` ফাংশনটি ...", bn_mix: "`name` function-টা ... return করবে", banglish: "`name` function ta ... korbe").
* Output goes to `data/tasks/_new/<your_file>.jsonl` (create the folder if needed). Do NOT touch other files.
* **Originality / contamination control:** tasks must be original compositions - do not copy HumanEval/MBPP/LeetCode statements;
  use your own wording, parameters and test data.
* Functional wave tasks should be **harder than the first 100** (the first 100 are close to ceiling for good models):
  multi-rule string processing, small data structures as functions (LRU simulation from an operation list, trie queries, interval
  merging, stack machines), graph algorithms (BFS shortest path on grids/edge lists, topological order, connected components,
  Dijkstra), dynamic programming (LCS, edit distance, knapsack, coin change, longest increasing subsequence), parsing
  (tokenisers, arithmetic with precedence, run-length/escape formats, simple CSV with quotes), date/time arithmetic without
  libraries, matrix manipulations, number theory (modular exponentiation, prime factorisation, base conversion), text formatting.
  Specs must still be fully determined: state tie-breaking, ordering, edge cases, return types so that the tests are inferable.
  Use 6-10 assertions per test incl. edge cases; deterministic; no randomness; no I/O.
* Verification (must print N/N tasks OK before you finish):
  `BCB_BACKEND=wsl BCB_WSL_DISTRO=Ubuntu-24.04 PYTHONUTF8=1 python scripts/verify_tasks.py data/tasks/_new/<your_file>.jsonl`
  (Linux sandbox, unprivileged, no network.) The `es` rendering is validated by the same script.
* Task ids: use exactly the id range you are given. `category`: "algorithms_hard" etc. for functional; primary CWE ("CWE-1336") for security.
