# DarkBGHelmet — Backgammon Engine Plan

Status: **draft for review** (2026-10-02). Nothing is implemented yet.

---

## 1. What the research says

| Topic | Finding | Consequence for us |
|---|---|---|
| Evaluator | All top bots (gnubg, XG, BGBlitz, wildbg) use a small feed‑forward net with ~200 Tesauro‑style inputs. Outputs are 5 cubeless probabilities: win, win‑gammon, win‑backgammon, lose‑gammon, lose‑backgammon. gnubg uses separate nets for contact, crashed and race positions, plus bearoff databases. | We keep the 5‑probability output (it is score‑independent, so one net covers money and every match score). We use phase‑specific output heads on a shared body. |
| NNUE in backgammon | Whittington's Backgammon‑NN (2025) uses a Stockfish‑style net: 198→256→128, squared‑ReLU, heads chosen by position class. It reaches about 49% against gnubg 2‑ply. Its main lesson: **label quality is the ceiling**. Self‑play TD plateaued. Distilling from stronger labels (gnubg 2‑ply) broke through. | Sparse, incrementally updated inputs suit backgammon well (see §3). Training must use labels from search or rollouts, not just TD. |
| MCTS vs. alpha‑beta | MCTS/PUCT is a poor fit. Each ply has 21 dice outcomes, which dilutes simulations, and the evaluators are already strong. Every strong bot uses **expectiminimax over chance nodes** with move filters, plus truncated rollouts. Ballard's Star1/Star2 (alpha‑beta at chance nodes, Hauk/Buro/Schaeffer 2004) clearly beats plain expectimax when move ordering and probing are good. | We use selective expectimax with Star2‑style bounds and sampled rolls deep in the tree (§5). No MCTS. |
| Depth / speed | XG 4‑ply error rate 0.686 vs. gnubg 3‑ply 1.224. "XG Roller" is a 6‑ply truncated rollout. XG is 3–15× faster than gnubg. Cost grows about 21× per ply. | 6‑ply is only feasible with pruning, transpositions, roll sampling with variance reduction, and many cores. |
| Variance reduction | Dahl's luck adjustment: 100 rollout games with VR ≈ 5,000 without. | Used in rollouts and in sampled chance nodes during search. |
| Cube | Standard is Janowski: `E = E_dead·(1−x) + E_live·x` with cube efficiency x ≈ 0.6–0.8, applied at leaves. For match play, gnubg uses a recursive live‑cube take‑point model over match equity tables (MET). Rockwell‑Kazaross / Kazaross‑XG2 are the standard METs. | Analytic cube layer on top of the cubeless net, plus one new idea: a learned per‑position cube efficiency x (§4). |
| Position formats | gnubg Position ID (14 chars) + Match ID (12 chars); XGID (`XGID=-b----E-C---eE---c-e----B-:0:0:1:52:0:0:0:0:10`). | Accept and emit both. |
| Protocol | **There is no UCI equivalent for backgammon.** gnubg has an ad‑hoc socket "external player" interface with FIBS board strings. wildbg offers an HTTP JSON API (`GET /move?die1=..&p24=..`). | We define a REST/JSON API (§7) and a small line‑based text protocol for CLI and testing. |

---

## 2. Language, toolchain, layout

* **C++20**, CMake, clang. Target: Apple M5 Max (arm64, 18 cores, 48 GB). NEON SIMD for inference, with a portable scalar/AVX2 fallback.
* Header‑only dependencies: `cpp-httplib` (HTTP server), `nlohmann/json`. Tests: `doctest`.
* **Training in C++ too.** The nets are small and the inputs sparse, so a CPU trainer with Adam is fast enough and avoids a PyTorch dependency (torch is not installed here). PyTorch/MPS stays optional.
* gnubg is installed locally (`/opt/local/bin/gnubg`). We use it as an oracle: to cross‑check move generation, as a benchmark opponent, and optionally as a teacher (see question Q1).

```
engine/
  core/      board (packed rep), movegen, match state, PositionID/MatchID/XGID
  nn/        NNUE inference (int16 accumulators, int8 dense), weight files
  eval/      bearoff DBs, MET, cube math (money + match), static evaluator
  search/    expectimax/Star2, move filters, transposition table, thread pool, rollouts
  server/    HTTP JSON API
  cli/       text protocol, perft, benchmarks
training/    self-play generator, labeler (search/rollout), trainer, quantizer
tests/
```

---

## 3. Board and match representation (backgammon "bitboards")

**Packed board: 256 bits = one position.** Each side stores 25 slots (24 points + bar) at 4 bits per slot (0–15 checkers), always seen from that side's own perspective. That is 100 bits per side, held in a 128‑bit lane. Checkers borne off are implicit (15 − sum).

```
struct Board { u128 side[2]; }   // side[0] = player on roll
```

* **Moving a checker** from a to b is one add and one subtract on that side's lane: `side[0] += (1<<4b) − (1<<4a)`. No loops, no branches.
* **Hit test:** `opp nibble at (25 − b)`. A hit is two more nibble operations.
* **Changing the side to move** just swaps the two lanes, because both lanes are stored from their own perspective.
* **Occupancy masks via SWAR:** 25‑bit masks for "≥1 checker" (`made1`) and "≥2 checkers" (`made2`, i.e. blocked points), computed with nibble tricks in a few instructions. The opponent's blocks in our coordinates are `bitreverse24(opp.made2)`. Legal landing squares for die d are `(made1 >> d) & ~oppBlocked`. Bearoff legality is `(made1 & ~homeMask) == 0` plus a highest‑checker test (count leading zeros).
* **Hash:** a 64‑bit hash of the 256 bits (one multiply‑xorshift per lane). It is used to drop duplicate results from move generation (doubles produce many) and as the transposition table key. The exact 80‑bit gnubg key can also be produced for I/O.
* **Move generation:** iterative, with dice ordering handled. Enforces the rules that you must use the maximum number of dice and must play the larger die if only one can be played. Each child carries a compact move record (up to 4 from/to pairs) and the list of NNUE feature toggles it causes (see §4), so the evaluator never has to diff boards.

**Match state, packed into 32 bits:** match length (0 = money, 1–25), away‑scores for both players, Crawford/post‑Crawford, cube log₂ (0–6 → 1–64), cube owner (center/me/opp), plus Jacoby and beaver flags for money play.

---

## 4. Position analysis: network and cube layer

### 4.1 Network ("BG‑NNUE")

**Input features: sparse, binary, thermometer‑coded.** For each perspective (player on roll and opponent) and each of 26 slots (24 points, bar, off), there are 15 features "at least k checkers", k = 1..15. That gives 2 × 26 × 15 = 780 features per perspective.

* With this coding, **one checker step toggles exactly 2 features**, and a hit toggles 2 more. A whole move toggles at most 8 features (16 for doubles), compared with about 60 active features in a full refresh.
* There are **two accumulators** (one per perspective), as in Stockfish's white/black accumulators. Changing the side to move every ply is then free: we just swap which accumulator comes first.
* Layer 1 is 780 → 512 per perspective, in int16. It is updated incrementally from the parent: all children of a node share the parent's accumulator, and each child only applies its own toggles.
* **Dense head:** [acc_me ‖ acc_opp] (1024) plus about 16 hand‑crafted dense features computed only at leaves (pip counts, escapes and containment as in gnubg, back‑game timing, contact flag). Then → 32 → 32 → outputs, int8 weights, clipped/squared ReLU.
* **Output heads chosen by phase:** contact, crashed, race, bear‑in. Each head outputs 5 logits, decoded into consistent probabilities (P(win) ≥ P(win‑gammon) ≥ P(win‑backgammon)).
* **Race endgames** with ≤15 checkers inside 6 points use the one‑sided bearoff DB (54,264 positions per side, exact distributions, including gammon chances). Positions with ≤7 checkers per side inside 6 points use an exact two‑sided cubeful DB. Both are generated by us at build time.

Estimated cost on the M5 Max: about 0.15–0.3 µs per incremental evaluation, so 3–5 M evals/s per core and about 50 M/s on 16 cores.

### 4.2 Cube and match equity

* **MET:** Kazaross‑XG2 for up to 25‑point matches, stored as data. We also add a generator that computes a MET from gammon rates, so we are not tied to a single table. Post‑Crawford and free‑drop handling are included.
* **Cubeful equity** uses gnubg's live‑cube model: money uses Janowski interpolation between take point and cash point. Match play uses the recursive live‑cube take points `TP_live(n) = TP_dead(n)·(1 − TP_live(2n))` over the MET, working in match‑winning chances (MWC).
* **New idea: learned cube efficiency.** Instead of a constant x ≈ 0.68, a tiny extra head predicts x per position. Volatile blitz positions get a high x; races get a low x, roughly matching the known race formulas. It is trained on cubeful self‑play and rollout results. Because x enters the formula linearly, the cube math stays analytic and consistent; the net only refines it.

### 4.3 What one analysis returns

```
{ probs: [win, wg, wbg, lg, lbg],
  cubeless_equity, cubeless_mwc,
  cubeful: { no_double, double_take, double_pass }   // as equity and as MWC
  decision: "No double" | "Double, take" | "Double, pass" | "Too good" | (re)double...,
  take_decision: "Take" | "Pass" | "Beaver",
  errors: { wrong_double, wrong_take, ... } }
```

In match play, "match equity" is MWC and "game equity" is the normalized money‑equivalent equity (EMG), as XG and gnubg report.

---

## 5. Search: Selective Expectimax with Sparse Chance Sampling (SESCS)

**Ply convention.** We count like XG: 1‑ply = static evaluation. XG 1‑ply = gnubg 0‑ply. "6‑ply" therefore means 5 dice layers below the move being judged. The depth limit is configurable up to 7, so gnubg‑style 6‑ply is also reachable, at about 21× the cost.

**Why not plain expectimax:** a full 6‑ply search costs about 21⁵ × 20 ≈ 80 M evaluations *per candidate*. Possible in principle, but too slow for analysis. These measures bring 6‑ply into the 5–20 s range per decision on the M5 Max:

1. **Root iterative deepening with successive filtering.** All moves at 1‑ply → best 16 (within 0.16 equity) at 2‑ply → 8 at 3‑ply → 5 at 4‑ply → 3 at 5‑ply → 2 at 6‑ply. Thresholds are tuned like gnubg/XG move filters.
2. **Inner nodes pick the move cheaply,** as gnubg does. For each roll, the move is chosen by 1‑ply (or 2‑ply near the root), and only that move is expanded. This keeps the branching at about 21 per ply instead of 21 × 20.
3. **Star2 bounds at chance nodes.** Probabilities are bounded ([0,1], equity in [−3, 3]), so once a candidate cannot beat the current best by more than ε, we cut it off. Rolls are probed in "most informative first" order: doubles and big hitting numbers first.
4. **Sparse sampling at deep chance layers.** At layers ≥4 we do not expand all 21 rolls, but a stratified, antithetic subset (e.g. 21 → 9 → 6), weighted, with Dahl luck correction against the 1‑ply evaluation. Effectively the search "turns into" a variance‑reduced truncated rollout at depth, like XG Roller, but inside a deterministic search.
5. **Transposition table:** a lock‑free, 16‑byte‑entry table with key (board hash ⊕ depth ⊕ cube state) storing the 5 probabilities in fixed‑point. Different dice orders very often lead to the same positions at depth.
6. **Batch evaluation and shared accumulators:** all children of a node are evaluated together, using the incremental feature toggles from move generation.
7. **Parallelism:** root candidates and top‑level rolls are spread over a work‑stealing pool (16 workers). The TT is shared.
8. **Cubeful recursion:** every node returns both the cubeless probability vector and the cubeful equities for the relevant cube states (center / owned / opp‑owned) *in one pass*. Cube decisions at each inner node are made with the parent's cube state, and Janowski with the learned x is applied at leaves.
9. **Rollouts** (cubeful, VR, truncated or to the end, using the bearoff DBs) are offered as an option above 6‑ply, and are used for training labels.

Rough targets on the M5 Max: 1‑ply all moves < 1 ms · 3‑ply ≈ 10–50 ms · 4‑ply < 0.5 s · **6‑ply ≈ 5–20 s** per checker decision. Cube decisions cost about the same as evaluating one position at that depth.

---

## 6. Training pipeline (C++)

1. **Bootstrap:** TD(λ) self‑play from scratch, about 1–2 M games. This gives a solid intermediate net within hours on 16 cores. Benchmarks: Tesauro's public `pubeval`, then gnubg 0‑ply.
2. **Expert iteration (breaks the self‑label ceiling):** sample positions from self‑play. Label them with our own *deeper* search (3‑ply) or short VR rollouts. Retrain, repeat. Labels from search are stronger than the net, so the net keeps improving.
3. **Optional gnubg distillation** (see Q1): label about 20 M positions with gnubg 2‑ply as a fast jump start. This is the route that took Backgammon‑NN to gnubg strength.
4. **Cube‑efficiency head:** trained afterwards on cubeful self‑play / rollout outcomes.
5. Quantize to int16/int8, write a versioned `.bgnn` weight file, and verify that the float and quantized nets agree.

Being honest: the engine infrastructure (rules, search, cube, API, training code) can be built in this project. **Reaching XG/gnubg‑level strength depends mostly on CPU hours of training,** probably days of compute in total, run in stages.

---

## 7. Interface

There is no standard, so we provide two front ends over the same core:

### 7.1 HTTP JSON API (`/v1`)

| Method & path | Purpose |
|---|---|
| `GET /v1/info` | Engine name, version, net id, supported options |
| `POST /v1/evaluate` | Position (+ match) → probabilities, equities, cube analysis |
| `POST /v1/move` | Position + dice → ranked candidate list (move, probs, equity, error vs. best, ply used) |
| `POST /v1/cube` | Cube action: double / no double / too good, and take / pass, with equities |
| `POST /v1/analyze` | Combined cube + move analysis, as for a game record |
| `POST /v1/rollout` | Start a rollout job → `job_id`; `GET /v1/jobs/{id}` polls for progress or streams it via SSE |

Example request:

```json
{ "position": { "xgid": "XGID=-b----E-C---eE---c-e----B-:0:0:1:52:0:0:0:0:10" },
  // or { "gnubg": "4HPwATDgc/ABMA:cAkAAAAAAAAA" } or { "board": [..26 ints..], ... }
  "match":  { "length": 7, "score": [2, 4], "crawford": false, "cube": 2, "owner": "opponent" },
  "dice":   [5, 2],
  "options":{ "plies": 6, "time_ms": 20000, "candidates": 5, "cubeful": true } }
```

### 7.2 Text protocol "BGEI" (for CLI and testing, UCI‑like)

```
bgei → id name DarkBGHelmet / readyok
position xgid <...> | position gnubg <posid>:<matchid>
go move dice 52 plies 4      → info ply 3 best 13/8 8/6 eq 0.123 ... / bestmove 13/8 8/6
go cube plies 4              → bestcube double take
setoption threads 16
```

---

## 8. Milestones

| # | Deliverable | Check |
|---|---|---|
| M1 | Core: packed board, movegen, match state, PositionID/MatchID/XGID | Move counts and positions match gnubg over a large random position set |
| M2 | NNUE inference (NEON), weight format, bearoff DBs, pubeval baseline | Incremental eval = full refresh, bit for bit; bearoff checked against gnubg |
| M3 | Self‑play TD trainer → first net | Beats pubeval clearly; error rate measured with gnubg |
| M4 | MET + cube layer (money & match 1–25) | Cube decisions agree with gnubg on reference positions |
| M5 | Search: expectimax, filters, TT, threads, Star2, sparse sampling to 6‑ply; rollouts | Speed targets in §5; deeper = better on a benchmark set |
| M6 | HTTP server + BGEI protocol | API tests, OpenAPI spec |
| M7 | Strength loop: expert iteration (+ optional gnubg distillation), cube‑efficiency head | Games against gnubg 2‑ply |

---

## 9. Open questions for you

* **Q1: Training labels.** Should we use the installed gnubg as a teacher to bootstrap (fastest path to strength; gnubg is GPL, but labels it generates are just data), or stay pure self‑play + expert iteration (fully independent, slower)?
* **Q2: Ply convention.** Should "6‑ply" mean XG counting (my proposal) or gnubg counting (one dice layer deeper, ~21× the cost)?
* **Q3: Language.** C++20 OK, or do you prefer C or Java?
* **Q4: Git.** The folder is not a git repository (and it lives in OneDrive). Should I `git init` here, or put the code somewhere outside OneDrive? Build artefacts and multi‑GB training data don't sync well.

---

## Sources

* [Backgammon‑NN development report (Whittington)](https://whittingtonchess.com/backgammon-report), [repo](https://github.com/Chris-Whittington-Chess/Backgammon-NN)
* [wildbg](https://github.com/carsten-wenderdel/wildbg), [engine docs](https://github.com/carsten-wenderdel/wildbg/blob/main/docs/dev/engine.md), [announcement / API discussion](https://lists.nongnu.org/archive/html/bug-gnubg/2023-11/msg00002.html), [Backgammon API thread](https://lists.nongnu.org/archive/html/bug-gnubg/2023-11/msg00001.html)
* [All About GNU Backgammon](https://bkgm.com/gnu/AllAboutGNU.html), [gnubg move filters](https://www.gnu.org/software/gnubg/manual/html_node/Introduction-to-move-filters.html), [pruning nets](https://www.gnu.org/software/gnubg/manual/html_node/Pruning-neural-networks.html), [live cube equities](https://www.gnu.org/software/gnubg/manual/html_node/Live-cube-equities.html)
* [Position ID](https://www.gnu.org/software/gnubg/manual/html_node/A-technical-description-of-the-Position-ID.html), [Match ID](https://www.gnu.org/software/gnubg/manual/html_node/A-technical-description-of-the-Match-ID.html), [bearoff databases](http://gnu.ist.utl.pt/manual/gnubg/html_node/Introduction-to-bearoff-databases.html)
* [Tesauro, Programming backgammon using self‑teaching neural nets](https://bkgm.com/articles/tesauro/ProgrammingBackgammon.pdf)
* [Hauk/Buro/Schaeffer, *‑Minimax Performance in Backgammon](https://link.springer.com/chapter/10.1007/11674399_4), [Rediscovering *‑Minimax Search](https://link.springer.com/chapter/10.1007/11674399_3), [ChanceProbCut](https://dke.maastrichtuniversity.nl/m.winands/documents/CIG2009.pdf)
* [Janowski formulas (Madsen)](https://lassehjorthmadsen.github.io/bganalyses/analyses/janowski.html), [Cube handling under a jump model](https://arxiv.org/pdf/1203.5692)
* [Depreli bot comparison](https://www.bkgm.com/articles/Keith/DepreliBotComparison/index.html), [XG major features](https://www.extremegammon.com/Majorfeatures.aspx)
* [PureTD (2026)](https://arxiv.org/pdf/2608.15146), [alexstrehl/backgammon-ai-engine](https://github.com/alexstrehl/backgammon-ai-engine)
