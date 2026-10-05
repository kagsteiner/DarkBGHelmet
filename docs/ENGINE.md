# How DarkBGHelmet works

This document describes the engine as of v0.4 (network `big1024_cube`). It covers:

* how positions are represented
* how the neural network evaluates them
* how the cube and match play are handled
* how the search, the rollouts and the training pipeline work

File and function names are given so you can follow along in the code.

Conventions used throughout:

* **Perspective.** Every evaluation is from the view of the **player on roll**.
* **Plies** follow the **XG convention**: 1-ply is the static evaluation. XG *n*-ply is
  gnubg (*n*−1)-ply.
* **Equities** are cubeless or cubeful **money equities** (points per game, normalised to the
  cube value). In match play the engine works internally in **match winning chances (MWC)**
  and reports the money-equivalent equity (EMG) as well.

---

## 1. Overview

```
            ┌─────────────────────────────── search (engine/search) ──────────────────────────────┐
 position → │ root: generate plays → filter by depth → expectimax over 21 rolls per layer          │ → ranked plays,
 + match    │       inner nodes choose the opponent's reply statically, cube states carried along  │   cube decision
            └──────────────┬──────────────────────────────────────────────────────────────────────┘
                           │ leaves
            ┌──────────────▼─────────────── evaluation (engine/eval, engine/nn) ───────────────────┐
            │ finished game → exact result   │  both sides in home board → bearoff database         │
            │ otherwise → neural network: 5 outcome probabilities + cube efficiency (cube head)     │
            └──────────────┬──────────────────────────────────────────────────────────────────────┘
                           │
            ┌──────────────▼─────────────── cube & match (engine/eval/cube.cpp) ───────────────────┐
            │ outcome probabilities + match equity table → cubeful values for each cube state      │
            └──────────────────────────────────────────────────────────────────────────────────────┘
```

The design deliberately separates two concerns:

1. **The neural network estimates what is likely to happen.** It outputs win, gammon and
   backgammon chances, which are almost independent of score and cube.
2. **The match and cube logic turns that into decisions** at the actual score and cube state,
   using a match equity table and an analytic cube model, refined by a learned cube efficiency.

One network therefore serves money play and every match length from 1 to 25 points.

---

## 2. Board representation (`engine/core`)

### Packed board ("bitboard")

`Board` (`board.hpp`) stores the position in **256 bits**, two 128-bit lanes, one per player:

* each lane holds 25 *nibbles* of 4 bits (0–15 checkers): nibble *s* = 0…23 is the player's own
  point *s*+1, and nibble 24 is the bar;
* checkers borne off are implicit (15 − total);
* `side[0]` is always the player on roll, `side[1]` the opponent;
* each lane is stored **from its owner's perspective**, so the opponent's point that faces our
  point *s* is its slot 23−*s*.

This gives some cheap operations:

* moving a checker is one subtraction and one addition on a 128-bit integer;
* passing the turn is swapping the two lanes;
* "all checkers home" is a single shift-and-test; the rearmost checker is a count-leading-zeros;
* the 64-bit hash for the transposition table and for duplicate detection mixes the four 64-bit
  words.

### Move generation (`movegen.cpp`)

A *move* is the complete play for one roll (up to four checker steps). The generator returns
**every distinct resulting position** and enforces the rules:

* as many dice as possible must be played;
* if only one die can be played, it must be the larger one if possible;
* checkers on the bar must enter first;
* bearing off is only allowed with all checkers home, with the usual rule for the highest
  checker.

Doubles are generated with non-increasing source points, which reaches every distinct play
exactly once. Non-doubles try both die orders, and duplicates are removed with a small hash
table. Correctness was verified against gnubg on 5,300 random positions, with identical move
lists. Speed is about 40 M positions/s in a perft test on one core.

### Position formats (`ids.cpp`)

The engine reads and writes the gnubg **Position ID / Match ID** and the eXtreme Gammon **XGID**.
Moves are written in gnubg notation (`13/7 8/7`, `bar/22*`, `6/off(2)`).

---

## 3. Static evaluation (`engine/eval`, `engine/nn`)

`Evaluator` (`evaluator.cpp`) evaluates a position with the player on roll as follows:

1. **Finished game:** exact result (single, gammon, backgammon).
2. **Both sides have all checkers in their home boards and there is no contact:** the
   **one-sided bearoff database** (`bearoff.cpp`).
3. **Otherwise:** the **neural network**.

### 3.1 Bearoff database

For each of the 54,264 ways to place up to 15 checkers on the six home points, the database stores:

* the exact probability distribution of the number of rolls needed to bear off, playing to
  minimise the expected number of rolls;
* for positions with all 15 checkers still on the board, the distribution of rolls needed to
  bear off the *first* checker, which is needed for gammons.

It is computed by dynamic programming on the first start (under a second) and cached in
`data/bearoff_os15.bin`. Win and gammon probabilities for two sides follow from the two
independent distributions: I win if I finish on my *n*-th roll while the opponent needs at least
*n* rolls. This matches gnubg's bearoff evaluation to an equity RMSE of 0.0005.

### 3.2 Network inputs

**Sparse inputs ("thermometer" coding).** For each perspective there are two blocks, own checkers
and opponent checkers. Each block has 26 slots (24 points, bar, borne off) × 15 features of the
form "at least *k* checkers here". That gives 2 × 26 × 15 = **780 binary features per
perspective**.

With this coding, moving one checker one step switches exactly two features in each
perspective; a hit switches two more. That is what makes incremental updates cheap (§3.4).

**Dense hand-crafted inputs** (`features.cpp`). These are 8 values computed at the leaf, things a
small network derives poorly by itself:

| # | Feature |
|---|---|
| 0, 1 | pip count of player on roll / of opponent (÷100) |
| 2 | pip difference (÷50) |
| 3, 4 | probability that the player on roll / the opponent can hit a blot with the next roll (exact for 99.5% of positions, computed with bit masks over the 21 rolls) |
| 5, 6 | probability that each side's rearmost checker escapes past the opponent's points within 12 pips |
| 7 | contact flag |

**Position classes.** Contact, crashed (one side nearly out of play, gnubg's definition) and race
each have their own output head on the shared body.

### 3.3 Architecture ("BG-NNUE")

```
sparse features (780, own perspective)      ─► W1 ─► accumulator A_me  (1024)  ┐
sparse features (780, opponent perspective) ─► W1 ─► accumulator A_opp (1024)  ├─ clamp to [0,1]
dense features (8)  ───────────────────────────────────────────────────────────┘
        │  2056 inputs
        ▼
    dense 2056 → 128, clamp [0,1]
        ▼
    dense  128 →  64, clamp [0,1]  ── h3 ──────────────┐
        ▼                                              ▼
    output head (contact | crashed | race)        cube head (§4.3)
    64 → 5, sigmoid
        ▼
    P(win), P(win gammon), P(win backgammon), P(lose gammon), P(lose backgammon)
```

* The same weight matrix W1 serves both perspectives, as in Stockfish's NNUE. The input to the
  dense part is the on-roll accumulator followed by the opponent's.
* The five outputs are unconditional probabilities in the gnubg/XG convention (P(win gammon)
  includes backgammons). They are made consistent afterwards: P(win) ≥ P(win gammon) ≥
  P(win backgammon), and the same for losses.
* Network size: about 0.8 M weights in W1, 0.26 M in the dense layers; the file is 4.3 MB. The
  format (`network.cpp`, magic `BGNN`, version 2) stores the dimensions, so other sizes
  (e.g. 512 → 64 → 64) load as well.

### 3.4 Incremental updates and inference speed

When the search evaluates all plays from a position, every child shares the parent's two
accumulators:

* `acc_update` compares parent and child lane by lane (an XOR shows the changed points);
* it collects the weight rows to add or subtract, at most a few per play;
* it writes the child accumulators in one fused pass.

Further optimisations in `net_forward`:

* About 70% of the clamped accumulator entries are zero. A NEON-vectorised scan collects the
  indices of the positive entries, and the first dense layer only processes those, with weights
  stored input-major (transposed) so the inner loop over the 128 outputs vectorises completely.
* The second dense layer also skips zero inputs.
* Everything runs in float32. With this sparsity, an int8 version would hardly be faster.

On one Apple M5 core:

| Network | Time per evaluation (incl. move generation) |
|---|---|
| 1024/128/64 | about 1.8 µs |
| 512/64/64 | about 0.95 µs |

---

## 4. Cube and match play (`engine/eval/cube.cpp`)

### 4.1 Value space and match equity

All cube and match calculations happen in one "value space" from the player on roll's view:

* **match winning chances** in match play;
* **money equity normalised to the cube** for money games.

`CubeModel::value(pts)` gives the value if the game ended now with the player on roll winning
*pts* points (negative = losing). In matches it looks up the **Kazaross-XG2 match equity table**
(25 × 25 plus post-Crawford row, embedded in `met_kazaross_xg2.inc`). It takes care of:

* the Crawford game;
* post-Crawford scores (including the free drop);
* scores beyond the match length.

**Every evaluation in the search is converted with these values.** The probability of each outcome
is multiplied by the value of that outcome at the current score. That is how the engine automatically:

* ignores gammon risk when the opponent is 1-away (losing a gammon costs nothing extra);
* goes for gammons when they are worth more;
* switches to "maximise wins" at double match point.

### 4.2 Cubeful equity: generalised Janowski model

The value of a position with a live cube is computed as

```
cubeful = x · live + (1 − x) · dead
```

* **dead** is the cubeless value at the current cube.
* **live** is the value if the cube could be used perfectly. It is a piecewise-linear function
  of P(win) through four points:
  * (0, average value of a loss)
  * (my take point, value of losing the current cube)
  * (opponent's take point, value of winning the current cube)
  * (1, average value of a win)

  If a player cannot double, for example because the opponent owns the cube, the cube is dead
  for them, or it is the Crawford game, that player's kink is left out.
* **x** is the **cube efficiency**: how much of the live cube's value is actually realised.

The take points are computed **recursively over higher cube levels**, so redoubles are accounted
for. My take point when doubled from *n* to 2*n* depends on the point where I would redouble
to 4*n*, which depends on the opponent's take point at 4*n*, and so on, until the cube is dead
(in matches) or reaches a maximum level (money). For money without gammons this reproduces the
classic 20% take point; with gammons it reproduces Janowski's formulas exactly.

Special cases handled:

* the Jacoby rule (gammons don't count while the cube is centred in money play);
* dead cubes in matches (no doubling when a player needs fewer points than the cube value);
* the Crawford rule.

Validation: fed with identical probabilities, the model agrees with gnubg 1.08's 0-ply cube
decisions on about 99% of match positions and 96.5% of money positions. That required
x = 0.76, which gnubg 1.08 effectively uses.

### 4.3 The cube head: learned cube efficiency

A single constant x ignores volatility. In a blitz the cube is much "more live" than in a quiet
holding game or a race. The **cube head** learns x per position:

* **Input:** the network's last hidden layer h3 (64 values) plus the five outcome probabilities.
* **Layers:** 32 hidden units (clamped ReLU), then 3 outputs, one per cube state: centred, owned
  by the player on roll, owned by the opponent. Each output is mapped to x = 0.3 + 0.9·sigmoid(z).
* **Training** (`bgtrain traincube`):
  * The network body stays frozen, so the cubeless outputs, and with them the checker play, are
    unchanged.
  * Targets come from **4-ply cubeful search** on 1.2 M positions (`bgtrain cubelabel`).
  * The loss is the cubeful-equity error dead + x·(live − dead) − target, computed with the
    search's own probabilities. Positions where the cube barely matters therefore count little.
* **Effect:**
  * the cubeful equity error against the 4-ply targets halves (0.043 → 0.020);
  * in 200,000-game duels with the same network, head vs. formula is +0.010 ± 0.004 ppg.

The same x per cube state is then used for every match score.

The cube head is a required part of every network used for play and analysis: `bgserver`,
`bgei`, `bgplay` and `bgselfplay` refuse networks without one. A newly trained network gets its
head in a second step (`bgtrain traincube`). The formula with x = 0.76 remains only as the
fallback for bearoff-database positions and in developer tools.

### 4.4 Cube decision

For the player on roll the engine computes three values:

* **ND:** no double, with the current cube state;
* **DT:** double/take, the cube at twice its value, owned by the opponent;
* **DP:** double/pass, the current cube value won.

From these it reads:

* **double** if min(DT, DP) > ND;
* **take** if DT ≤ DP;
* **too good** if ND > DP;
* **beaver** in money play if a take would give the taker a positive equity.

---

## 5. Search (`engine/search/search.cpp`)

### 5.1 Expectimax over dice

Evaluating a position *P* (player to move, before the roll) at depth *d* means:

* **d = 1:** static evaluation.
* **d > 1:** average over the 21 distinct rolls (doubles weighted 1/36, others 2/36). For each
  roll, the mover's best play is chosen by its **static** value, and the resulting position
  (opponent to move) is evaluated at depth *d* − 1.

This "static move choice at inner nodes" keeps the branching factor at 21 per ply instead of
21 × (number of plays); gnubg and XG do the same. The move choice uses the score-aware value at
the node's cube level, so the opponent's replies are also chosen with correct gammon values.

A candidate play at depth *d* is valued by evaluating the position after the play (opponent to
move) at depth *d*. Cost grows by about 21× per ply:

| Depth | Full evaluations per candidate |
|---|---|
| 2-ply | about 500 |
| 3-ply | about 11,000 |
| 4-ply | about 230,000 |

### 5.2 Cube states carried through the tree

The search doesn't evaluate a single value per node but a small **set of cube states** at once
(gnubg's approach), for example:

* "cube centred";
* "opponent owns a 2-cube";
* "I own a 4-cube".

At every node, the player to move takes its **optimal cube action for each state**: no double,
or double/take, or double/pass, whichever the cube logic of §4.4 prefers. The child nodes are
therefore evaluated for the original states *and* their doubled versions.

Leaves use the cubeful model of §4.2 with the cube head's x. The set stays small (at most 16
states), and the cube work per leaf is negligible compared with the network evaluations.

### 5.3 Root: iterative deepening with move filters

`analyze_moves`:

1. Generates all plays and evaluates them at 1-ply.
2. Deepens step by step. Going from depth *d* to *d*+1, it keeps only the plays within a
   threshold of the best:

   | Step | Plays kept | Threshold |
   |---|---|---|
   | 1 → 2 | up to 16 | 0.16 equity |
   | 2 → 3 | 8 | 0.10 |
   | 3 → 4 | 5 | 0.07 |
   | 4 → 5 | 3 | 0.05 |
   | then | 2 | 0.04 |

   An option caps the number of candidates further.
3. Ranks the plays first by depth reached, then by value.

`analyze_cube` evaluates the current position at the requested depth for the states "current
cube" and "doubled, opponent owns" and applies §4.4. By default, cube decisions in `bgserver`,
`bgplay` and `bgselfplay` can search extra plies (option `cube_extra_plies`, 0 = off).

### 5.4 Sparse chance sampling (control variates)

For 5-ply and deeper, expanding all 21 rolls at every chance node is too expensive. Below the
top two chance layers the engine searches only *k* representative rolls deeply (k = 9 at 5-ply,
7 at 6–7 ply):

1. The static value of the best play is known for **all 21 rolls** anyway, because it was needed
   to choose the play.
2. The rolls are sorted by that static value and split into *k* groups of equal probability.
3. One representative per group is searched deeply.
4. The node value is the sum of all static values plus, for each group, the group's probability
   times (deep − static) of its representative.

This is a control-variate estimator: deterministic, and accurate wherever the deep correction is
similar within a group. With k = 7 the values change by less than 0.01 equity compared with full
expansion, at a fraction of the cost.

### 5.5 Other details

* **Transposition table:** replace-always, with striped locks. The key combines the board hash,
  the depth, the side to move, the cube-state set, and a "salt" from the match state and search
  settings.
* **Parallelism:** a thread pool spreads (candidate play × first-layer roll) tasks over the cores.
* **Performance:** on all cores of an M5 Max, a 3-ply checker decision takes a few hundred
  milliseconds or less, and a 6-ply cube decision about 1.5–2 s (512-network). Single-threaded:
  see the timings in `docs/RESULTS.md`.

---

## 6. Rollouts (`engine/search/rollout.cpp`)

Monte Carlo rollouts play each game out with 1-ply checker play and static cube decisions:

* the first two rolls are **stratified** over all 36 × 36 combinations;
* games can optionally be truncated after *n* plies;
* the result is cubeful (game value at the final cube) and cubeless (probabilities).

Optional **variance reduction** subtracts each turn's luck: the static value after the actual roll
minus the average over all 21 rolls. That lowers the variance about 16×, but also costs about 21×
per game with 1-ply play, so it's off by default.

---

## 7. Training

### 7.1 Data

| Stage | Source of positions | Labels |
|---|---|---|
| Bootstrap | gnubg self-play, later also games played by our engine (DAgger-style, to cover positions our own play reaches) | gnubg 2-ply (= XG 3-ply) for the top plays, gnubg 0-ply for the rest (`tools/gnubg_label.py`) |
| Expert iteration | the same position pool (about 55 M unique positions) | **our own 3-ply search** (`bgtrain relabel`) |
| Cube head | 1.2 M positions from the pool | our cubeful 4-ply search (`bgtrain cubelabel`) |

A controlled experiment (identical positions and training) showed that labels from our own
3-ply search are as good a teacher as gnubg 2-ply. Since then the network is trained only on its
own search labels. The current `big1024` body was trained on 38.2 M positions with own 3-ply
labels.

### 7.2 Procedure (`training/train.cpp`)

* **Dataset:** `bgtrain build` de-duplicates positions (a deeper label wins) and drops finished
  and bearoff-database positions.
* **Loss:** binary cross-entropy on each of the five outputs against the label probabilities.
  Each position trains only its class's output head.
* **Optimiser:** Adam, batch 2048, 500 warm-up steps, then cosine decay from 3·10⁻³; 12 epochs.
  It runs multi-threaded on the CPU with per-thread gradients, about 90,000 samples/s for the
  1024 network on 17 cores.
* **Validation:** 1% of the positions are held out and the equity RMSE is reported per class.

### 7.3 What we learned

* **Network capacity mattered.** Going from 512 to 1024 hidden units on the same data gave
  +0.011 ppg in duels.
* **Label quality matters more than quantity.** Once the network matched its teacher, more
  labels of the same quality gave nothing: 23.9 M → 38 M gnubg labels, and 38 M → 47 M own
  labels, were both ties.
* **Validation error and benchmarks graded by another engine are only rough guides.** Decisions
  were made with **duplicate engine-vs-engine duels** (`bgcli duel`) of 20,000–200,000 games.

---

## 8. Interfaces and tools

| Binary | Purpose |
|---|---|
| `bgserver` | HTTP JSON API (`/v1/evaluate`, `/v1/move`, `/v1/cube`, `/v1/analyze`, `/v1/rollout`); see `docs/API.md` and `deploy/DEPLOY.md` |
| `bgei` | Line-based, UCI-like text protocol on stdin/stdout |
| `bgplay` | Play against the engine in a terminal (money or match, cube, manual or automatic dice) |
| `bgselfplay` | Engine self-play with the cube, exported as Jellyfish `.mat` (for analysis in BGBlitz, XG, gnubg) |
| `bgcli` | Developer tool: move lists, evaluation, analysis, rollouts, benchmarks, duels, speed tests, checks against gnubg |
| `bgtrain` | Dataset building, training, relabelling, cube labels, cube-head training |

The scripts in `tools/` drive gnubg for labels, benchmarks and matches, run the expert iteration
(`iterate.sh`), and combine match results (`agg_matches.py`).

---

## 9. Results

See `docs/RESULTS.md`. In short:

* **Against gnubg at equal depth** (money games with the cube, duplicate dice):

  | Level | DarkBGHelmet ppg |
  |---|---|
  | our 2-ply vs gnubg 1-ply | +0.13 ± 0.04 |
  | our 3-ply vs gnubg 2-ply | +0.15 ± 0.04 |
  | our 4-ply vs gnubg 3-ply | +0.11 ± 0.05 |

  At equal depth our engine uses about 2.5–5× more time than gnubg.
* **Self-play money games analysed by BGBlitz:** PR about 0.85 at BGBlitz 4-ply and 0.39–0.50 at
  5-ply.

---

## 10. Possible next steps

* **Time controls** (time per move plus a match budget) with iterative deepening, as soon as
  engine interfaces such as BGBlitz's support them.
* **Faster deep search:** for example a cheaper evaluator to pre-rank plays at inner nodes, or
  selective deepening.
* **Better teachers:** 4-ply or rollout labels to go beyond the current plateau.
* **Exact two-sided bearoff database** and cube head training on match positions.
