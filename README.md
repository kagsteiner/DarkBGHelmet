# DarkBGHelmet

A backgammon engine in C++20:

* an NNUE-style neural evaluator
* a selective expectimax search up to 7 plies (XG convention)
* cubeful money and match play (1–25 points)
* rollouts
* an HTTP JSON API and a UCI-like text protocol

The design rationale is in [PLAN.md](PLAN.md). The interfaces are described in
[docs/API.md](docs/API.md).

* **How it works:** [docs/ENGINE.md](docs/ENGINE.md) covers the network, the search, cube and
  match handling, and training.
* **Results against gnubg:** [docs/RESULTS.md](docs/RESULTS.md).
* **Running it on a server:** [deploy/DEPLOY.md](deploy/DEPLOY.md).
* **Using it from your own app** (analysis, grading played moves, quizzes):
  [docs/CLIENT_GUIDE.md](docs/CLIENT_GUIDE.md).

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bgtests
```

The build produces these binaries:

| Binary | Purpose |
|---|---|
| `bgserver` | HTTP JSON API (`/v1/evaluate`, `/v1/move`, `/v1/cube`, `/v1/analyze`, `/v1/rollout`) |
| `bgei` | BGEI text protocol on stdin/stdout (UCI-like) |
| `bgcli` | Developer tool: `moves`, `eval`, `move`, `cube`, `rollout`, `bench`, `speed`, `perft`, checks against gnubg |
| `bgtrain` | Dataset builder and network trainer |
| `bgplay` | Play against the engine in a terminal |
| `bgselfplay` | Engine self-play exported as Jellyfish `.mat` |
| `bgtests` | Unit tests |

The first run creates `data/bearoff_os15.bin`, the one-sided bearoff database (under a second to generate).

**Local settings.** Copy `.env.example` to `.env` and set `THREADS=<n>`. That's the default number
of search threads for `bgplay`, `bgselfplay`, `bgei` and `bgserver`, read from the folder you start
them in. An environment variable `THREADS` takes precedence over `.env`, and `--threads` on the
command line takes precedence over both.

## Quick start

```
./build/bgcli move 4HPwATDgc/ABMA 31 3            # rank the plays for 3-1 at 3-ply
./build/bgcli cube "XGID=-b----E-C---eE---c-e----B-:0:0:1:00:3:3:0:5:10" 4
./build/bgserver --port 8080 --plies 3 &
curl -s -X POST localhost:8080/v1/move \
  -d '{"position":{"gnubg":"4HPwATDgc/ABMA"},"dice":[3,1],"options":{"plies":3}}'
```

## Architecture

| Area | Files | Notes |
|---|---|---|
| Board | `engine/core/board.hpp` | 2 × 128-bit lanes of 25 nibbles, one per side from its own perspective. A checker step is one add and one subtract; changing the side to move swaps the lanes. |
| Move generation | `engine/core/movegen.cpp` | All distinct plays, enforcing the maximum-dice and larger-die rules. Verified against gnubg on 5,300 random positions. |
| Position formats | `engine/core/ids.cpp` | gnubg Position ID / Match ID and XGID |
| Network | `engine/nn/network.cpp` | 780 sparse thermometer features ("at least k checkers on point p") feed two perspective accumulators of 1024 units, updated incrementally. They are joined by 8 hand-crafted features (pips, hit chances, escapes). Then 128 → 64 → three output heads (contact / crashed / race), each with 5 outcome probabilities, plus a cube head with learned cube efficiency. About 1.8 µs per evaluation per core. |
| Bearoff | `engine/eval/bearoff.cpp` | One-sided database for 15 checkers on 6 points, matching gnubg |
| Cube | `engine/eval/cube.cpp` | Kazaross-XG2 MET, generalised Janowski model with recursive live-cube take points. The cube efficiency per position comes from the cube head; the formula is the fallback. |
| Search | `engine/search/search.cpp` | Expectimax over the 21 rolls with static move choice at inner nodes. Cubeful recursion over several cube states at once. Root move filters, a transposition table, and parallel search over candidates × first rolls. Deep chance layers search only k representative rolls and correct the static value of the others (control variate). |
| Rollouts | `engine/search/rollout.cpp` | Cubeful, stratified first two rolls, optional luck variance reduction |
| Training | `training/`, `tools/` | Bootstrapped from gnubg 2-ply labels, then expert iteration with our own 3-ply search labels. Cube head trained on 4-ply cubeful search. Adam trainer in C++. |

## Training a network

```
tools/label_run.sh 14 1200000 2          # 14 gnubg workers write data/labels/*.txt
./build/bgtrain build data/train.bin data/labels/g2_*.txt
./build/bgtrain train data/train.bin nets/new.bgnn --epochs 12 --threads 16 --hidden 256 --lr 3e-3
./build/bgcli bench data/bench/bench1.txt 1 nets/new.bgnn     # error vs gnubg 2-ply
```

## Measuring strength

* `tools/gnubg_bench.py` produces checker-play positions in which gnubg 2-ply (XG 3-ply) has
  evaluated every legal play.
* `bgcli bench` reports our average equity loss per decision against those evaluations.
* `tools/play_vs_gnubg.py` plays duplicate money games with the cube against gnubg.

## How the networks were trained, and the role of GNU Backgammon

The networks were initially trained with the help of
[GNU Backgammon](https://www.gnu.org/software/gnubg/), and this project is grateful for it:

* **Bootstrap.** The first networks learned from gnubg's evaluations. gnubg played games
  against itself (later also against our engine), and every position was labelled with
  gnubg's 2-ply (= XG 3-ply) evaluation via its Python scripting interface
  (`tools/gnubg_label.py`). About 60 million positions were labelled this way: the top
  candidate plays of each roll at 2-ply, the rest at 0-ply.
* **Expert iteration.** From there on, the engine taught itself: positions were relabelled
  with DarkBGHelmet's own 3-ply search, and new networks were trained on those labels only.
  The current default network (`nets/latest.bgnn`, 1024/128/64 with a cube head) was trained
  on our own search labels. Its lineage, however, goes back to the gnubg-trained networks.
* **Cube head.** The learned cube efficiency was trained on DarkBGHelmet's own 4-ply cubeful
  search.
* **Testing.** gnubg also served as the reference throughout: for verifying move generation,
  the bearoff database and the cube formulas, and as the opponent in strength tests
  (`docs/RESULTS.md`).

No gnubg source code is part of this repository. gnubg is only run as an external program.

## License

DarkBGHelmet is released under the [MIT License](LICENSE).

Bundled third-party components keep their own licences:

* [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT), `third_party/httplib.h`
* [nlohmann/json](https://github.com/nlohmann/json) (MIT), `third_party/json.hpp`
* [doctest](https://github.com/doctest/doctest) (MIT), `third_party/doctest.h`
* The Kazaross-XG2 match equity table (`engine/eval/met_kazaross_xg2.inc`) is © 2011 Neil
  Kazaross and was transcribed for gnubg by Michael Petch. It is redistributed under its
  original permissive notice, which is preserved in the file.
