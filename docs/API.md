# DarkBGHelmet engine interfaces

Backgammon has no common engine protocol comparable to UCI in chess. DarkBGHelmet offers two
front ends over the same engine:

1. **HTTP JSON API** (`bgserver`) for GUIs, bots, and web apps.
2. **BGEI**, a line-based, UCI-like text protocol on stdin/stdout (`bgei`), for scripting and testing.

Conventions used everywhere:

* **Perspective.** Every result is from the view of the **player on roll**.
* **Plies** use the XG convention. 1-ply is the static evaluation. 2-ply looks one roll ahead,
  and so on, up to 7.
* **Probabilities** are cubeless and unconditional: `win`, `win_gammon` (includes backgammons),
  `win_backgammon`, `lose`, `lose_gammon`, `lose_backgammon`.
* **Equities.**
  * In **money games** they are normalised to the current cube value.
  * In **matches**, `equity` is the money-equivalent equity (EMG) and `mwc` is match winning
    chances (0..1).

---

## HTTP API (`/v1`)

Start the server:

```
bgserver --port 8080 --threads 16 --net nets/latest.bgnn --plies 3
```

Several networks can be loaded at once and chosen per request. For example:

```
bgserver --net nets/latest.bgnn --add-net fast=nets/iter1_cube.bgnn
```

Then `"options": {"network": "fast"}` selects the second one. `/v1/info` lists all loaded networks.

| Method | Path | Purpose |
|---|---|---|
| GET  | `/v1/info` | Engine name, version, network id, limits |
| GET  | `/v1/health` | Liveness check |
| POST | `/v1/evaluate` | Probabilities, cubeless/cubeful equity, match equity, cube analysis |
| POST | `/v1/move` | Ranked checker plays for the given dice |
| POST | `/v1/cube` | Cube action: double / no double / too good, take / pass / beaver |
| POST | `/v1/analyze` | Cube analysis plus, if dice are given, the checker play analysis |
| POST | `/v1/rollout` | Monte Carlo rollout of the position or, with dice, of the best candidate plays |

### Request

```json
{
  "position": { "xgid": "XGID=-b----E-C---eE---c-e----B-:0:0:1:52:0:0:0:0:10" },
  "match":    { "length": 7, "score": [2, 4], "cube": 2, "owner": "opponent",
                "crawford": false, "jacoby": true, "beavers": true },
  "dice":     [5, 2],
  "played":   "13/8 13/11",
  "options":  { "plies": 4, "cubeful": true, "candidates": 8, "max_moves": 20 }
}
```

**`position`** takes exactly one of the following:

* `{"xgid": "XGID=..."}`. The XGID's match, cube and dice fields are used as well.
* `{"gnubg": "<PositionID>:<MatchID>"}` or `{"gnubg": "<PositionID>"}`.
* `{"points": [26 ints]}`, seen from the player on roll:
  * Index 1..24 are the points 1..24. A positive count means checkers of the player on roll;
    a negative count means opponent checkers.
  * Index 25 is the bar of the player on roll.
  * Index 0 is the opponent's bar.

  Checkers not on the board count as borne off.

**`match`** is optional and overrides the values from the position ID:

* `length`: `0` means a money game; otherwise 1..25.
* `score`: `[on roll, opponent]`, the points already won.
* `owner`: one of `center`, `onroll`, `opponent`.

**`dice`** is required for `/v1/move`, unless the position ID already includes dice.

**`played`** (optional, `/v1/move` and `/v1/analyze`) is the play actually made, in any common
notation. It's always searched at the full depth, and the response contains
`"played": {move, rank, equity, diff, plies, is_best}` (plus `mwc` and `mwc_diff` in match
play). An illegal play gives HTTP 400.

**`options`**:

| Option | Meaning |
|---|---|
| `plies` | Search depth, 1..7 |
| `cube_extra_plies` | Cube decisions (`/v1/cube`, `/v1/evaluate`) search this many plies deeper than `plies`. Default 2 (server flag `--cube-extra`); 0 = same depth |
| `cube_plies` | Exact depth for cube decisions (overrides `cube_extra_plies`) |
| `network` | Name of a loaded network (see `/v1/info`); default `default`. Load more with `bgserver --add-net name=file` |
| `cubeful` | Cubeful (default) or cubeless search |
| `candidates` | Maximum number of plays searched at full depth |
| `max_moves` | Maximum number of plays returned |
| `sampled_rolls` | Rolls searched per deep chance node: 21 = full expansion, fewer = faster |

### `/v1/move` response

```json
{
  "dice": [5, 2], "plies": 4, "legal_moves": 8, "best": "13/11 13/8",
  "moves": [
    { "move": "13/11 13/8", "equity": 0.0253, "diff": 0.0, "plies": 4,
      "mwc": 0.512,
      "probabilities": { "win": 0.50, "win_gammon": 0.14, "...": "..." },
      "position_after": { "gnubg": "...:...", "xgid": "XGID=..." } }
  ],
  "nodes": 1234, "time_ms": 85.1
}
```

* `mwc` appears only in match play.
* `position_after` is seen from the opponent, who is on roll next.

### `/v1/cube` response

```json
{
  "decision": "Double, take", "take": "take", "cube_available": true,
  "equity": { "no_double": 0.62, "double_take": 0.71, "double_pass": 1.0 },
  "mwc":    { "no_double": 0.61, "double_take": 0.63, "double_pass": 0.68 },
  "probabilities": { "...": "..." }, "plies": 4, "cube_efficiency": 0.76
}
```

`mwc` appears only in match play.

The possible `decision` values are:

* `No double, take`
* `No double, beaver`
* `Double, take`
* `Double, pass`
* `Too good to double, pass`
* `Too good to double, take`
* `Cube not available`

When the player on roll owns the cube, the wording is "redouble" instead of "double".

### `/v1/evaluate` response

The response contains:

* `probabilities`
* `cubeless_equity`
* `cubeful_equity`: EMG of the correct cube action
* `cube`: the same object as `/v1/cube`
* in match play only:
  * `match_equity`: `{cubeless, cubeful}` as MWC
  * `match`: `{length, away, crawford, cube, owner}`

### Experimental: time-controlled rollout search

`/v1/move` and `/v1/cube` (and so `/v1/analyze`) accept `"options": {"mc": {...}}`. Instead of
a fixed-depth search, the engine then uses a 2/3-ply pre-ranking plus truncated rollouts with
paired Thompson sampling within a time budget:

```json
"options": { "mc": { "time_ms": 2000, "truncate": 3, "prior_games": 100, "prefilter_plies": 3 } }
```

Fields:

| Field | Default |
|---|---|
| `time_ms` | 1000 |
| `cube_time_ms` | half of `time_ms` |
| `truncate` | 7 |
| `batch` | 36 |
| `prefilter_plies` | 2 |
| `candidates` | 8 |
| `window` | 0.20 |
| `prior_games` | 36 |
| `cube_window` | 0.25 |
| `variance_reduction` | false |

The response gains an `mc` object (`candidates`, `games`, `batches`, `time_ms`, `used_rollouts`),
and each play gets a `rollouts` count.

At equal time this mode is about as strong as the normal search, not stronger (see
`docs/ENGINE.md` §6a). It's meant for deep analysis on machines with many cores, not for small
servers.

### `/v1/rollout`

`options` for rollouts:

| Option | Meaning |
|---|---|
| `trials` | Number of games, default 1296. The first two rolls are stratified, so multiples of 1296 work best |
| `truncate` | Stop after this many plies and use the static evaluation (0 = play to the end) |
| `cubeful` | Play with the cube (default) or cubeless |
| `variance_reduction` | Dahl-style luck removal: about 16x lower variance, about 21x the cost per trial |
| `candidates` | With dice: how many of the best 1-ply plays to roll out |

The response contains `probabilities`, `cubeless_equity`, `cubeful_equity` and `std_error`, plus
`mwc` in match play. With dice, one such object is returned per candidate play, under `moves`.
The cubeful value of a position rollout assumes the player on roll does not double first.

### Errors

* Malformed requests return HTTP 400 with `{"error": "..."}`.
* Engine failures return HTTP 500.

---

## BGEI text protocol

`bgei [--net nets/latest.bgnn] [--threads N]` reads commands from stdin and writes answers to
stdout, one per line:

| Command | Answer |
|---|---|
| `bgei` | `id name …`, `id network …`, `option …` lines, then `bgeiok` |
| `isready` | `readyok` |
| `setoption name Threads\|Plies\|Net\|CubeExtraPlies value <v>` | (none, or `error …`) |
| `setoption name AddNet value <name>=<file>` / `setoption name Network value <name>` | Load an additional network / use it for the following commands |
| `position start` / `position xgid <XGID>` / `position gnubg <posid>[:<matchid>]` | (none) |
| `match length <n> score <a> <b> cube <c> owner <center\|onroll\|opponent> crawford <0\|1>` | (none) |
| `go move dice <d1d2> [plies n]` | `info rank … move … equity … diff … plies …` lines, then `bestmove <move>` |
| `go cube [plies n]` | `bestcube <decision>; take <take\|pass\|beaver>; nd … dt … dp …` |
| `go eval [plies n]` | `eval win … wg … wbg … lg … lbg … cubeless … cubeful …` |
| `go rollout [dice d1d2] [trials n] [truncate t]` | One `rollout <move\|position> win … cubeful … se … trials …` line per candidate |
| `json <request>` | One line: the `/v1/analyze` JSON response |
| `quit` | Exits |

Moves use gnubg notation, for example `bar/22* 13/11`, `6/off(2)`, `24/18/13`.

Example session:

```
> bgei
< id name DarkBGHelmet 0.1.0
< bgeiok
> position xgid XGID=-b----E-C---eE---c-e----B-:0:0:1:52:0:0:0:0:10
> go move plies 3
< info rank 1 move 13/11 13/8 equity 0.0253 diff 0.0000 plies 3
< bestmove 13/11 13/8
```
