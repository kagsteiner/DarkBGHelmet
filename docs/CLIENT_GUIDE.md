# Using the DarkBGHelmet engine from another app (e.g. secondroll)

This guide is written for a developer, or an AI coding assistant, who works on an app running on
the same VPS as the engine. It explains how to ask the engine to evaluate positions, rank checker
plays, judge a play that was actually made, and analyse cube decisions.

The complete reference is [`docs/API.md`](API.md). This guide focuses on practical use.

---

## 1. Where the engine runs

| Item | Value |
|---|---|
| Base URL | `http://127.0.0.1:3050` (VPS-local only, no HTTPS, not reachable from outside) |
| Process | pm2 app `bgengine`; rebuild/restart with `~/builddarkbg.sh` |
| Protocol | HTTP + JSON. All analysis endpoints are `POST` with a JSON body; responses are JSON |
| Health check | `GET /v1/health` returns `{"ok":true}` |
| Info | `GET /v1/info` returns version, loaded networks, default depth |

**Call it from the server side only (Node).** The port is not exposed through nginx and should
not be. Deep searches cost CPU.

**One request at a time.** The engine processes requests one after another; concurrent requests
wait in line. Each one uses up to `--threads` cores (2 on the VPS). Don't fire hundreds of
requests in parallel; send them sequentially or with a small concurrency (1–2).

---

## 2. The ten-second version

```js
const ENGINE = "http://127.0.0.1:3050";

async function engine(path, body, timeoutMs = 30000) {
  const res = await fetch(ENGINE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(`engine ${path}: ${json.error}`);
  return json;
}

// Rank the plays for 3-1 in the opening position, and judge the play "13/9" (actually 13/10 10/9)
const r = await engine("/v1/move", {
  position: { gnubg: "4HPwATDgc/ABMA" },
  dice: [3, 1],
  played: "13/10 10/9",
  options: { plies: 3, max_moves: 5 },
});
console.log(r.best);            // "8/5 6/5"
console.log(r.played);          // { move: "13/9", rank: 5, diff: -0.251, equity: -0.021, plies: 3, is_best: false }
```

Node 18+ has `fetch` built in. With older Node versions use `node-fetch`.

---

## 3. Core concepts (read this once)

**Perspective.** Everything is from the view of the **player on roll** (the player who has the
dice, or who is about to roll). Equities, probabilities and decisions all refer to that player.

**Plies (search depth)** follow the XG convention:

| `plies` | Meaning | Typical time on the VPS | Use for |
|---|---|---|---|
| 1 | static network evaluation | about 1 ms | bulk pre-screening |
| 2 | looks one roll ahead | a few ms | fast analysis |
| **3** | two rolls ahead (≈ gnubg 2-ply "World Class") | tens of ms | **default; good analysis quality** |
| 4 | three rolls ahead | about 0.3–2 s | high-quality analysis of selected positions |
| 5+ | very deep | seconds to minutes | avoid in interactive use |

The response field `time_ms` shows the actual cost. Measure it on the VPS with your positions.

**Equity**
- **Money game:** points per game normalised to the current cube, from −3 to +3. A value of
  +0.25 means the player on roll expects to win 0.25 points per game.
- **Match:** `equity` is the money-equivalent equity (EMG), so error sizes are comparable across
  scores, and `mwc` is the match winning chance (0…1). Error classification (§7) should use
  `equity`/`diff`, as XG and gnubg do.

**Probabilities** are cubeless and unconditional: `win`, `win_gammon` (which includes
backgammons), `win_backgammon`, `lose`, `lose_gammon`, `lose_backgammon`.

**Cube.** The engine handles the cube fully: Jacoby, beavers, Crawford, post-Crawford, dead cubes
and match equity (Kazaross-XG2 table). Checker plays are already chosen with the cube state and
match score in mind. For example, it ignores gammon risk when the opponent is 1-away.

---

## 4. Describing a position

Each request needs a `position` (one of three formats), optionally `match` and, for checker plays,
`dice`.

### 4.1 Position formats

**a) XGID.** Easiest if you have it; it also contains cube, score, dice and turn:

```json
"position": { "xgid": "XGID=-b----E-C---eE---c-e----B-:0:0:1:52:0:0:0:0:10" }
```

The player whose turn it is in the XGID becomes the "player on roll".

**b) gnubg Position ID, optionally with Match ID:**

```json
"position": { "gnubg": "4HPwATDgc/ABMA" }
"position": { "gnubg": "4HPwATDgc/ABMA:cAkAAAAAAAAA" }
```

The Position ID is always from the view of the player on roll. The Match ID adds cube, score and
dice.

**c) A points array.** This is the format to use if your app keeps its own board, for example
parsed from DailyGammon. It's an array of 26 integers **from the view of the player on roll**:

| Index | Content |
|---|---|
| `0` | number of checkers of the **opponent** on the bar (sign ignored) |
| `1 … 24` | points 1–24 in the on-roll player's numbering (point 1 = their ace point, which they bear off from). **Positive = own checkers, negative = opponent checkers.** |
| `25` | number of checkers of the **player on roll** on the bar |

Checkers that are not on the board (15 minus the ones listed) count as borne off.

```json
"position": { "points": [0, -2,0,0,0,0,5, 0,3,0,0,0,-5, 5,0,0,0,-3,0, -5,0,0,0,0,2, 0] }
```

That's the starting position: the player on roll has 2 on the 24 point, 5 on 13, 3 on 8 and 5
on 6, and the opponent's checkers appear as negative numbers on the mirrored points.

If your board is stored from a fixed player's view (say "player A" always moves from 24 to 1),
then for positions where **player B** is on roll, **mirror** the array: point *i* becomes
25 − *i*, the signs flip, and the bars swap:

```js
// board: { points: [ ... 24 entries for points 1..24 from A's view, +A / -B ],
//          barA, barB }   (A moves 24 -> 1)
function toEnginePoints(board, onRoll /* "A" | "B" */) {
  const p = new Array(26).fill(0);
  if (onRoll === "A") {
    for (let i = 1; i <= 24; i++) p[i] = board.points[i - 1];
    p[25] = board.barA; p[0] = board.barB;
  } else {
    for (let i = 1; i <= 24; i++) p[i] = -board.points[24 - i]; // point i (B's view) = A's 25-i
    p[25] = board.barB; p[0] = board.barA;
  }
  return p;
}
```

The engine rejects invalid boards with HTTP 400: more than 15 checkers, both colours on one
point, or a game that is already over.

### 4.2 Match / cube state

```json
"match": { "length": 7, "score": [2, 4], "cube": 2, "owner": "opponent", "crawford": false }
```

| Field | Meaning | Default |
|---|---|---|
| `length` | 0 = money game, otherwise 1…25 | 0 |
| `score` | `[on roll, opponent]`: points already won (not "away") | `[0, 0]` |
| `cube` | cube value (1, 2, 4, …) | 1 |
| `owner` | `"center"`, `"onroll"` or `"opponent"` | `"center"` |
| `crawford` | `true` if this game is the Crawford game | `false` |
| `jacoby` | money only | `true` |
| `beavers` | money only | `true` |

Values in `match` override what the XGID or Match ID say. A 1-cube must be centred. Scores must
be below the match length.

### 4.3 Dice

`"dice": [6, 4]`. The order doesn't matter. Required for `/v1/move`, unless the XGID or Match
ID already contains dice.

---

## 5. Endpoints

### 5.1 `POST /v1/move`: rank checker plays, judge a played move

Request:

```json
{
  "position": { "xgid": "XGID=..." },
  "match":    { "length": 7, "score": [4, 6] },
  "dice":     [6, 4],
  "played":   "24/14",
  "options":  { "plies": 3, "max_moves": 5 }
}
```

Response (abridged; the numbers are real, from a 3-ply run on the Mac):

```json
{
  "best": "8/2 6/2",
  "legal_moves": 14,
  "plies": 3,
  "moves": [
    { "move": "8/2 6/2", "equity": 0.3406, "diff": 0.0, "mwc": 0.3271, "plies": 3,
      "probabilities": { "win": 0.499, "win_gammon": 0.155, "...": "..." },
      "position_after": { "gnubg": "hmfwATDgc/ABMA:cAngAEAAMAAA", "xgid": "XGID=..." } },
    { "move": "24/18 13/9", "equity": 0.3135, "diff": -0.0271, "...": "..." }
  ],
  "played": { "move": "24/14", "rank": 3, "equity": 0.2879, "diff": -0.0527,
              "mwc": 0.3143, "mwc_diff": -0.0129, "plies": 3, "is_best": false },
  "time_ms": 69.0
}
```

**Fields**
- `moves` is sorted best first. Each play has `equity`, `diff` (to the best play; ≤ 0),
  `plies` (the depth it was evaluated at), the probabilities after the play, and the position
  after the play (from the opponent's view).
- **Move filters.** Only promising plays are searched at full depth; the others keep a
  shallower value, as in gnubg and XG. Plays further down the list may show a lower `plies`.
- **`played`** (optional) is the play actually made:
  - It accepts any common notation, e.g. `"24/14"`, `"24/18/14"`, `"24/18 18/14"`, `"bar/22*"`,
    `"6/off(2)"`, `"8/2(2) 6/2(2)"`. Hit markers `*` are optional.
  - **It's always searched at the full requested depth**, so its `diff` is a fair comparison
    with the best play. That makes it the right field for "how big was the player's error?".
  - `rank` is its position among all legal plays by equity (1 = best).
  - If the play isn't legal for the dice, the response is HTTP 400 with an explanation.
  - **No legal move** (the player is closed out): `legal_moves` is 1, the only entry has
    `move: "(no move)"`, and `played` can be any string.
- `max_moves` limits the returned list (default 20). It doesn't change the analysis.

### 5.2 `POST /v1/cube`: cube decision of the player on roll (before rolling)

Request:

```json
{ "position": { "gnubg": "hmfhASRw5+ABMA" }, "match": { "length": 7, "score": [3, 5] },
  "options": { "plies": 3 } }
```

Response:

```json
{
  "decision": "Double, take",
  "take": "take",
  "cube_available": true,
  "equity": { "no_double": 0.787, "double_take": 0.9599, "double_pass": 1.0 },
  "mwc":    { "no_double": 0.3776, "double_take": 0.3962, "double_pass": 0.4005 },
  "probabilities": { "win": 0.594, "win_gammon": 0.198, "...": "..." },
  "cube_efficiency": 0.767,
  "plies": 3,
  "time_ms": 9
}
```

**Fields**

- `decision` is one of:
  - `No double, take`
  - `No double, beaver`
  - `Double, take`
  - `Double, pass`
  - `Too good to double, pass`
  - `Too good to double, take`
  - `Cube not available`

  When the player on roll owns the cube, the wording is "Redouble" instead of "Double".
- `take` is the correct response if a double were offered (`take` / `pass` / `beaver`).
- The `equity` triple is from the doubler's view. It's all that's needed to grade both sides'
  decisions (§7). `mwc` appears in match play only.
- `cube_available: false` means the player on roll may not double (the opponent owns the cube,
  the cube is dead, or it's the Crawford game).
- The cube depth equals `plies` on the VPS, because the server runs with `--cube-extra 0`.
  `options.cube_plies` sets it explicitly.

### 5.3 `POST /v1/evaluate`: position evaluation (before the roll)

Returns `probabilities`, `cubeless_equity`, `cubeful_equity` (the equity of the correct cube
action), the same `cube` object as `/v1/cube`, and in match play `match_equity`
(`{cubeless, cubeful}` as MWC) and `match` (`{length, away, crawford, cube, owner}`). This is
useful for "who's ahead and by how much" displays.

### 5.4 `POST /v1/analyze`: cube plus checker play in one call

```json
{ "position": { "xgid": "..." }, "dice": [5, 2], "played": "13/8 13/11", "options": { "plies": 3 } }
```

Response: `{ "cube": { ...as /v1/cube... }, "checker": { ...as /v1/move... }, "position": {...} }`.
Without dice only `cube` is returned. This is convenient for analysing one turn: first the cube
decision before the roll, then the play.

### 5.5 `POST /v1/rollout`: Monte Carlo rollout (slow; use sparingly)

Options: `trials` (default 1296), `truncate`, `candidates` (with dice). It returns rolled-out
probabilities and equities with standard errors. It typically takes seconds to minutes; it's not
meant for bulk analysis.

### Errors

- **HTTP 400** with `{"error": "..."}`: the request is malformed or the position is illegal. The
  message says what's wrong. Fix the input; don't retry.
- **HTTP 500:** engine problem. Check with `pm2 logs bgengine`.
- **Connection refused:** the engine isn't running. Check with `pm2 status bgengine`.

---

## 6. Analysing a whole match (recipe)

For each turn of a game, in order:

1. **Before the roll** (when the player could double): call `/v1/cube` with the position, score
   and cube state from the on-roll player's view. Compare with what happened (§7).
2. **If a double was offered:** the taker's decision is graded with the same `/v1/cube` result
   (the doubler is on roll; look at `take` and the equity triple).
3. **After the roll:** call `/v1/move` with `dice` and `played` set to the move from the match
   file. Record `played.diff`.
4. Apply the move to your board, switch the player on roll, and continue.

Tips:

- Skip forced moves (`legal_moves == 1`): there's no decision.
- `/v1/analyze` combines steps 1 and 3 in one request.
- To build the next position you can use `position_after` from the best play, but only if the
  player made that play. In general, apply the played move to your own board.
- Throughput at 3-ply on the VPS: a typical game has about 40 decisions, so roughly a few seconds
  per game. For long matches, analyse in a background job and store the results.

---

## 7. Grading decisions

**Checker play.** Use `played.diff` (negative = equity lost). The usual thresholds in equity:

| Category | XG / gnubg style | BGBlitz style |
|---|---|---|
| doubtful / small error | > 0.02 – 0.04 | > 0.02 |
| error / bad | > 0.04 – 0.08 | > 0.05 |
| blunder / very bad | > 0.08 – 0.16 | > 0.12 |

**Cube.** From one `/v1/cube` response, with `nd`, `dt`, `dp` from `equity` (doubler's view):

```js
function cubeErrors({ nd, dt, dp }) {
  const doubled = Math.min(dt, dp);          // what doubling is worth (the opponent picks take/pass)
  const correctDouble = doubled > nd;
  return {
    // Error if the player doubled when they shouldn't have, or didn't when they should
    errorIfDoubled:    correctDouble ? 0 : nd - doubled,
    errorIfNotDoubled: correctDouble ? doubled - nd : 0,
    // Taker's decision (positive = equity lost by the taker)
    errorIfTaken:      dt > dp ? dt - dp : 0,  // should have passed
    errorIfPassed:     dt < dp ? dp - dt : 0,  // should have taken
  };
}
```

"Too good to double" positions (where `nd > dp`) are handled by this too: doubling is then an
error of `nd - min(dt, dp)`.

**Match play.** The `equity` values are already money-equivalent (EMG), so the same thresholds
apply. For "match winning chance lost", use `mwc` (`played.mwc_diff` for checker plays, and the
`mwc` triple for the cube).

---

## 8. Quiz ideas

- **Multiple-choice checker quiz.** Use `moves[0]` (correct) plus 2–3 plausible alternatives
  from `moves[1..4]`, preferably with `diff` between −0.03 and −0.30. Show `diff` after the
  answer.
- **Cube quiz.** Ask the player to choose "No double / Double, take / Double, pass" (or
  "Take / Pass" for the taker); the answer is `decision` / `take`, and the explanation comes
  from the equity triple.
- **Pre-computing.** Analyse quiz positions once at `plies: 4`, store the JSON, and serve from
  the database. That avoids engine load at quiz time and gives the most accurate answers.

---

## 9. Practical notes

- **Timeouts:** 30 s is generous for `plies` ≤ 3; for 4-ply on complex positions allow 60 s.
  Use `max_moves` to keep responses small.
- **Determinism:** the same request gives the same answer, apart from tiny floating-point
  differences when the thread count changes.
- **Engine version:** `GET /v1/info` shows `network` (e.g. `dbgh-h1024-ep12+cube`). Store it
  with your analyses, so you know which version produced a result.
- **Updating the engine:** run `~/builddarkbg.sh` on the VPS. That's a short downtime of a few
  seconds, so callers should retry once after a connection error.
