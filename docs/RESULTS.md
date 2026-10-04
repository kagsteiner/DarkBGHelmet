# Results against GNU Backgammon (2026-10-04)

Network: `big1024_cube` (v0.4). The setup for all three levels:

* **Games:** money games with the cube and the Jacoby rule. Games are played in duplicate:
  each dice sequence is replayed with the roles swapped.
* **Depth:** equal search depth on both sides. Our N-ply equals gnubg (N−1)-ply, and cube
  decisions use the same depth as checker play.
* **Hardware:** both engines single-threaded, 16 games in parallel on an M5 Max.
* **Harness:** `tools/gnubg_levels.sh`, summarised with `tools/agg_matches.py`.

| Level | Games | DarkBGHelmet ppg | Significance |
|---|---|---|---|
| our 2-ply vs gnubg 1-ply | 4,992 | **+0.127 ± 0.038** | 3.4 σ |
| our 3-ply vs gnubg 2-ply | 4,992 | **+0.147 ± 0.035** | 4.1 σ |
| our 4-ply vs gnubg 3-ply | 2,016 | **+0.111 ± 0.053** | 2.1 σ |

Average thinking time per decision. gnubg's numbers exclude its Python API overhead of
about 12 ms per call.

| Level | ours: move / cube | gnubg: move / cube |
|---|---|---|
| 2-ply vs 1-ply | 4.4 / 1.1 ms | ≈0 / ≈0 ms |
| 3-ply vs 2-ply | 85.5 / 23.2 ms | 34.8 / 5.5 ms |
| 4-ply vs 3-ply | 1017 / 493 ms | 187 / 200 ms |

Cube statistics for 3-ply vs 2-ply: we doubled 3,270 times and gnubg 3,189 times. gnubg
passed 1,606 of our doubles; we passed 895 of gnubg's.

BGBlitz analysis of self-play money games (same network): PR 0.83/0.86 at 4-ply and
0.39/0.50 at 5-ply.
