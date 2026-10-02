// Match equity and cube handling.
//
// All values are from the view of the player on roll ("me") and live in one
// "value space": match winning chances (MWC) in match play, or money equity
// normalised to the cube value of the position being analysed.
//
// Cubeful values follow the generalised Janowski model used by gnubg/XG:
//   cubeful = x * live + (1 - x) * dead
// where "dead" counts outcomes at the current cube and "live" is a piecewise
// linear function of P(win) through the take/cash points, which are computed
// recursively over higher cube levels so later redoubles are accounted for.
// x is the cube efficiency.
#pragma once

#include <string>

#include "core/board.hpp"
#include "core/match.hpp"
#include "eval/probs.hpp"

namespace bg {

// MWC for a player `a` away against an opponent `b` away at the start of a game.
// `post_crawford` selects the post-Crawford table when someone is 1-away.
float met_mwc(int a, int b, bool post_crawford);

class CubeModel {
 public:
  explicit CubeModel(const MatchState& ms);

  bool money() const { return ms_.money(); }
  const MatchState& state() const { return ms_; }

  // Value for me if the game ends now and I win `pts` (negative: lose).
  float value(int pts) const;

  // May `player` (0 = me, 1 = opponent) double when the cube is at `cube`?
  bool can_double(int player, int cube) const;

  // Dead-cube value with the cube at `cube`.
  float dead(const Probs& p, int cube, bool jacoby_active) const;

  // Cubeful value with the cube at `cube` owned by `owner`, efficiency x.
  float cubeful(const Probs& p, int cube, CubeOwner owner, float x) const;

  // My take point (in my P(win)) when the opponent doubles from n to 2n, and the
  // opponent's take point (in my P(win)) when I double from n to 2n.
  float my_take_point(const Probs& p, int n) const;
  float opp_take_point(const Probs& p, int n) const;

  // Money equity equivalent (EMG) of a value at the current cube.
  float to_emg(float v) const;

 private:
  float avg_win(const Probs& p, int n, bool jacoby) const;
  float avg_lose(const Probs& p, int n, bool jacoby) const;
  MatchState ms_;
  int max_cube_;
};

// Cube efficiency for a position (side 0 on roll).
float cube_efficiency(const Board& b);

enum class CubeDecision : int8_t {
  NoDouble,          // "No double" / "No redouble"
  DoubleTake,
  DoublePass,
  TooGoodTake,       // too good to double, opponent should take
  TooGoodPass,       // too good to double, opponent should pass
  NoDoubleBeaver,    // money: no double; if doubled, beaver
  NotAvailable       // the player on roll cannot double
};

const char* to_string(CubeDecision d, bool redouble);

struct CubeAnalysis {
  Probs probs;
  float x = 0.76f;
  float cubeless = 0;  // dead cube value at the current cube
  float nd = 0, dt = 0, dp = 0;  // no double / double-take / double-pass (value space)
  bool can_double = false;
  CubeDecision decision = CubeDecision::NotAvailable;
  bool take = true;    // correct response to a double
  bool beaver = false;
  float best() const;  // value of the correct cube action
};

CubeAnalysis analyze_cube(const Probs& p, const MatchState& ms, float x);

}  // namespace bg
