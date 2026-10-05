// Static (XG "1-ply") cubeless evaluation: terminal positions, bearoff database,
// otherwise the network.
#pragma once

#include <string>
#include <vector>

#include "core/movegen.hpp"
#include "eval/bearoff.hpp"
#include "eval/probs.hpp"
#include "nn/network.hpp"

namespace bg {

class Evaluator {
 public:
  // Loads the network (if the file exists) and the bearoff database.
  bool init(const std::string& net_path, const std::string& data_dir, std::string* err = nullptr);
  void set_network(const Network& n) { net_ = n; net_.prepare(); }
  const Network& network() const { return net_; }
  bool has_network() const { return !net_.empty(); }

  // Board side 0 on roll. Probabilities from side 0's view.
  Probs evaluate(const Board& b) const;

  // Evaluates the position after each play in `moves` (from `b`, side 0 = mover).
  // out[i] is from the mover's view. Uses incremental accumulator updates.
  void evaluate_moves(const Board& b, const std::vector<Move>& moves, std::vector<Probs>& out) const;

  // Cube efficiency for the cube states (centered, owned by the player on roll,
  // owned by the opponent), from the network's cube head. The fixed formula
  // (cube_efficiency) is used for bearoff-database and finished positions, and
  // for networks without a cube head (only developer tools load those).
  void cube_x(const Board& b, float x[3]) const;
  bool has_cube_head() const { return net_.has_cube_head(); }

  // Probabilities of a finished game from the view of board side 0 (the winner
  // is whoever has no checkers left).
  static Probs terminal(const Board& b);

 private:
  Probs eval_net(const Board& b, const Accumulator& acc, int stm) const;
  Probs heuristic(const Board& b) const;
  Network net_;
};

}  // namespace bg
