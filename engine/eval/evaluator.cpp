#include "evaluator.hpp"

#include <cmath>

#include "eval/cube.hpp"

namespace bg {

bool Evaluator::init(const std::string& net_path, const std::string& data_dir, std::string* err) {
  bearoff_db().init(data_dir + "/bearoff_os15.bin");
  if (!net_path.empty()) {
    Network n;
    if (!n.load(net_path, err)) return false;
    net_ = std::move(n);
  }
  return true;
}

Probs Evaluator::terminal(const Board& b) {
  Probs p = Probs::zero();
  int w = b.winner();
  GameResult r = b.result_for_winner(w);
  Probs q = Probs::zero();
  q[kWin] = 1.f;
  q[kWinG] = r >= GameResult::Gammon ? 1.f : 0.f;
  q[kWinBG] = r >= GameResult::Backgammon ? 1.f : 0.f;
  p = w == 0 ? q : q.inverted();
  return p;
}

Probs Evaluator::heuristic(const Board& b) const {
  // Fallback when no network is loaded: race formula on pip counts.
  float pm = float(b.pips(0)), po = float(b.pips(1));
  float d = (po - pm + 4.f) / std::sqrt(std::max(10.f, pm + po));
  Probs p = Probs::zero();
  p[kWin] = 1.f / (1.f + std::exp(-1.2f * d));
  p.sanitize();
  return p;
}

Probs Evaluator::eval_net(const Board& b, const Accumulator& acc, int stm) const {
  // b is oriented with `stm` on roll at side stm; features need side 0 = on roll.
  const Board v = stm == 0 ? b : b.swapped();
  float extra[kExtraFeatures];
  extra_features(v, extra);
  return net_forward(net_, acc, stm, extra, head_for(classify(v)));
}

Probs Evaluator::evaluate(const Board& b) const {
  switch (classify(b)) {
    case PosClass::Over: return terminal(b);
    case PosClass::Bearoff: return bearoff_db().evaluate(b);
    default: break;
  }
  if (net_.empty()) return heuristic(b);
  Accumulator acc;
  acc_refresh(net_, b, acc);
  return eval_net(b, acc, 0);
}

void Evaluator::cube_x(const Board& b, float x[3]) const {
  const PosClass cls = classify(b);
  if (!cube_head_active() || cls == PosClass::Over || cls == PosClass::Bearoff) {
    x[0] = x[1] = x[2] = cube_efficiency(b);
    return;
  }
  Accumulator acc;
  acc_refresh(net_, b, acc);
  float extra[kExtraFeatures];
  extra_features(b, extra);
  float h3[kMaxDense];
  const int head = head_for(cls);
  const Probs p = net_forward(net_, acc, 0, extra, head, h3);
  net_cube_x(net_, h3, p, head, x);
}

void Evaluator::evaluate_moves(const Board& b, const std::vector<Move>& moves, std::vector<Probs>& out) const {
  out.resize(moves.size());
  Accumulator pa, ca;
  bool have_parent = false;
  for (size_t i = 0; i < moves.size(); ++i) {
    const Board& c = moves[i].after;  // mover = side 0, opponent to roll next
    const Board opp_view = c.swapped();
    PosClass cls = classify(opp_view);
    if (cls == PosClass::Over) { out[i] = terminal(c); continue; }
    if (cls == PosClass::Bearoff) { out[i] = bearoff_db().evaluate(opp_view).inverted(); continue; }
    if (net_.empty()) { out[i] = heuristic(opp_view).inverted(); continue; }
    if (!have_parent) { acc_refresh(net_, b, pa); have_parent = true; }
    acc_update(net_, b, pa, c, ca);
    out[i] = eval_net(c, ca, 1).inverted();
  }
}

}  // namespace bg
