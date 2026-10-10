#include "search.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>

namespace bg {

namespace {

struct Roll {
  int d1, d2;
  float w;  // probability
};

const std::array<Roll, 21>& all_rolls() {
  static const std::array<Roll, 21> r = [] {
    std::array<Roll, 21> a{};
    int k = 0;
    for (int i = 1; i <= 6; ++i)
      for (int j = i; j <= 6; ++j) a[k++] = {i, j, (i == j ? 1.f : 2.f) / 36.f};
    return a;
  }();
  return r;
}

inline CubeOwner flip_owner(CubeOwner o) {
  return o == CubeOwner::Me ? CubeOwner::Opp : (o == CubeOwner::Opp ? CubeOwner::Me : CubeOwner::Center);
}

}  // namespace

// ---------------------------------------------------------------------------
// Transposition table (replace-always, striped locks)

class TranspositionTable {
 public:
  struct Entry {
    u64 key = 0;
    float p[5];
    float v[kMaxStates];
    int32_t n = -1;
  };
  explicit TranspositionTable(size_t mb) {
    size_t n = std::max<size_t>(1024, mb * 1024 * 1024 / sizeof(Entry));
    size_t p2 = 1;
    while (p2 * 2 <= n) p2 *= 2;
    table_.resize(p2);
    mask_ = p2 - 1;
  }
  void clear() {
    for (auto& e : table_) e.n = -1, e.key = 0;
  }
  bool probe(u64 key, int n, Probs& p, float* v) {
    Entry& e = table_[key & mask_];
    std::lock_guard<std::mutex> g(locks_[(key >> 40) & (kLocks - 1)]);
    if (e.key != key || e.n != n) return false;
    std::memcpy(p.p.data(), e.p, sizeof e.p);
    std::memcpy(v, e.v, sizeof(float) * n);
    return true;
  }
  void store(u64 key, int n, const Probs& p, const float* v) {
    Entry& e = table_[key & mask_];
    std::lock_guard<std::mutex> g(locks_[(key >> 40) & (kLocks - 1)]);
    e.key = key;
    e.n = n;
    std::memcpy(e.p, p.p.data(), sizeof e.p);
    std::memcpy(e.v, v, sizeof(float) * n);
  }

 private:
  static constexpr int kLocks = 4096;
  std::vector<Entry> table_;
  size_t mask_;
  std::mutex locks_[kLocks];
};

// ---------------------------------------------------------------------------

SearchConfig SearchConfig::for_plies(int plies) {
  SearchConfig c;
  c.plies = std::clamp(plies, 1, 7);
  if (c.plies <= 4) {
    c.full_layers = 8;
    c.sampled_rolls = 21;
  } else if (c.plies == 5) {
    c.full_layers = 2;
    c.sampled_rolls = 9;
  } else {
    c.full_layers = 2;
    c.sampled_rolls = 7;
  }
  return c;
}

struct Searcher::Root {
  CubeModel cm[2];
  SearchConfig cfg;
  CubeState s0;
  u64 salt;
  bool money;
  Root(const MatchState& ms, const SearchConfig& c) : cm{CubeModel(ms), CubeModel(ms.swapped())}, cfg(c) {
    s0.cube = int16_t(ms.cube);
    s0.owner = ms.owner;
    money = ms.money();
    salt = mix64(u64(ms.length) * 1315423911u ^ u64(ms.score[0]) << 8 ^ u64(ms.score[1]) << 16 ^
                 u64(ms.cube) << 24 ^ u64(ms.owner) << 40 ^ u64(ms.crawford) << 44 ^ u64(ms.jacoby) << 45 ^
                 u64(c.sampled_rolls) << 48 ^ u64(c.full_layers) << 54 ^ u64(c.cubeful) << 60);
  }
  float flip(float v) const { return money ? -v : 1.f - v; }
};

struct Searcher::Ctx {
  std::vector<Move> moves[16];
  std::vector<Probs> probs[16];
};

// Everything needed to expand one node: its states, the child states and the
// mapping between them, and the move-choice utility.
struct Searcher::Node {
  const Board* P;
  int parity;
  StateSet S, C;
  int nd[kMaxStates], dt[kMaxStates];
  float W1, W2, W3, L1, L2, L3;  // dead-cube outcome values at the primary cube
  float utility(const Probs& q) const {
    const float w = q[kWin], l = 1.f - w;
    return (w - q[kWinG]) * W1 + (q[kWinG] - q[kWinBG]) * W2 + q[kWinBG] * W3 + (l - q[kLoseG]) * L1 +
           (q[kLoseG] - q[kLoseBG]) * L2 + q[kLoseBG] * L3;
  }
};

struct Searcher::RollOut {
  Board child;            // position after the chosen play, opponent to move
  bool terminal = false;
  Probs sp;               // static probs, mover's view
  float sv[kMaxStates];   // static values per node state, mover's view
};

int Searcher::StateSet::add(CubeState c) {
  for (int i = 0; i < n; ++i)
    if (s[i] == c) return i;
  if (n >= kMaxStates) return 0;  // cannot happen with sane cube limits
  s[n] = c;
  return n++;
}

u64 Searcher::StateSet::hash() const {
  u64 h = 0x12345 + n;
  for (int i = 0; i < n; ++i) h = mix64(h ^ (u64(s[i].cube) << 4 | u64(s[i].owner)));
  return h;
}

Searcher::Searcher(const Evaluator& ev, int threads, size_t tt_mb)
    : ev_(ev), pool_(threads), tt_(std::make_unique<TranspositionTable>(tt_mb)) {}

Searcher::~Searcher() = default;

void Searcher::set_threads(int threads) { pool_.resize(threads); }
void Searcher::clear_cache() { tt_->clear(); }

void Searcher::make_node(const Root& R, const Board& P, int parity, const StateSet& S, Node& n) const {
  n.P = &P;
  n.parity = parity;
  n.S = S;
  n.C.n = 0;
  const CubeModel& ccm = R.cm[1 - parity];
  for (int i = 0; i < S.n; ++i) {
    CubeState c{S.s[i].cube, flip_owner(S.s[i].owner)};
    n.nd[i] = n.C.add(c);
    n.dt[i] = -1;
    if (R.cfg.cubeful && c.owner != CubeOwner::Opp && ccm.can_double(0, c.cube))
      n.dt[i] = n.C.add(CubeState{int16_t(2 * c.cube), CubeOwner::Opp});
  }
  const CubeModel& cm = R.cm[parity];
  const int pc = S.s[0].cube;
  n.W1 = cm.value(pc); n.W2 = cm.value(2 * pc); n.W3 = cm.value(3 * pc);
  n.L1 = cm.value(-pc); n.L2 = cm.value(-2 * pc); n.L3 = cm.value(-3 * pc);
}

float Searcher::child_optimal(const Root& R, int cp, const float* v, int nd, int dt, int cube) const {
  const float v_nd = v[nd];
  if (dt < 0) return v_nd;
  const float v_dt = v[dt];
  const float v_dp = R.cm[cp].value(cube);
  if (v_dt >= v_nd && v_dp >= v_nd) return std::min(v_dt, v_dp);
  return v_nd;
}

void Searcher::leaf_values(const Root& R, const Board& P, const Probs& p, int parity, const StateSet& S,
                           float* v) const {
  const CubeModel& cm = R.cm[parity];
  float xs[3] = {0.76f, 0.76f, 0.76f};
  if (R.cfg.cubeful) ev_.cube_x(P, xs);  // (centered, owned, opponent owns)
  for (int i = 0; i < S.n; ++i) {
    const CubeState& s = S.s[i];
    if (R.cfg.cubeful) {
      const float x = xs[s.owner == CubeOwner::Center ? 0 : (s.owner == CubeOwner::Me ? 1 : 2)];
      v[i] = cm.cubeful(p, s.cube, s.owner, x);
    } else {
      const bool jac = R.money && cm.state().jacoby && s.owner == CubeOwner::Center && s.cube == 1;
      v[i] = cm.dead(p, s.cube, jac);
    }
  }
}

void Searcher::terminal_values(const Root& R, const Board& after, int parity, const StateSet& S, Probs& p,
                               float* v) const {
  // `after` is from the mover's view; the mover has just borne off its last checker.
  p = Evaluator::terminal(after);
  const CubeModel& cm = R.cm[parity];
  const int k = p[kWinBG] > 0.5f ? 3 : (p[kWinG] > 0.5f ? 2 : 1);
  for (int i = 0; i < S.n; ++i) {
    const CubeState& s = S.s[i];
    const bool jac = R.money && cm.state().jacoby && s.owner == CubeOwner::Center && s.cube == 1;
    v[i] = cm.value((jac ? 1 : k) * s.cube);
  }
}

void Searcher::static_roll(const Root& R, const Node& n, int r, RollOut& o, Ctx& ctx, int layer) const {
  const Roll& roll = all_rolls()[r];
  std::vector<Move>& moves = ctx.moves[layer];
  std::vector<Probs>& probs = ctx.probs[layer];
  generate_moves(*n.P, roll.d1, roll.d2, moves);
  ev_.evaluate_moves(*n.P, moves, probs);
  int best = 0;
  float bu = -1e30f;
  for (size_t i = 0; i < moves.size(); ++i) {
    const float u = n.utility(probs[i]);
    if (u > bu) { bu = u; best = int(i); }
  }
  const Board& after = moves[best].after;
  o.child = after.swapped();
  o.terminal = after.side[0] == 0;
  if (o.terminal) {
    terminal_values(R, after, n.parity, n.S, o.sp, o.sv);
    return;
  }
  o.sp = probs[best];
  float cv[kMaxStates];
  leaf_values(R, o.child, o.sp.inverted(), 1 - n.parity, n.C, cv);
  for (int i = 0; i < n.S.n; ++i)
    o.sv[i] = R.flip(child_optimal(R, 1 - n.parity, cv, n.nd[i], n.dt[i], n.S.s[i].cube));
}

void Searcher::deep_roll(const Root& R, const Node& n, const RollOut& o, int d, Probs& dp, float* dv, Ctx& ctx,
                         int layer) {
  if (o.terminal) {
    dp = o.sp;
    for (int i = 0; i < n.S.n; ++i) dv[i] = o.sv[i];
    return;
  }
  Probs cp;
  float cv[kMaxStates];
  eval_fixed(R, o.child, d - 1, 1 - n.parity, n.C, cp, cv, ctx, layer + 1);
  dp = cp.inverted();
  for (int i = 0; i < n.S.n; ++i)
    dv[i] = R.flip(child_optimal(R, 1 - n.parity, cv, n.nd[i], n.dt[i], n.S.s[i].cube));
}

void Searcher::eval_fixed(const Root& R, const Board& P, int d, int parity, const StateSet& S, Probs& out_p,
                          float* out_v, Ctx& ctx, int layer) {
  nodes_.fetch_add(1, std::memory_order_relaxed);
  if (d <= 1) {
    out_p = ev_.evaluate(P);
    leaf_values(R, P, out_p, parity, S, out_v);
    return;
  }
  const u64 key = mix64(P.hash() ^ R.salt ^ S.hash() ^ (u64(d) << 56) ^ (u64(parity) << 61) ^
                        (u64(layer >= R.cfg.full_layers) << 62));
  if (tt_->probe(key, S.n, out_p, out_v)) return;

  Node n;
  make_node(R, P, parity, S, n);
  const auto& rolls = all_rolls();
  RollOut ro[21];
  for (int r = 0; r < 21; ++r) static_roll(R, n, r, ro[r], ctx, layer);

  Probs acc_p = Probs::zero();
  float acc_v[kMaxStates] = {};
  if (d == 2) {
    for (int r = 0; r < 21; ++r) {
      acc_p += ro[r].sp * rolls[r].w;
      for (int i = 0; i < S.n; ++i) acc_v[i] += rolls[r].w * ro[r].sv[i];
    }
  } else if (layer < R.cfg.full_layers || R.cfg.sampled_rolls >= 21) {
    for (int r = 0; r < 21; ++r) {
      Probs dp;
      float dv[kMaxStates];
      deep_roll(R, n, ro[r], d, dp, dv, ctx, layer);
      acc_p += dp * rolls[r].w;
      for (int i = 0; i < S.n; ++i) acc_v[i] += rolls[r].w * dv[i];
    }
  } else {
    // Control variate: every roll contributes its static value; one representative
    // per group of similar rolls (sorted by static value, equal probability mass)
    // is searched deeply and its deep-minus-static correction is applied to the
    // whole group.
    int order[21], m = 0;
    for (int r = 0; r < 21; ++r) {
      acc_p += ro[r].sp * rolls[r].w;
      for (int i = 0; i < S.n; ++i) acc_v[i] += rolls[r].w * ro[r].sv[i];
      if (!ro[r].terminal) order[m++] = r;
    }
    std::sort(order, order + m, [&](int a, int b) { return ro[a].sv[0] < ro[b].sv[0]; });
    float total = 0;
    for (int i = 0; i < m; ++i) total += rolls[order[i]].w;
    const int k = std::min(R.cfg.sampled_rolls, m);
    int start = 0;
    float cum = 0;
    for (int g = 0; g < k && start < m; ++g) {
      const float target = total * float(g + 1) / float(k);
      int end = start;
      float gw = 0;
      while (end < m && (end == start || g == k - 1 || cum + gw + rolls[order[end]].w <= target + 1e-6f)) {
        gw += rolls[order[end]].w;
        ++end;
      }
      cum += gw;
      const int rep = order[(start + end - 1) / 2];
      Probs dp;
      float dv[kMaxStates];
      deep_roll(R, n, ro[rep], d, dp, dv, ctx, layer);
      for (int q = 0; q < 5; ++q) acc_p[q] += gw * (dp[q] - ro[rep].sp[q]);
      for (int i = 0; i < S.n; ++i) acc_v[i] += gw * (dv[i] - ro[rep].sv[i]);
      start = end;
    }
  }
  out_p = acc_p;
  out_p.sanitize();
  for (int i = 0; i < S.n; ++i) out_v[i] = acc_v[i];
  if (!stop_.load(std::memory_order_relaxed)) tt_->store(key, S.n, out_p, out_v);
}

// Evaluates several nodes at depth d in parallel over (node, first roll) pairs.
void Searcher::eval_nodes_parallel(const Root& R, const std::vector<Board>& boards, int d, int parity,
                                   const std::vector<StateSet>& states, std::vector<Probs>& out_p,
                                   std::vector<std::array<float, kMaxStates>>& out_v) {
  const int n = int(boards.size());
  out_p.assign(n, Probs::zero());
  out_v.assign(n, {});
  if (d <= 2) {
    pool_.parallel_for(n, [&](int i, int) {
      Ctx ctx;
      eval_fixed(R, boards[i], d, parity, states[i], out_p[i], out_v[i].data(), ctx, 0);
    });
    return;
  }
  std::vector<Node> nodes(n);
  for (int i = 0; i < n; ++i) make_node(R, boards[i], parity, states[i], nodes[i]);
  struct Part { Probs p; float v[kMaxStates]; };
  std::vector<Part> parts(size_t(n) * 21);
  pool_.parallel_for(n * 21, [&](int task, int) {
    const int i = task / 21, r = task % 21;
    Ctx ctx;
    RollOut o;
    static_roll(R, nodes[i], r, o, ctx, 0);
    deep_roll(R, nodes[i], o, d, parts[task].p, parts[task].v, ctx, 0);
  });
  const auto& rolls = all_rolls();
  for (int i = 0; i < n; ++i) {
    Probs p = Probs::zero();
    for (int r = 0; r < 21; ++r) {
      const Part& pt = parts[size_t(i) * 21 + r];
      p += pt.p * rolls[r].w;
      for (int s = 0; s < states[i].n; ++s) out_v[i][s] += rolls[r].w * pt.v[s];
    }
    p.sanitize();
    out_p[i] = p;
  }
}

MoveAnalysis Searcher::analyze_moves(const Board& b, int d1, int d2, const MatchState& ms, const SearchConfig& cfg,
                                     const Board* must_include) {
  const auto t0 = std::chrono::steady_clock::now();
  stop_.store(false);
  nodes_.store(0);
  const Root R(ms, cfg);
  MoveAnalysis res;
  std::vector<Move> moves;
  generate_moves(b, d1, d2, moves);
  res.moves.resize(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) res.moves[i].move = moves[i];

  StateSet S0;
  S0.add(R.s0);
  // Child (opponent-to-move) states shared by all candidates.
  Node rootn;
  make_node(R, b, 0, S0, rootn);
  const StateSet& C = rootn.C;
  const int nd0 = rootn.nd[0], dt0 = rootn.dt[0];

  auto evaluate_at = [&](int count, int d) {
    std::vector<Board> boards;
    std::vector<int> idx;
    for (int i = 0; i < count; ++i) {
      MoveEval& me = res.moves[i];
      me.plies = d;
      if (me.move.after.side[0] == 0) {
        float v[kMaxStates];
        terminal_values(R, me.move.after, 0, S0, me.probs, v);
        me.value = v[0];
        continue;
      }
      boards.push_back(me.move.after.swapped());
      idx.push_back(i);
    }
    std::vector<StateSet> states(boards.size(), C);
    std::vector<Probs> ps;
    std::vector<std::array<float, kMaxStates>> vs;
    if (d == 1) {
      ps.resize(boards.size());
      vs.resize(boards.size());
      for (size_t k = 0; k < boards.size(); ++k) {
        ps[k] = ev_.evaluate(boards[k]);
        leaf_values(R, boards[k], ps[k], 1, C, vs[k].data());
      }
    } else {
      eval_nodes_parallel(R, boards, d, 1, states, ps, vs);
    }
    for (size_t k = 0; k < boards.size(); ++k) {
      MoveEval& me = res.moves[idx[k]];
      me.probs = ps[k].inverted();
      me.value = R.flip(child_optimal(R, 1, vs[k].data(), nd0, dt0, R.s0.cube));
    }
  };
  auto sort_moves = [&]() {
    std::stable_sort(res.moves.begin(), res.moves.end(), [](const MoveEval& a, const MoveEval& b) {
      if (a.plies != b.plies) return a.plies > b.plies;
      return a.value > b.value;
    });
  };

  evaluate_at(int(res.moves.size()), 1);
  sort_moves();
  for (int d = 2; d <= cfg.plies && !stop_.load(); ++d) {
    const float best = R.cm[0].to_emg(res.moves[0].value);
    int keep = std::max(1, cfg.keep[std::min(d - 1, 7)]);
    if (cfg.max_candidates > 0) keep = std::min(keep, cfg.max_candidates);
    const float width = cfg.width[std::min(d - 1, 7)];
    int n = 1;
    while (n < int(res.moves.size()) && n < keep && res.moves[n].plies == d - 1 &&
           best - R.cm[0].to_emg(res.moves[n].value) <= width)
      ++n;
    if (must_include) {
      // Keep the requested play among the candidates (move it into slot n).
      for (int i = n; i < int(res.moves.size()); ++i)
        if (res.moves[i].move.after == *must_include && res.moves[i].plies == d - 1) {
          std::swap(res.moves[i], res.moves[n]);
          ++n;
          break;
        }
    }
    evaluate_at(n, d);
    sort_moves();
  }
  for (auto& m : res.moves) m.emg = R.cm[0].to_emg(m.value);
  res.nodes = nodes_.load();
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

CubeResult Searcher::analyze_cube(const Board& b, const MatchState& ms, const SearchConfig& cfg) {
  const auto t0 = std::chrono::steady_clock::now();
  stop_.store(false);
  nodes_.store(0);
  const Root R(ms, cfg);
  CubeResult res;
  StateSet S;
  S.add(R.s0);
  const bool can_double = ms.owner != CubeOwner::Opp && R.cm[0].can_double(0, ms.cube);
  int dt_idx = -1;
  if (can_double) dt_idx = S.add(CubeState{int16_t(2 * ms.cube), CubeOwner::Opp});
  Probs p;
  float v[kMaxStates];
  if (cfg.plies <= 1) {
    p = ev_.evaluate(b);
    leaf_values(R, b, p, 0, S, v);
  } else {
    std::vector<Probs> ps;
    std::vector<std::array<float, kMaxStates>> vs;
    eval_nodes_parallel(R, {b}, cfg.plies, 0, {S}, ps, vs);
    p = ps[0];
    std::copy(vs[0].begin(), vs[0].end(), v);
  }
  CubeAnalysis& a = res.analysis;
  // Start from the static analysis for the bookkeeping fields, then use the
  // searched values.
  float xs[3];
  ev_.cube_x(b, xs);  // learned cube efficiency for (centered, owned, opponent owns)
  a = bg::analyze_cube(p, ms, xs[ms.owner == CubeOwner::Center ? 0 : (ms.owner == CubeOwner::Me ? 1 : 2)]);
  a.nd = v[0];
  if (dt_idx >= 0) a.dt = v[dt_idx];
  a.dp = R.cm[0].value(ms.cube);
  a.take = a.dt <= a.dp;
  a.beaver = ms.money() && ms.beavers && a.take && a.dt < 0.f;
  a.can_double = can_double;
  if (!can_double) {
    a.decision = CubeDecision::NotAvailable;
  } else if (std::min(a.dt, a.dp) > a.nd) {
    a.decision = a.take ? CubeDecision::DoubleTake : CubeDecision::DoublePass;
  } else if (a.nd > a.dp) {
    a.decision = a.take ? CubeDecision::TooGoodTake : CubeDecision::TooGoodPass;
  } else {
    a.decision = a.beaver ? CubeDecision::NoDoubleBeaver : CubeDecision::NoDouble;
  }
  res.nd_emg = R.cm[0].to_emg(a.nd);
  res.dt_emg = R.cm[0].to_emg(a.dt);
  res.dp_emg = R.cm[0].to_emg(a.dp);
  res.plies = cfg.plies;
  res.nodes = nodes_.load();
  res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return res;
}

Probs Searcher::evaluate(const Board& b, const MatchState& ms, int plies) {
  SearchConfig cfg = SearchConfig::for_plies(plies);
  cfg.cubeful = false;
  const Root R(ms, cfg);
  StateSet S;
  S.add(R.s0);
  if (plies <= 1) return ev_.evaluate(b);
  std::vector<Probs> ps;
  std::vector<std::array<float, kMaxStates>> vs;
  eval_nodes_parallel(R, {b}, plies, 0, {S}, ps, vs);
  return ps[0];
}

}  // namespace bg
