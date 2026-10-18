#include "mcsearch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

namespace bg {

namespace {

using clk = std::chrono::steady_clock;

// One bandit arm: a candidate play (or a cube action) evaluated by rollouts.
struct Arm {
  Board root;           // rollout start position (player to move = side 0)
  MatchState ms;        // match state from that player's view
  bool flip = false;    // rollout values are from the opponent's view (checker play)
  bool exact = false;   // value known exactly (the play ends the game)
  float scale = 1.f;    // money: rollout values are normalised to the arm's own cube; rescale to the root cube
  float prior = 0;      // pre-ranking value, decision maker's view
  double sum = 0, sumsq = 0;
  long n = 0;
  int next_t = 0;       // next trial index (common random numbers across arms)
  Probs psum = Probs::zero();
  std::vector<float> vals;  // per-game values by trial index (for paired comparisons)
};

float flip_value(float v, bool money) { return money ? -v : 1.f - v; }

struct Bandit {
  std::vector<Arm> arms;
  float prior_games;
  bool money;
  float sigma0;

  // Pooled per-game standard deviation (around each arm's own mean).
  float sigma() const {
    double ss = 0;
    long dof = 0;
    for (const Arm& a : arms) {
      if (a.exact || a.n < 2) continue;
      ss += a.sumsq - a.sum * a.sum / double(a.n);
      dof += a.n - 1;
    }
    if (dof < 20) return sigma0;
    return std::max(float(std::sqrt(ss / double(dof))), 0.1f * sigma0);
  }
  float mean(const Arm& a) const {
    if (a.exact) return a.prior;
    return float((prior_games * a.prior + a.sum) / (prior_games + double(a.n)));
  }
  float sd(const Arm& a, float sig) const {
    if (a.exact) return 0.f;
    return sig / std::sqrt(prior_games + float(a.n));
  }
};

// Rolls out `count` more games of arm `a` (trial indices continue from next_t).
void run_batch(const Evaluator& ev, ThreadPool& pool, const RolloutConfig& rc, Arm& a, int count, bool money) {
  if (a.exact || count <= 0) return;
  std::vector<RolloutTrial> res(count);
  const int t0 = a.next_t;
  pool.parallel_for(count, [&](int i, int) { res[i] = rollout_trial(ev, a.root, a.ms, rc, t0 + i); });
  for (const RolloutTrial& r : res) {
    const float raw = money ? r.cubeful * a.scale : r.cubeful;
    const float v = a.flip ? flip_value(raw, money) : raw;
    a.sum += v;
    a.sumsq += double(v) * v;
    a.vals.push_back(v);
    a.psum += a.flip ? r.probs.inverted() : r.probs;
  }
  a.n += count;
  a.next_t += count;
}

void decide(CubeAnalysis& a, const MatchState& ms) {
  a.take = a.dt <= a.dp;
  a.beaver = ms.money() && ms.beavers && a.take && a.dt < 0.f;
  if (!a.can_double) a.decision = CubeDecision::NotAvailable;
  else if (std::min(a.dt, a.dp) > a.nd) a.decision = a.take ? CubeDecision::DoubleTake : CubeDecision::DoublePass;
  else if (a.nd > a.dp) a.decision = a.take ? CubeDecision::TooGoodTake : CubeDecision::TooGoodPass;
  else a.decision = a.beaver ? CubeDecision::NoDoubleBeaver : CubeDecision::NoDouble;
}

}  // namespace

MoveAnalysis mc_analyze_moves(Searcher& s, const Evaluator& ev, const Board& b, int d1, int d2, const MatchState& ms,
                              const McConfig& cfg, McStats* stats) {
  const auto t0 = clk::now();
  const auto deadline = t0 + std::chrono::milliseconds(std::max(1, cfg.time_ms));
  McStats st;
  // Pre-ranking with wide move filters, so close alternatives reach the full
  // pre-ranking depth (the static 1-ply ranking can be far off).
  SearchConfig pre = SearchConfig::for_plies(std::max(1, cfg.prefilter_plies));
  for (int d = 1; d < 8; ++d) {
    pre.keep[d] = std::max(pre.keep[d], 2 * cfg.max_candidates);
    pre.width[d] = std::max(pre.width[d], 2.f * cfg.window);
  }
  MoveAnalysis base = s.analyze_moves(b, d1, d2, ms, pre);
  const CubeModel cm(ms);
  const bool money = ms.money();

  // Candidates: plays at the pre-ranking depth within the window of the best.
  std::vector<int> cand;
  const float best_emg = base.moves.empty() ? 0.f : base.moves[0].emg;
  for (int i = 0; i < int(base.moves.size()) && int(cand.size()) < cfg.max_candidates; ++i)
    if (base.moves[i].plies == base.moves[0].plies && best_emg - base.moves[i].emg <= cfg.window) cand.push_back(i);
  st.candidates = int(cand.size());
  if (cand.size() < 2) {  // forced or clear-cut: the pre-ranking decides
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    if (stats) *stats = st;
    base.seconds = st.seconds;
    return base;
  }

  Bandit bd;
  bd.prior_games = std::max(0.f, cfg.prior_games);
  bd.money = money;
  bd.sigma0 = money ? 1.2f : 0.25f;
  for (int i : cand) {
    Arm a;
    const MoveEval& me = base.moves[i];
    a.prior = me.value;
    a.exact = me.move.after.side[0] == 0;
    a.root = me.move.after.swapped();
    a.ms = ms.swapped();
    a.flip = true;
    bd.arms.push_back(a);
  }
  RolloutConfig rc;
  rc.truncate = cfg.truncate;
  rc.cubeful = true;
  rc.root_may_double = true;  // the opponent may double before its first roll
  rc.cube_actions = cfg.rollout_cube_actions;
  rc.variance_reduction = cfg.variance_reduction;
  rc.seed = cfg.seed;
  ThreadPool& pool = s.pool();
  const int batch = std::max(1, cfg.batch);

  // One batch per arm first (common dice prefix).
  for (Arm& a : bd.arms) {
    if (clk::now() >= deadline) break;
    run_batch(ev, pool, rc, a, batch, money);
    st.batches++;
  }
  std::mt19937_64 rng(cfg.seed ^ b.hash());
  std::normal_distribution<float> nd01(0.f, 1.f);
  const int na = int(bd.arms.size());
  std::vector<float> final_value(na);

  if (cfg.paired) {
    // Paired Thompson sampling against the current leader.
    const float k = std::max(1.f, cfg.prior_games);
    const float sigma_d0 = money ? 0.5f : 0.12f;
    auto val = [&](int i, int t) { const Arm& a = bd.arms[i]; return a.exact ? a.prior : a.vals[t]; };
    auto count = [&](int i) { return bd.arms[i].exact ? (1 << 30) : int(bd.arms[i].n); };
    // Posterior of (arm i - arm L): mean and standard deviation; also returns the paired sum of squares.
    struct Diff { float mu = 0, sd = 0; double ss = 0; int m = 0; };
    auto diff = [&](int i, int L, float sigma_d) {
      Diff d;
      const int m = std::min(count(i), count(L));
      d.m = (bd.arms[i].exact && bd.arms[L].exact) ? 0 : m;
      double sum = 0, sq = 0;
      for (int t = 0; t < d.m; ++t) {
        const double x = val(i, t) - val(L, t);
        sum += x;
        sq += x * x;
      }
      const double mean = d.m ? sum / d.m : 0.0;
      d.ss = d.m ? sq - sum * mean : 0.0;
      const float pd = bd.arms[i].prior - bd.arms[L].prior;
      if (bd.arms[i].exact && bd.arms[L].exact) { d.mu = pd; d.sd = 0; return d; }
      d.mu = float((k * pd + d.m * mean) / (k + d.m));
      d.sd = sigma_d / std::sqrt(k + float(d.m));
      return d;
    };
    int L = 0;  // the pre-ranking's best play starts as leader
    auto pooled_sigma = [&]() {
      double ss = 0;
      long dof = 0;
      for (int i = 0; i < na; ++i) {
        if (i == L) continue;
        Diff d = diff(i, L, sigma_d0);
        if (d.m >= 2) { ss += d.ss; dof += d.m - 1; }
      }
      return dof >= 20 ? std::max(float(std::sqrt(ss / double(dof))), 0.05f * sigma_d0) : sigma_d0;
    };
    while (clk::now() < deadline) {
      const float sig = pooled_sigma();
      int c = -1;
      float best = -1e30f;
      for (int i = 0; i < na; ++i) {
        if (i == L || bd.arms[i].exact) continue;
        Diff d = diff(i, L, sig);
        const float sample = d.mu + d.sd * nd01(rng);
        if (sample > best) { best = sample; c = i; }
      }
      if (c < 0) {  // only exact challengers left: refine the leader itself if possible
        if (bd.arms[L].exact) break;
        c = L;
      }
      run_batch(ev, pool, rc, bd.arms[c], batch, money);
      st.batches++;
      if (c != L && !bd.arms[L].exact && bd.arms[L].n < bd.arms[c].n) {
        run_batch(ev, pool, rc, bd.arms[L], int(bd.arms[c].n - bd.arms[L].n), money);
        st.batches++;
      }
      // Change of leader if a challenger is now estimated to be better.
      int nl = L;
      float top = 0.f;
      for (int i = 0; i < na; ++i) {
        if (i == L) continue;
        Diff d = diff(i, L, sig);
        if (d.mu > top) { top = d.mu; nl = i; }
      }
      L = nl;
    }
    // Absolute values: the leader's own estimate plus each arm's paired difference.
    const Arm& la = bd.arms[L];
    const float leader_value =
        la.exact ? la.prior : float((k * la.prior + la.sum) / (k + double(la.n)));
    const float sig = pooled_sigma();
    for (int i = 0; i < na; ++i) final_value[i] = i == L ? leader_value : leader_value + diff(i, L, sig).mu;
  } else {
    // Independent Thompson sampling (one posterior per arm).
    while (clk::now() < deadline) {
      const float sig = bd.sigma();
      int pick = -1;
      float best = -1e30f;
      for (int i = 0; i < na; ++i) {
        const Arm& a = bd.arms[i];
        const float sample = bd.mean(a) + bd.sd(a, sig) * nd01(rng);
        if (sample > best) { best = sample; pick = i; }
      }
      if (pick < 0 || bd.arms[pick].exact) {
        int rival = -1;
        float rv = -1e30f;
        for (int i = 0; i < na; ++i)
          if (!bd.arms[i].exact && bd.mean(bd.arms[i]) > rv) { rv = bd.mean(bd.arms[i]); rival = i; }
        if (rival < 0) break;
        pick = rival;
      }
      run_batch(ev, pool, rc, bd.arms[pick], batch, money);
      st.batches++;
    }
    for (int i = 0; i < na; ++i) final_value[i] = bd.mean(bd.arms[i]);
  }

  // Write the posterior values back and re-rank the candidates.
  std::vector<MoveEval> out;
  for (size_t k = 0; k < cand.size(); ++k) {
    MoveEval me = base.moves[cand[k]];
    const Arm& a = bd.arms[k];
    me.value = final_value[k];
    me.emg = cm.to_emg(me.value);
    me.rollouts = int(a.n);
    if (a.n > 0) {
      Probs p = Probs::zero();
      const float w0 = bd.prior_games, wn = float(a.n);
      for (int q = 0; q < 5; ++q) p[q] = (w0 * me.probs[q] + a.psum[q]) / (w0 + wn);
      p.sanitize();
      me.probs = p;
    }
    st.games += a.n;
    out.push_back(me);
  }
  std::stable_sort(out.begin(), out.end(), [](const MoveEval& x, const MoveEval& y) { return x.value > y.value; });
  for (int i = 0; i < int(base.moves.size()); ++i)
    if (std::find(cand.begin(), cand.end(), i) == cand.end()) out.push_back(base.moves[i]);
  base.moves = out;
  st.used_rollouts = true;
  st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
  base.seconds = st.seconds;
  if (stats) *stats = st;
  return base;
}

CubeResult mc_analyze_cube(Searcher& s, const Evaluator& ev, const Board& b, const MatchState& ms, const McConfig& cfg,
                           McStats* stats) {
  const auto t0 = clk::now();
  McStats st;
  CubeResult base = s.analyze_cube(b, ms, SearchConfig::for_plies(std::max(1, cfg.prefilter_plies)));
  CubeAnalysis& a = base.analysis;
  if (!a.can_double) {
    if (stats) *stats = st;
    return base;
  }
  const CubeModel cm(ms);
  const float nd = cm.to_emg(a.nd), dt = cm.to_emg(a.dt), dp = cm.to_emg(a.dp);
  const bool close_double = std::fabs(nd - std::min(dt, dp)) < cfg.cube_window;
  const bool close_take = std::fabs(dt - dp) < cfg.cube_window;
  if (!close_double && !close_take) {  // clear decision: the pre-ranking decides
    st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
    if (stats) *stats = st;
    return base;
  }
  const int budget = cfg.cube_time_ms >= 0 ? cfg.cube_time_ms : cfg.time_ms / 2;
  const auto deadline = t0 + std::chrono::milliseconds(std::max(1, budget));
  const bool money = ms.money();

  Bandit bd;
  bd.prior_games = std::max(0.f, cfg.prior_games);
  bd.money = money;
  bd.sigma0 = money ? 1.5f : 0.25f;
  Arm no_double, double_take;
  no_double.root = double_take.root = b;
  no_double.ms = ms;
  no_double.prior = a.nd;
  double_take.ms = ms;
  double_take.ms.cube = 2 * ms.cube;
  double_take.ms.owner = CubeOwner::Opp;
  double_take.prior = a.dt;
  double_take.scale = 2.f;
  bd.arms = {no_double, double_take};

  RolloutConfig rc;
  rc.truncate = cfg.truncate;
  rc.cubeful = true;
  rc.root_may_double = false;  // "no double" means: the player to move does not double now
  rc.cube_actions = cfg.rollout_cube_actions;
  rc.variance_reduction = cfg.variance_reduction;
  rc.seed = cfg.seed;
  // Both values are needed for the decision, so both arms get the same games
  // (common random numbers make their difference precise).
  const int batch = std::max(1, cfg.batch);
  while (clk::now() < deadline) {
    run_batch(ev, s.pool(), rc, bd.arms[0], batch, money);
    run_batch(ev, s.pool(), rc, bd.arms[1], batch, money);
    st.batches += 2;
  }
  a.nd = bd.mean(bd.arms[0]);
  a.dt = bd.mean(bd.arms[1]);
  decide(a, ms);
  base.nd_emg = cm.to_emg(a.nd);
  base.dt_emg = cm.to_emg(a.dt);
  base.dp_emg = cm.to_emg(a.dp);
  st.games = bd.arms[0].n + bd.arms[1].n;
  st.used_rollouts = true;
  st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
  base.seconds = st.seconds;
  if (stats) *stats = st;
  return base;
}

}  // namespace bg
