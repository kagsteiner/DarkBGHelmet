#include "rollout.hpp"

#include <chrono>
#include <cmath>
#include <vector>

#include "core/movegen.hpp"

namespace bg {

namespace {

inline u64 splitmix(u64& s) {
  u64 z = (s += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

inline CubeOwner flip_owner(CubeOwner o) {
  return o == CubeOwner::Me ? CubeOwner::Opp : (o == CubeOwner::Opp ? CubeOwner::Me : CubeOwner::Center);
}

// Probability-vector difference seen by the other player.
inline Probs flip_diff(const Probs& d) {
  Probs q;
  q.p = {-d[kWin], d[kLoseG], d[kLoseBG], d[kWinG], d[kWinBG]};
  return q;
}

struct Acc {
  double p[5] = {}, cl = 0, cl2 = 0, cf = 0, cf2 = 0;
  long n = 0;
};

}  // namespace

RolloutTrial rollout_trial(const Evaluator& ev, const Board& root, const MatchState& ms, const RolloutConfig& cfg,
                           int t) {
  const CubeModel cm[2] = {CubeModel(ms), CubeModel(ms.swapped())};
  const bool money = ms.money();
  // Value seen by the root player of a value seen by the player at `parity`.
  auto to_root_v = [&](float v, int parity) { return parity == 0 ? v : (money ? -v : 1.f - v); };
  auto to_root_dv = [&](float dv, int parity) { return parity == 0 ? dv : -dv; };
  // Learned cube efficiency for a cube state (relative to the player to move at pos).
  auto cube_x = [&](const Board& pos, CubeOwner own) {
    float xs[3];
    ev.cube_x(pos, xs);
    return xs[own == CubeOwner::Center ? 0 : (own == CubeOwner::Me ? 1 : 2)];
  };

  u64 rng = cfg.seed * 0x9E3779B97F4A7C15ULL + u64(t) * 0xD1B54A32D192ED03ULL;
  Board pos = root;
  int parity = 0;
  int cube = ms.cube;
  CubeOwner owner_root = ms.owner;  // relative to the root player
  bool turned = false;
  Probs luck_p = Probs::zero();
  float luck_v = 0;
  Probs res_p;
  float res_v = 0;
  std::vector<Move> moves;
  std::vector<Probs> probs;

  for (int ply = 0;;) {  // ply counts rolls; a cube action does not advance it
    const CubeModel& m = cm[parity];
    const CubeModel& om = cm[1 - parity];
    const CubeOwner own = parity == 0 ? owner_root : flip_owner(owner_root);

    // Cube action of the player to move (on the root player's first turn only if allowed).
    if (cfg.cubeful && cfg.cube_actions && (ply > 0 || cfg.root_may_double) && own != CubeOwner::Opp &&
        m.can_double(0, cube)) {
      const Probs sp = ev.evaluate(pos);
      float xs[3];
      ev.cube_x(pos, xs);
      const float nd = m.cubeful(sp, cube, own, xs[own == CubeOwner::Center ? 0 : 1]);
      const float dt = m.cubeful(sp, 2 * cube, CubeOwner::Opp, xs[2]);
      const float dp = m.value(cube);
      if (std::min(dt, dp) > nd) {
        if (dt > dp) {  // double, pass
          res_p = parity == 0 ? sp : sp.inverted();
          res_v = to_root_v(dp, parity);
          break;
        }
        cube *= 2;
        turned = true;
        owner_root = parity == 0 ? CubeOwner::Opp : CubeOwner::Me;
        continue;  // the same player now rolls with the new cube state
      }
    }
    const CubeOwner own2 = parity == 0 ? owner_root : flip_owner(owner_root);
    const bool jac2 = money && ms.jacoby && !turned && own2 == CubeOwner::Center;

    if (cfg.truncate > 0 && ply >= cfg.truncate) {
      const Probs sp = ev.evaluate(pos);
      res_p = parity == 0 ? sp : sp.inverted();
      res_v = to_root_v(cfg.cubeful ? m.cubeful(sp, cube, own2, cube_x(pos, own2)) : m.dead(sp, cube, jac2), parity);
      break;
    }

    // Roll: stratified for the first two plies.
    int d1, d2;
    if (ply < 2) {
      int k = ply == 0 ? t % 36 : (t / 36) % 36;
      d1 = k / 6 + 1;
      d2 = k % 6 + 1;
    } else {
      u64 r = splitmix(rng);
      d1 = int(r % 6) + 1;
      d2 = int((r >> 8) % 6) + 1;
    }

    // Move-choice utility: dead value at the current cube, mover's view.
    const float W1 = m.value(cube), W2 = m.value(2 * cube), W3 = m.value(3 * cube);
    const float L1 = m.value(-cube), L2 = m.value(-2 * cube), L3 = m.value(-3 * cube);
    auto utility = [&](const Probs& q) {
      const float w = q[kWin], l = 1.f - w;
      return (w - q[kWinG]) * W1 + (q[kWinG] - q[kWinBG]) * W2 + q[kWinBG] * W3 + (l - q[kLoseG]) * L1 +
             (q[kLoseG] - q[kLoseBG]) * L2 + q[kLoseBG] * L3;
    };
    // Value of a play for the mover: the opponent then has its cube options.
    const CubeOwner opp_own = flip_owner(own2);
    auto play_value = [&](const Move& mv, const Probs& q) -> float {
      if (mv.after.side[0] == 0) {
        int k = q[kWinBG] > 0.5f ? 3 : (q[kWinG] > 0.5f ? 2 : 1);
        return m.value((jac2 ? 1 : k) * cube);
      }
      if (!cfg.cubeful) return m.dead(q, cube, jac2);
      const Probs oq = q.inverted();
      const Board ob = mv.after.swapped();
      float xs[3];
      ev.cube_x(ob, xs);
      float v = om.cubeful(oq, cube, opp_own, xs[opp_own == CubeOwner::Center ? 0 : (opp_own == CubeOwner::Me ? 1 : 2)]);
      if (opp_own != CubeOwner::Opp && om.can_double(0, cube)) {
        const float dt = om.cubeful(oq, 2 * cube, CubeOwner::Opp, xs[2]), dp = om.value(cube);
        if (dt >= v && dp >= v) v = std::min(dt, dp);
      }
      return money ? -v : 1.f - v;
    };
    auto best_of = [&](int a, int b, Move& best, Probs& bp) {
      generate_moves(pos, a, b, moves);
      ev.evaluate_moves(pos, moves, probs);
      int bi = 0;
      float bu = -1e30f;
      for (size_t i = 0; i < moves.size(); ++i) {
        float u = utility(probs[i]);
        if (u > bu) { bu = u; bi = int(i); }
      }
      best = moves[bi];
      bp = probs[bi];
    };

    Move best;
    Probs bp;
    if (cfg.variance_reduction) {
      Probs mean_p = Probs::zero();
      float mean_v = 0, act_v = 0;
      Probs act_p;
      Move act_m;
      for (int a = 1; a <= 6; ++a)
        for (int b = a; b <= 6; ++b) {
          Move mv;
          Probs q;
          best_of(a, b, mv, q);
          const float w = (a == b ? 1.f : 2.f) / 36.f;
          const float v = play_value(mv, q);
          mean_p += q * w;
          mean_v += w * v;
          if ((a == d1 && b == d2) || (a == d2 && b == d1)) { act_v = v; act_p = q; act_m = mv; }
        }
      best = act_m;
      bp = act_p;
      Probs dp = act_p;
      for (int i = 0; i < 5; ++i) dp[i] -= mean_p[i];
      luck_p += parity == 0 ? dp : flip_diff(dp);
      luck_v += to_root_dv(act_v - mean_v, parity);
    } else {
      best_of(d1, d2, best, bp);
    }

    if (best.after.side[0] == 0) {  // mover has borne off all checkers
      const Probs tp = Evaluator::terminal(best.after);
      res_p = parity == 0 ? tp : tp.inverted();
      const int k = tp[kWinBG] > 0.5f ? 3 : (tp[kWinG] > 0.5f ? 2 : 1);
      res_v = to_root_v(cfg.cubeful ? m.value((jac2 ? 1 : k) * cube) : m.dead(tp, cube, jac2), parity);
      break;
    }
    pos = best.after.swapped();
    parity ^= 1;
    ++ply;
  }

  RolloutTrial out;
  out.probs = res_p;
  for (int i = 0; i < 5; ++i) out.probs[i] -= luck_p[i];
  const bool jac_root = money && ms.jacoby && ms.owner == CubeOwner::Center;
  out.cubeless = cm[0].dead(out.probs, ms.cube, jac_root);
  out.cubeful = cfg.cubeful ? res_v - luck_v : out.cubeless;
  return out;
}

RolloutResult rollout(const Evaluator& ev, const Board& root, const MatchState& ms, const RolloutConfig& cfg,
                      ThreadPool& pool) {
  const auto t0 = std::chrono::steady_clock::now();
  const int T = std::max(1, cfg.trials);
  std::vector<Acc> acc(pool.size());
  pool.parallel_for(T, [&](int t, int worker) {
    const RolloutTrial r = rollout_trial(ev, root, ms, cfg, t);
    Acc& a = acc[worker];
    for (int i = 0; i < 5; ++i) a.p[i] += r.probs[i];
    a.cl += r.cubeless; a.cl2 += double(r.cubeless) * r.cubeless;
    a.cf += r.cubeful; a.cf2 += double(r.cubeful) * r.cubeful;
    a.n++;
  });

  Acc tot;
  for (const Acc& a : acc) {
    for (int i = 0; i < 5; ++i) tot.p[i] += a.p[i];
    tot.cl += a.cl; tot.cl2 += a.cl2; tot.cf += a.cf; tot.cf2 += a.cf2; tot.n += a.n;
  }
  RolloutResult r;
  const double n = double(tot.n);
  for (int i = 0; i < 5; ++i) r.probs[i] = float(tot.p[i] / n);
  r.probs.sanitize();
  r.cubeless = float(tot.cl / n);
  r.cubeful = float(tot.cf / n);
  auto se = [&](double s, double s2) { double var = std::max(0.0, s2 / n - (s / n) * (s / n)); return float(std::sqrt(var / n)); };
  r.se_cubeless = se(tot.cl, tot.cl2);
  r.se_cubeful = se(tot.cf, tot.cf2);
  r.trials = int(tot.n);
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return r;
}

}  // namespace bg
