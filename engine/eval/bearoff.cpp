#include "bearoff.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>

#include "core/movegen.hpp"

namespace bg {

namespace {

// ways[k][r] = number of ways to place at most r checkers on k points = C(r+k, k).
struct Ways {
  int w[7][16];
  Ways() {
    for (int k = 0; k <= 6; ++k)
      for (int r = 0; r <= 15; ++r) {
        long long c = 1;
        for (int i = 1; i <= k; ++i) c = c * (r + i) / i;
        w[k][r] = int(c);
      }
  }
};
const Ways kWays;

}  // namespace

int BearoffDB::index(u128 side) {
  int idx = 0, rem = 15;
  for (int i = 0; i < 6; ++i) {
    int c = nibble(side, i);
    for (int v = 0; v < c; ++v) idx += kWays.w[5 - i][rem - v];
    rem -= c;
  }
  return idx;
}

float BearoffDB::mean_rolls(int idx) const {
  const float* d = off_dist(idx);
  float m = 0;
  for (int n = 0; n < kMaxRolls; ++n) m += n * d[n];
  return m;
}

void BearoffDB::compute() {
  off_.assign(size_t(kPositions) * kMaxRolls, 0.f);
  first_.assign(size_t(kPositions) * kMaxRolls, 0.f);
  // Enumerate all positions, then process in order of increasing pip count.
  std::vector<u128> pos;
  pos.reserve(kPositions);
  int c[6];
  for (c[0] = 0; c[0] <= 15; ++c[0])
    for (c[1] = 0; c[0] + c[1] <= 15; ++c[1])
      for (c[2] = 0; c[0] + c[1] + c[2] <= 15; ++c[2])
        for (c[3] = 0; c[0] + c[1] + c[2] + c[3] <= 15; ++c[3])
          for (c[4] = 0; c[0] + c[1] + c[2] + c[3] + c[4] <= 15; ++c[4])
            for (c[5] = 0; c[0] + c[1] + c[2] + c[3] + c[4] + c[5] <= 15; ++c[5]) {
              u128 s = 0;
              for (int i = 0; i < 6; ++i) s |= u128(c[i]) << (4 * i);
              pos.push_back(s);
            }
  auto pips = [](u128 s) { int p = 0; for (int i = 0; i < 6; ++i) p += nibble(s, i) * (i + 1); return p; };
  std::sort(pos.begin(), pos.end(), [&](u128 a, u128 b) { return pips(a) < pips(b); });
  std::vector<float> mean(kPositions, 0.f), mean_first(kPositions, 0.f);
  std::vector<Move> moves;
  for (u128 s : pos) {
    int idx = index(s);
    float* od = &off_[size_t(idx) * kMaxRolls];
    float* fd = &first_[size_t(idx) * kMaxRolls];
    int checkers = popcount_checkers(s);
    if (checkers == 0) {
      od[0] = 1.f;
      fd[0] = 1.f;
      continue;
    }
    Board b;
    b.side[0] = s;
    for (int d1 = 1; d1 <= 6; ++d1)
      for (int d2 = d1; d2 <= 6; ++d2) {
        float w = (d1 == d2 ? 1.f : 2.f) / 36.f;
        generate_moves(b, d1, d2, moves);
        int best = -1, best_f = -1;
        float bm = 1e9f, bf = 1e9f;
        for (const Move& m : moves) {
          int ci = index(m.after.side[0]);
          if (mean[ci] < bm) { bm = mean[ci]; best = ci; }
          if (checkers == 15) {
            float f = popcount_checkers(m.after.side[0]) < 15 ? 0.f : mean_first[ci];
            if (f < bf) { bf = f; best_f = popcount_checkers(m.after.side[0]) < 15 ? -2 : ci; }
          }
        }
        const float* cd = &off_[size_t(best) * kMaxRolls];
        for (int n = 0; n + 1 < kMaxRolls; ++n) od[n + 1] += w * cd[n];
        od[kMaxRolls - 1] += w * cd[kMaxRolls - 1];
        if (checkers == 15) {
          if (best_f == -2) fd[1] += w;
          else {
            const float* cf = &first_[size_t(best_f) * kMaxRolls];
            for (int n = 0; n + 1 < kMaxRolls; ++n) fd[n + 1] += w * cf[n];
          }
        }
      }
    if (checkers < 15) fd[0] = 1.f;
    float m = 0, mf = 0;
    for (int n = 0; n < kMaxRolls; ++n) { m += n * od[n]; mf += n * fd[n]; }
    mean[idx] = m;
    mean_first[idx] = mf;
  }
}

bool BearoffDB::init(const std::string& path) {
  if (ready()) return true;
  if (FILE* f = std::fopen(path.c_str(), "rb")) {
    off_.resize(size_t(kPositions) * kMaxRolls);
    first_.resize(size_t(kPositions) * kMaxRolls);
    bool ok = std::fread(off_.data(), sizeof(float), off_.size(), f) == off_.size() &&
              std::fread(first_.data(), sizeof(float), first_.size(), f) == first_.size();
    std::fclose(f);
    if (ok) return true;
    off_.clear();
    first_.clear();
  }
  compute();
  if (FILE* f = std::fopen(path.c_str(), "wb")) {
    std::fwrite(off_.data(), sizeof(float), off_.size(), f);
    std::fwrite(first_.data(), sizeof(float), first_.size(), f);
    std::fclose(f);
  }
  return true;
}

Probs BearoffDB::evaluate(const Board& b) const {
  const int ia = index(b.side[0]), ib = index(b.side[1]);
  const float* a = off_dist(ia);
  const float* bd = off_dist(ib);
  // Tail sums: P(opponent needs >= n rolls).
  float tail_b[kMaxRolls + 1], tail_bf[kMaxRolls + 1], tail_af[kMaxRolls + 1];
  tail_b[kMaxRolls] = tail_bf[kMaxRolls] = tail_af[kMaxRolls] = 0.f;
  const float* bf = first_dist(ib);
  const float* af = first_dist(ia);
  for (int n = kMaxRolls - 1; n >= 0; --n) {
    tail_b[n] = tail_b[n + 1] + bd[n];
    tail_bf[n] = tail_bf[n + 1] + bf[n];
    tail_af[n] = tail_af[n + 1] + af[n];
  }
  Probs p = Probs::zero();
  float win = 0, wg = 0, lg = 0;
  for (int n = 1; n < kMaxRolls; ++n) {
    win += a[n] * tail_b[n];
    wg += a[n] * tail_bf[n];
  }
  for (int m = 1; m < kMaxRolls; ++m) lg += bd[m] * tail_af[m + 1 < kMaxRolls + 1 ? m + 1 : kMaxRolls];
  p[kWin] = win;
  p[kWinG] = popcount_checkers(b.side[1]) == 15 ? wg : 0.f;
  p[kLoseG] = popcount_checkers(b.side[0]) == 15 ? lg : 0.f;
  p.sanitize();
  return p;
}

BearoffDB& bearoff_db() {
  static BearoffDB db;
  return db;
}

}  // namespace bg
