#include "features.hpp"

#include "core/movegen.hpp"

namespace bg {

int active_features(const Board& b, int p, int* out) {
  int n = 0;
  for (int block = 0; block < 2; ++block) {
    u128 x = b.side[block == 0 ? p : 1 - p];
    for (int s = 0; s < kSlots; ++s) {
      int c = nibble(x, s);
      for (int k = 1; k <= c; ++k) out[n++] = feature_index(block, s, k);
    }
    int off = kCheckers - popcount_checkers(x);
    for (int k = 1; k <= off; ++k) out[n++] = feature_index(block, 25, k);
  }
  return n;
}

namespace {

u32 occupied_mask(u128 x) {  // bits 0..23, bar excluded
  u32 m = 0;
  for (int s = 0; s < 24; ++s)
    if (nibble(x, s)) m |= 1u << s;
  return m;
}

u32 blot_mask_of_opp(u128 opp) {  // opponent blots in our coordinates
  u32 m = 0;
  for (int t = 0; t < 24; ++t)
    if (nibble(opp, 23 - t) == 1) m |= 1u << t;
  return m;
}

// Slots reachable by a single checker starting from any bit in `start` with one
// roll, given a set of blocked points. `start` may include bit 24 (bar).
u32 reach(u32 start, int d1, int d2, u32 blocked, int max_steps_double = 4) {
  const u32 open = ~blocked;
  if (d1 != d2) {
    u32 a = (start >> d1) & open;
    u32 b = (start >> d2) & open;
    return a | b | ((a >> d2) & open) | ((b >> d1) & open);
  }
  u32 r = 0, m = start;
  for (int i = 0; i < max_steps_double; ++i) {
    m = (m >> d1) & open;
    r |= m;
  }
  return r;
}

}  // namespace

float hit_chance(const Board& b) {
  const u128 me = b.side[0], opp = b.side[1];
  const u32 blots = blot_mask_of_opp(opp);
  if (!blots) return 0.f;
  const u32 blocked = blocked_mask(b);
  const u32 occ = occupied_mask(me);
  const int nb = nibble(me, kBar);
  const u32 bar = 1u << 24;
  int hits = 0;
  for (int d1 = 1; d1 <= 6; ++d1)
    for (int d2 = d1; d2 <= 6; ++d2) {
      bool hit = false;
      if (d1 != d2) {
        if (nb >= 2) {
          hit = (((bar >> d1) | (bar >> d2)) & ~blocked) & blots;
        } else if (nb == 1) {
          for (int o = 0; o < 2 && !hit; ++o) {
            int da = o ? d2 : d1, db = o ? d1 : d2;
            u32 e = (bar >> da) & ~blocked;
            if (!e) continue;
            if (e & blots) { hit = true; break; }
            u32 occ2 = occ | e;
            if (((occ2 >> db) & ~blocked) & blots) hit = true;
          }
        } else {
          hit = reach(occ, d1, d2, blocked) & blots;
        }
      } else {
        u32 e = (bar >> d1) & ~blocked;
        if (nb >= 1) {
          if (e) {
            if (e & blots) hit = true;
            int r = 4 - nb;
            if (!hit && r > 0) hit = reach(occ | e, d1, d1, blocked, r) & blots;
          }
        } else {
          hit = reach(occ, d1, d1, blocked) & blots;
        }
      }
      if (hit) hits += d1 == d2 ? 1 : 2;
    }
  return hits / 36.f;
}

float escape_chance(const Board& b) {
  if (!b.contact()) return 1.f;
  const int rear = highest_slot(b.side[0]);
  const u32 blocked = blocked_mask(b);
  const int lo_win = rear - 12 < 0 ? 0 : rear - 12;
  u32 window = 0;
  for (int s = lo_win; s < rear && s < 24; ++s) window |= 1u << s;
  u32 blockers = blocked & window;
  if (!blockers) return 1.f;
  const int lo = __builtin_ctz(blockers);
  const u32 past = (1u << lo) - 1;
  const u32 start = 1u << rear;
  int esc = 0;
  for (int d1 = 1; d1 <= 6; ++d1)
    for (int d2 = d1; d2 <= 6; ++d2)
      if (reach(start, d1, d2, blocked) & past) esc += d1 == d2 ? 1 : 2;
  return esc / 36.f;
}

void extra_features(const Board& b, float* out) {
  const int pm = b.pips(0), po = b.pips(1);
  const Board s = b.swapped();
  out[0] = pm / 100.f;
  out[1] = po / 100.f;
  out[2] = (po - pm) / 50.f;
  out[3] = hit_chance(b);
  out[4] = hit_chance(s);
  out[5] = escape_chance(b);
  out[6] = escape_chance(s);
  out[7] = b.contact() ? 1.f : 0.f;
}

PosClass classify(const Board& b) {
  if (b.side[0] == 0 || b.side[1] == 0) return PosClass::Over;
  if (!b.contact()) {
    if (((b.side[0] | b.side[1]) & kOutsideHome) == 0) return PosClass::Bearoff;
    return PosClass::Race;
  }
  for (int p = 0; p < 2; ++p) {
    const int tot = popcount_checkers(b.side[p]);
    const int b0 = b.count(p, 0), b1 = b.count(p, 1);
    constexpr int N = 6;
    if (tot <= N) return PosClass::Crashed;
    if (b0 > 1) {
      if (tot <= N + b0) return PosClass::Crashed;
      if (b1 > 1 && 1 + tot - (b0 + b1) <= N) return PosClass::Crashed;
    } else if (tot <= N + (b1 - 1)) {
      return PosClass::Crashed;
    }
  }
  return PosClass::Contact;
}

}  // namespace bg
