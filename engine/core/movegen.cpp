#include "movegen.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace bg {

u32 blocked_mask(const Board& b) {
  const u128 opp = b.side[1];
  u32 m = 0;
  for (int t = 0; t < 24; ++t)
    if (nibble(opp, 23 - t) >= 2) m |= 1u << t;
  return m;
}

namespace {

struct Gen {
  u32 blocked;
  std::vector<Move>* out;
  size_t base;
  int max_steps;
  int die_used_single;  // for one-step plays: the die that was used (larger wins)
  std::vector<uint8_t> single_die;  // parallel to out[base..]: die used when n == 1
  Move cur;

  bool can_step(u128 me, int s, int d) const {
    if (!nibble(me, s)) return false;
    if (s != kBar && nibble(me, kBar)) return false;
    int dst = s - d;
    if (dst >= 0) return !((blocked >> dst) & 1);
    if (me & kOutsideHome) return false;
    if (dst == -1) return true;
    return highest_slot(me) == s;
  }

  // Applies one step; returns true on a hit.
  static bool apply(u128& me, u128& opp, int s, int d) {
    me -= nibble_bit(s);
    int dst = s - d;
    if (dst < 0) return false;
    me += nibble_bit(dst);
    int t = 23 - dst;
    if (nibble(opp, t) == 1) {
      opp -= nibble_bit(t);
      opp += nibble_bit(kBar);
      return true;
    }
    return false;
  }

  void record(u128 me, u128 opp, int k, int die) {
    if (k < max_steps) return;
    if (k > max_steps) {
      out->resize(base);
      single_die.clear();
      max_steps = k;
    }
    cur.after.side[0] = me;
    cur.after.side[1] = opp;
    cur.n = uint8_t(k);
    out->push_back(cur);
    single_die.push_back(uint8_t(die));
  }

  // Doubles: up to four steps of die d, sources in non-increasing order (this
  // canonical order reaches every distinct play: moving higher checkers first
  // never makes a later step illegal).
  void rec_double(u128 me, u128 opp, int k, int d, int max_src) {
    bool any = false;
    if (k < 4) {
      for (int s = max_src; s >= 0; --s) {
        if (!can_step(me, s, d)) continue;
        any = true;
        u128 m2 = me, o2 = opp;
        bool hit = apply(m2, o2, s, d);
        cur.from[k] = int8_t(s);
        cur.to[k] = int8_t(s - d < 0 ? -1 : s - d);
        cur.hits = uint8_t(hit ? (cur.hits | (1 << k)) : (cur.hits & ~(1 << k)));
        rec_double(m2, o2, k + 1, d, s);
      }
    }
    if (!any) record(me, opp, k, d);
  }

  void rec_pair(u128 me, u128 opp, int k, int d_first, int d_second) {
    // k = 0: play d_first then d_second. k = 1: play d_second.
    int d = k == 0 ? d_first : d_second;
    bool any = false;
    if (k < 2) {
      for (int s = kBar; s >= 0; --s) {
        if (!can_step(me, s, d)) continue;
        any = true;
        u128 m2 = me, o2 = opp;
        bool hit = apply(m2, o2, s, d);
        cur.from[k] = int8_t(s);
        cur.to[k] = int8_t(s - d < 0 ? -1 : s - d);
        cur.hits = uint8_t(hit ? (cur.hits | (1 << k)) : (cur.hits & ~(1 << k)));
        rec_pair(m2, o2, k + 1, d_first, d_second);
      }
    }
    if (!any) record(me, opp, k, k == 1 ? d_first : 0);
  }
};

// Per-thread dedup table with generation stamps so it never needs clearing.
struct DedupTable {
  static constexpr int kSize = 1 << 13;
  u64 key[kSize];
  u32 stamp[kSize];
  int idx[kSize];
  u32 gen = 0;
  DedupTable() { std::memset(stamp, 0, sizeof stamp); }
};

}  // namespace

void generate_moves(const Board& b, int d1, int d2, std::vector<Move>& out) {
  out.clear();
  Gen g;
  g.blocked = blocked_mask(b);
  g.out = &out;
  g.base = 0;
  g.max_steps = 0;
  g.cur = Move{};
  g.single_die.clear();
  const u128 me = b.side[0], opp = b.side[1];
  if (d1 == d2) {
    g.rec_double(me, opp, 0, d1, kBar);
  } else {
    g.rec_pair(me, opp, 0, d1, d2);
    g.rec_pair(me, opp, 0, d2, d1);
    if (g.max_steps == 1) {
      // Only one die playable: the larger one must be played if possible.
      int hi = std::max(d1, d2);
      bool hi_possible = false;
      for (size_t i = 0; i < out.size(); ++i) hi_possible |= g.single_die[i] == hi;
      if (hi_possible) {
        size_t w = 0;
        for (size_t i = 0; i < out.size(); ++i)
          if (g.single_die[i] == hi) out[w++] = out[i];
        out.resize(w);
      }
    }
  }
  if (out.empty()) {  // no legal play: the position is unchanged
    Move m{};
    m.after = b;
    out.push_back(m);
    return;
  }
  if (out.size() == 1) return;

  static thread_local DedupTable t;
  if (++t.gen == 0) { std::memset(t.stamp, 0, sizeof t.stamp); t.gen = 1; }
  size_t w = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    const Board& a = out[i].after;
    u64 h = a.hash();
    int slot = int(h & (DedupTable::kSize - 1));
    bool dup = false;
    while (t.stamp[slot] == t.gen) {
      if (t.key[slot] == h && out[t.idx[slot]].after == a) { dup = true; break; }
      slot = (slot + 1) & (DedupTable::kSize - 1);
    }
    if (dup) continue;
    t.stamp[slot] = t.gen;
    t.key[slot] = h;
    t.idx[slot] = int(w);
    if (w != i) out[w] = out[i];
    ++w;
  }
  out.resize(w);
}

std::string Move::str() const {
  if (n == 0) return "(no move)";
  struct Step { int from, to; bool hit; };
  Step st[4];
  int k = n;
  for (int i = 0; i < k; ++i) st[i] = {from[i], to[i], bool((hits >> i) & 1)};
  // Merge chains of the same checker where the intermediate point is not a hit.
  struct Seg { int from, to; bool hit; std::string mid; };
  std::vector<Seg> segs;
  bool used[4] = {};
  auto pt = [](int s) -> std::string {
    if (s == kBar) return "bar";
    if (s < 0) return "off";
    return std::to_string(s + 1);
  };
  // Process steps in order of source (highest first) to build chains.
  int order[4] = {0, 1, 2, 3};
  std::sort(order, order + k, [&](int a, int b) { return st[a].from > st[b].from; });
  for (int oi = 0; oi < k; ++oi) {
    int i = order[oi];
    if (used[i]) continue;
    used[i] = true;
    Seg sg{st[i].from, st[i].to, st[i].hit, ""};
    while (true) {
      int nxt = -1;
      for (int oj = 0; oj < k; ++oj) {
        int j = order[oj];
        if (!used[j] && sg.to >= 0 && st[j].from == sg.to) { nxt = j; break; }
      }
      if (nxt < 0) break;
      used[nxt] = true;
      if (sg.hit) sg.mid += "/" + pt(sg.to) + "*";
      sg.to = st[nxt].to;
      sg.hit = st[nxt].hit;
    }
    segs.push_back(sg);
  }
  std::sort(segs.begin(), segs.end(), [](const Seg& a, const Seg& b) {
    if (a.from != b.from) return a.from > b.from;
    return a.to > b.to;
  });
  std::string s;
  for (size_t i = 0; i < segs.size();) {
    size_t j = i;
    while (j < segs.size() && segs[j].from == segs[i].from && segs[j].to == segs[i].to &&
           segs[j].mid == segs[i].mid && segs[j].hit == segs[i].hit)
      ++j;
    if (!s.empty()) s += ' ';
    // mid holds the "/x*" pieces of intermediate hits.
    s += pt(segs[i].from) + segs[i].mid + "/" + pt(segs[i].to) + (segs[i].hit ? "*" : "");
    if (j - i > 1) s += "(" + std::to_string(j - i) + ")";
    i = j;
  }
  return s;
}

int find_move(const std::vector<Move>& moves, const Board& before, const std::string& text) {
  // Parse into steps.
  std::vector<std::pair<int, int>> steps;
  auto parse_pt = [](const std::string& t, int& out) -> bool {
    std::string u;
    for (char c : t) if (c != '*') u += char(std::tolower(c));
    if (u == "bar" || u == "b" || u == "25") { out = kBar; return true; }
    if (u == "off" || u == "o" || u == "0") { out = -1; return true; }
    if (u.empty() || u.size() > 2) return false;
    for (char c : u) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    int v = std::stoi(u);
    if (v < 1 || v > 24) return false;
    out = v - 1;
    return true;
  };
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() && (text[i] == ' ' || text[i] == ',')) ++i;
    if (i >= text.size()) break;
    size_t j = i;
    while (j < text.size() && text[j] != ' ' && text[j] != ',') ++j;
    std::string tok = text.substr(i, j - i);
    i = j;
    int rep = 1;
    size_t lp = tok.find('(');
    if (lp != std::string::npos) {
      rep = std::atoi(tok.c_str() + lp + 1);
      tok = tok.substr(0, lp);
      if (rep < 1 || rep > 4) return -1;
    }
    std::vector<int> pts;
    size_t a = 0;
    while (a <= tok.size()) {
      size_t b = tok.find('/', a);
      if (b == std::string::npos) b = tok.size();
      int p;
      if (!parse_pt(tok.substr(a, b - a), p)) return -1;
      pts.push_back(p);
      a = b + 1;
    }
    if (pts.size() < 2) return -1;
    for (int r = 0; r < rep; ++r)
      for (size_t q = 0; q + 1 < pts.size(); ++q) steps.push_back({pts[q], pts[q + 1]});
  }
  // Apply blindly (final position does not depend on step order).
  u128 me = before.side[0], opp = before.side[1];
  for (auto [f, t] : steps) {
    if (!nibble(me, f)) return -1;
    me -= nibble_bit(f);
    if (t >= 0) {
      me += nibble_bit(t);
      int o = 23 - t;
      if (nibble(opp, o) == 1) { opp -= nibble_bit(o); opp += nibble_bit(kBar); }
    }
  }
  Board target{{me, opp}};
  for (size_t m = 0; m < moves.size(); ++m)
    if (moves[m].after == target) return int(m);
  return -1;
}

u64 perft(const Board& b, int depth) {
  if (depth == 0) return 1;
  if (b.winner() >= 0) return 1;
  std::vector<Move> moves;
  u64 total = 0;
  for (int d1 = 1; d1 <= 6; ++d1)
    for (int d2 = d1; d2 <= 6; ++d2) {
      generate_moves(b, d1, d2, moves);
      if (depth == 1) { total += moves.size(); continue; }
      std::vector<Move> copy = moves;
      for (const Move& m : copy) total += perft(m.after.swapped(), depth - 1);
    }
  return total;
}

}  // namespace bg
