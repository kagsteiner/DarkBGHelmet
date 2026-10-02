#include "ids.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <vector>

namespace bg {

namespace {

const char* kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64val(char c) {
  const char* p = std::strchr(kB64, c);
  return (p && c) ? int(p - kB64) : -1;
}

// Little-endian bit string helpers (bit i = byte i/8, bit i%8).
struct Bits {
  std::vector<u8> bytes;
  explicit Bits(size_t n) : bytes(n, 0) {}
  void set(int pos, int nbits, u32 v) {
    for (int i = 0; i < nbits; ++i)
      if ((v >> i) & 1) bytes[(pos + i) / 8] |= u8(1 << ((pos + i) % 8));
  }
  u32 get(int pos, int nbits) const {
    u32 v = 0;
    for (int i = 0; i < nbits; ++i)
      if ((bytes[(pos + i) / 8] >> ((pos + i) % 8)) & 1) v |= 1u << i;
    return v;
  }
};

// gnubg base64: groups of 3 bytes -> 4 chars, no padding; output length fixed.
std::string encode64(const std::vector<u8>& in, size_t nchars) {
  std::string s;
  u32 acc = 0;
  int nb = 0;
  size_t i = 0;
  while (s.size() < nchars) {
    if (nb < 6) {
      acc = (acc << 8) | (i < in.size() ? in[i] : 0);
      ++i;
      nb += 8;
    }
    s += kB64[(acc >> (nb - 6)) & 63];
    nb -= 6;
  }
  return s;
}

bool decode64(const std::string& s, std::vector<u8>& out, size_t nbytes) {
  out.assign(nbytes, 0);
  u32 acc = 0;
  int nb = 0;
  size_t o = 0;
  for (char c : s) {
    int v = b64val(c);
    if (v < 0) return false;
    acc = (acc << 6) | u32(v);
    nb += 6;
    if (nb >= 8) {
      if (o < nbytes) out[o++] = u8((acc >> (nb - 8)) & 255);
      nb -= 8;
    }
  }
  return o == nbytes;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> v;
  std::string cur;
  for (char c : s) {
    if (c == sep) { v.push_back(cur); cur.clear(); }
    else cur += c;
  }
  v.push_back(cur);
  return v;
}

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

}  // namespace

bool MatchState::valid(std::string* why) const {
  auto fail = [&](const char* m) { if (why) *why = m; return false; };
  if (length < 0 || length > kMaxMatch) return fail("match length must be 0 (money) or 1..25");
  if (cube < 1 || cube > (1 << 12) || (cube & (cube - 1))) return fail("cube must be a power of two");
  if (!money()) {
    if (score[0] < 0 || score[1] < 0 || score[0] >= length || score[1] >= length)
      return fail("scores must be below the match length");
    if (crawford && away(0) != 1 && away(1) != 1) return fail("Crawford game needs a player 1-away");
  }
  return true;
}

// The gnubg key stores the player not on roll first (anBoard[0]), then the
// player on roll; each as 25 slots (ace .. 24, bar): n ones followed by a zero.
std::string position_id(const Board& b) {
  Bits bits(10);
  int pos = 0;
  for (int p : {1, 0})
    for (int s = 0; s < kSlots; ++s) {
      int n = b.count(p, s);
      for (int i = 0; i < n; ++i) bits.set(pos++, 1, 1);
      ++pos;
    }
  return encode64(bits.bytes, 14);
}

bool parse_position_id(const std::string& id, Board& out) {
  std::string s = trim(id);
  if (s.size() != 14) return false;
  std::vector<u8> bytes;
  if (!decode64(s, bytes, 10)) return false;
  Bits bits(10);
  bits.bytes = bytes;
  Board b;
  int pos = 0;
  for (int p : {1, 0})
    for (int slot = 0; slot < kSlots; ++slot) {
      int n = 0;
      while (pos < 80 && bits.get(pos, 1)) { ++n; ++pos; }
      ++pos;
      if (n > kCheckers || pos > 80) return false;
      b.set(p, slot, n);
    }
  if (!b.valid()) return false;
  out = b;
  return true;
}

// Match key layout (66 bits, LSB first): cube log2 (4), cube owner (2, 3 =
// centered), dice owner (1), Crawford (1), game state (3), turn (1), doubled (1),
// resigned (2), die 1 (3), die 2 (3), match length (15), score 0 (15), score 1 (15).
// We always write the player on roll as gnubg player 1.
std::string match_id(const MatchState& m, int d1, int d2) {
  Bits bits(9);
  const int me = 1;
  bits.set(0, 4, u32(cube_log2(m.cube)));
  u32 owner = m.owner == CubeOwner::Center ? 3 : (m.owner == CubeOwner::Me ? me : 1 - me);
  bits.set(4, 2, owner);
  bits.set(6, 1, me);
  bits.set(7, 1, m.crawford ? 1 : 0);
  bits.set(8, 3, 1);  // game playing
  bits.set(11, 1, me);
  bits.set(15, 3, u32(d1));
  bits.set(18, 3, u32(d2));
  bits.set(21, 15, u32(m.length));
  bits.set(36, 15, u32(m.score[1]));  // gnubg player 0 = opponent
  bits.set(51, 15, u32(m.score[0]));                     // gnubg player 1 = on roll
  return encode64(bits.bytes, 12);
}

bool parse_match_id(const std::string& id, MatchState& m, int& d1, int& d2) {
  std::string s = trim(id);
  if (s.size() != 12) return false;
  std::vector<u8> bytes;
  if (!decode64(s, bytes, 9)) return false;
  Bits bits(9);
  bits.bytes = bytes;
  MatchState r;
  int lc = int(bits.get(0, 4));
  r.cube = 1 << lc;
  int owner = int(bits.get(4, 2));
  int roller = int(bits.get(6, 1));
  r.crawford = bits.get(7, 1);
  d1 = int(bits.get(15, 3));
  d2 = int(bits.get(18, 3));
  r.length = int(bits.get(21, 15));
  int sc[2] = {int(bits.get(36, 15)), int(bits.get(51, 15))};
  r.score[0] = sc[roller];
  r.score[1] = sc[1 - roller];
  r.owner = owner == 3 ? CubeOwner::Center : (owner == roller ? CubeOwner::Me : CubeOwner::Opp);
  if (d1 > 6 || d2 > 6) return false;
  if (!r.valid()) return false;
  m = r;
  return true;
}

// XGID position string: char 0 = O's bar, chars 1..24 = points 1..24 from X's
// side, char 25 = X's bar. 'A'..'O' = X checkers, 'a'..'o' = O checkers.
bool parse_xgid(const std::string& text, Position& out, std::string* err) {
  auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
  std::string s = trim(text);
  if (s.rfind("XGID=", 0) == 0) s = s.substr(5);
  auto f = split(s, ':');
  if (f.size() < 10) return fail("XGID needs 10 fields");
  const std::string& ps = f[0];
  if (ps.size() != 26) return fail("XGID position must have 26 characters");
  int x[26] = {}, o[26] = {};
  for (int i = 0; i < 26; ++i) {
    char c = ps[i];
    if (c == '-') continue;
    if (c >= 'A' && c <= 'P') x[i] = c - 'A' + 1;
    else if (c >= 'a' && c <= 'p') o[i] = c - 'a' + 1;
    else return fail("bad XGID position character");
  }
  // X: slot s = point s+1 -> char s+1; bar = char 25.
  // O: O's slot t = X's point 24-t -> char 24-t; bar = char 0.
  Board bx;  // side 0 = X, side 1 = O
  for (int sl = 0; sl < 24; ++sl) {
    bx.set(0, sl, x[sl + 1]);
    bx.set(1, sl, o[24 - sl]);
  }
  bx.set(0, kBar, x[25]);
  bx.set(1, kBar, o[0]);
  if (x[0] || o[25]) return fail("checkers on the wrong bar");
  std::string why;
  if (!bx.valid(&why)) return fail(why);

  int cube_log = std::atoi(f[1].c_str());
  int cube_pos = std::atoi(f[2].c_str());
  int turn = std::atoi(f[3].c_str());
  const std::string& dice = f[4];
  int score_x = std::atoi(f[5].c_str());
  int score_o = std::atoi(f[6].c_str());
  int cj = std::atoi(f[7].c_str());
  int len = std::atoi(f[8].c_str());

  bool x_on_roll = turn != -1;
  Position p;
  p.board = x_on_roll ? bx : bx.swapped();
  MatchState& m = p.match;
  m.length = len;
  m.cube = 1 << cube_log;
  int sx = x_on_roll ? 0 : 1;  // our index of X
  m.score[sx] = score_x;
  m.score[1 - sx] = score_o;
  if (cube_pos == 0) m.owner = CubeOwner::Center;
  else {
    bool owner_is_x = cube_pos == 1;
    m.owner = (owner_is_x == x_on_roll) ? CubeOwner::Me : CubeOwner::Opp;
  }
  if (len > 0) {
    m.crawford = cj != 0;
  } else {
    m.jacoby = (cj & 1) != 0;
    m.beavers = (cj & 2) != 0;
  }
  if (dice.size() == 2 && std::isdigit(static_cast<unsigned char>(dice[0])) &&
      std::isdigit(static_cast<unsigned char>(dice[1]))) {
    p.dice[0] = dice[0] - '0';
    p.dice[1] = dice[1] - '0';
    if (p.dice[0] > 6 || p.dice[1] > 6 || (p.dice[0] == 0) != (p.dice[1] == 0))
      return fail("bad dice");
  }
  if (!m.valid(&why)) return fail(why);
  out = p;
  return true;
}

std::string to_xgid(const Position& p) {
  // Player on roll is written as X.
  const Board& b = p.board;
  char ps[27];
  ps[26] = 0;
  for (int i = 0; i < 26; ++i) ps[i] = '-';
  for (int sl = 0; sl < 24; ++sl) {
    if (int n = b.count(0, sl)) ps[sl + 1] = char('A' + n - 1);
    if (int n = b.count(1, sl)) ps[24 - sl] = char('a' + n - 1);
  }
  if (int n = b.count(0, kBar)) ps[25] = char('A' + n - 1);
  if (int n = b.count(1, kBar)) ps[0] = char('a' + n - 1);
  const MatchState& m = p.match;
  int cube_pos = m.owner == CubeOwner::Center ? 0 : (m.owner == CubeOwner::Me ? 1 : -1);
  std::ostringstream os;
  os << "XGID=" << ps << ':' << cube_log2(m.cube) << ':' << cube_pos << ":1:";
  if (p.dice[0]) os << p.dice[0] << p.dice[1];
  else os << "00";
  int cj = m.money() ? (m.jacoby ? 1 : 0) | (m.beavers ? 2 : 0) : (m.crawford ? 1 : 0);
  os << ':' << m.score[0] << ':' << m.score[1] << ':' << cj << ':' << m.length << ":10";
  return os.str();
}

bool parse_any(const std::string& text, Position& out, std::string* err) {
  std::string s = trim(text);
  if (s.rfind("XGID=", 0) == 0 || std::count(s.begin(), s.end(), ':') >= 9) return parse_xgid(s, out, err);
  auto parts = split(s, ':');
  Position p;
  if (!parse_position_id(parts[0], p.board)) {
    if (err) *err = "invalid position ID";
    return false;
  }
  if (parts.size() >= 2 && !parts[1].empty()) {
    int d1, d2;
    if (!parse_match_id(parts[1], p.match, d1, d2)) {
      if (err) *err = "invalid match ID";
      return false;
    }
    p.dice[0] = d1;
    p.dice[1] = d2;
  }
  out = p;
  return true;
}

}  // namespace bg
