#include "board.hpp"

#include <sstream>

namespace bg {

Board Board::start() {
  Board b;
  for (int p = 0; p < 2; ++p) {
    b.set(p, 23, 2);
    b.set(p, 12, 5);
    b.set(p, 7, 3);
    b.set(p, 5, 5);
  }
  return b;
}

int Board::pips(int player) const {
  int sum = 0;
  u128 x = side[player];
  for (int s = 0; x; ++s, x >>= 4) sum += int(x & 15) * (s + 1);
  return sum;
}

bool Board::contact() const {
  int a = highest_slot(side[0]);
  int b = highest_slot(side[1]);
  if (a < 0 || b < 0) return false;
  return a + b >= 24;
}

int Board::winner() const {
  if (side[0] == 0) return 0;
  if (side[1] == 0) return 1;
  return -1;
}

GameResult Board::result_for_winner(int w) const {
  if (side[w] != 0) return GameResult::None;
  const u128 loser = side[1 - w];
  if (off(1 - w) > 0) return GameResult::Single;
  // Loser's slots 18..24 are the winner's home board plus the bar.
  if (loser >> (4 * 18)) return GameResult::Backgammon;
  return GameResult::Gammon;
}

bool Board::valid(std::string* why) const {
  auto fail = [&](const char* m) { if (why) *why = m; return false; };
  for (int p = 0; p < 2; ++p)
    if (on_board(p) > kCheckers) return fail("more than 15 checkers");
  for (int s = 0; s < 24; ++s)
    if (count(0, s) && count(1, 23 - s)) return fail("point occupied by both sides");
  return true;
}

std::string Board::ascii() const {
  // X = player on roll (moves 24 -> 1), O = opponent. Points in X's numbering.
  auto cell = [&](int pt) -> std::string {
    int s = pt - 1;
    int x = count(0, s), o = count(1, 23 - s);
    if (x) return (x < 10 ? " X" : "X") + std::to_string(x);
    if (o) return (o < 10 ? " O" : "O") + std::to_string(o);
    return "  .";
  };
  std::ostringstream os;
  os << " 13 14 15 16 17 18   19 20 21 22 23 24\n";
  for (int pt = 13; pt <= 24; ++pt) os << cell(pt) << (pt == 18 ? "  " : "");
  os << "   O bar " << count(1, kBar) << " off " << off(1) << "\n";
  for (int pt = 12; pt >= 1; --pt) os << cell(pt) << (pt == 7 ? "  " : "");
  os << "   X bar " << count(0, kBar) << " off " << off(0) << "\n";
  os << " 12 11 10  9  8  7    6  5  4  3  2  1   (X on roll, pips X " << pips(0) << " O " << pips(1) << ")\n";
  return os.str();
}

}  // namespace bg
