// Packed backgammon board ("bitboard" representation).
//
// Each side is a 128-bit lane holding 25 nibbles (4 bits = 0..15 checkers):
//   nibble s, s = 0..23  -> the side's own point s+1 (s = 0 is the ace point)
//   nibble 24            -> the bar
// Checkers borne off are implicit (15 - sum). Both lanes are stored from their
// owner's perspective, so passing the turn is a swap of the two lanes, and the
// opponent's point that faces our slot s is the opponent's slot 23 - s.
#pragma once

#include <cstdint>
#include <string>

namespace bg {

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = unsigned __int128;

constexpr int kBar = 24;
constexpr int kSlots = 25;
constexpr int kCheckers = 15;

constexpr u128 nibble_bit(int s) { return u128(1) << (4 * s); }
constexpr int nibble(u128 x, int s) { return int((x >> (4 * s)) & 15); }

// Bits of all nibbles at slots 6..24, i.e. everything outside the home board.
constexpr u128 kOutsideHome = ~((u128(1) << 24) - 1);

inline int popcount_checkers(u128 x) {
  // Sum of all nibbles: split into byte lanes (each <= 30), then sum bytes.
  const u64 m = 0x0F0F0F0F0F0F0F0FULL;
  u64 lo = u64(x), hi = u64(x >> 64);
  u64 a = (lo & m) + ((lo >> 4) & m);
  u64 b = (hi & m) + ((hi >> 4) & m);
  return int(((a * 0x0101010101010101ULL) >> 56) + ((b * 0x0101010101010101ULL) >> 56));
}

// Index of the highest occupied slot, or -1 if empty.
inline int highest_slot(u128 x) {
  if (!x) return -1;
  u64 hi = u64(x >> 64);
  int bit = hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll(u64(x));
  return bit >> 2;
}

inline u64 mix64(u64 x) {
  x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27; x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

enum class GameResult : int8_t { None = 0, Single = 1, Gammon = 2, Backgammon = 3 };

struct Board {
  u128 side[2] = {0, 0};  // side[0] = player on roll, side[1] = opponent

  static Board start();

  int count(int player, int slot) const { return nibble(side[player], slot); }
  void set(int player, int slot, int n) {
    side[player] = (side[player] & ~(u128(15) << (4 * slot))) | (u128(n) << (4 * slot));
  }
  int on_board(int player) const { return popcount_checkers(side[player]); }
  int off(int player) const { return kCheckers - on_board(player); }
  int pips(int player) const;

  Board swapped() const { return Board{{side[1], side[0]}}; }

  // True if the checkers of the two sides can still meet.
  bool contact() const;

  // Result for the side that has borne off all checkers (sides are symmetric).
  // Returns None if neither side has finished.
  GameResult result_for_winner(int winner) const;
  int winner() const;  // 0, 1 or -1

  bool valid(std::string* why = nullptr) const;

  u64 hash() const {
    return mix64(u64(side[0]) ^ mix64(u64(side[0] >> 64) ^ mix64(u64(side[1]) ^ mix64(u64(side[1] >> 64) + 0x9e37))));
  }
  bool operator==(const Board& o) const { return side[0] == o.side[0] && side[1] == o.side[1]; }
  bool operator!=(const Board& o) const { return !(*this == o); }

  std::string ascii() const;  // human-readable diagram, player on roll = X at the bottom
};

}  // namespace bg
