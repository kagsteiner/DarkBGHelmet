// Legal move generation.
//
// A "move" is the full play for one roll (up to four checker steps). The
// generator returns every distinct resulting position, enforcing the rules that
// as many dice as possible must be used and that the larger die must be played
// when only one of the two can be.
#pragma once

#include <string>
#include <vector>

#include "board.hpp"

namespace bg {

struct Move {
  Board after;       // position after the play, still from the mover's perspective
  int8_t from[4];    // slots 0..24 (24 = bar)
  int8_t to[4];      // slots 0..23, or -1 = borne off
  uint8_t n = 0;     // number of checker steps
  uint8_t hits = 0;  // bit i set if step i hit a blot

  std::string str() const;  // gnubg-style notation, e.g. "24/18 13/11*" or "8/2(2)"
};

// Mask of slots (in the mover's coordinates, bits 0..23) that the opponent blocks.
u32 blocked_mask(const Board& b);

void generate_moves(const Board& b, int d1, int d2, std::vector<Move>& out);

// Parses a play like "13/7 8/7", "bar/22*", "6/off(2)", "24/18/13" and returns the
// index of the matching legal move in `moves`, or -1.
int find_move(const std::vector<Move>& moves, const Board& before, const std::string& text);

// Counts all positions after `depth` plies over all 21 rolls (with roll weights
// ignored) -- a "perft"-style test helper.
u64 perft(const Board& b, int depth);

}  // namespace bg
