// Network input features and position classes.
//
// Sparse part (NNUE accumulator input): for a perspective player P there are two
// blocks -- P's own checkers and the other player's checkers, each in its owner's
// coordinates. A block has 26 slots (24 points, bar, borne off) x 15 thermometer
// features "at least k checkers here" (k = 1..15). Moving one checker one step
// therefore toggles exactly two features in each perspective.
//
// Dense part: a few hand-crafted features computed at the leaf (pip counts,
// hitting chances, escapes) that a small network struggles to derive itself.
#pragma once

#include "core/board.hpp"

namespace bg {

constexpr int kThermo = 15;
constexpr int kSlots26 = 26;  // 0..23 points, 24 bar, 25 off
constexpr int kBlockFeatures = kSlots26 * kThermo;  // 390
constexpr int kSparseFeatures = 2 * kBlockFeatures; // 780
constexpr int kMaxActive = 2 * kCheckers + 2 * kSlots26;  // generous upper bound

inline int feature_index(int block, int slot26, int k) { return block * kBlockFeatures + slot26 * kThermo + (k - 1); }

// Active features for the perspective of board side p. Returns the count.
int active_features(const Board& b, int p, int* out);

constexpr int kExtraFeatures = 8;
// Dense features from the view of board side 0 (player on roll).
void extra_features(const Board& b, float* out);

// Fraction of the 36 rolls with which side 0 can hit a blot of side 1.
float hit_chance(const Board& b);
// Fraction of the 36 rolls with which side 0's rearmost checker gets past the
// opponent's points within 12 pips in front of it (1 if no contact).
float escape_chance(const Board& b);

enum class PosClass : int8_t { Over = 0, Bearoff = 1, Race = 2, Contact = 3, Crashed = 4 };
PosClass classify(const Board& b);

// Network output head for a class (Race, Contact, Crashed; Bearoff uses Race).
inline int head_for(PosClass c) {
  switch (c) {
    case PosClass::Contact: return 0;
    case PosClass::Crashed: return 1;
    default: return 2;
  }
}
constexpr int kHeads = 3;

}  // namespace bg
