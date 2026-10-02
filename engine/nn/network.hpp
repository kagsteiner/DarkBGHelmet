// BG-NNUE: sparse thermometer inputs -> two perspective accumulators -> small
// dense network -> one of three output heads (contact, crashed, race), each
// producing the five outcome probabilities.
//
//   acc[p]  = b1 + sum_{f in active(p)} W1[f]          (p = each board side)
//   x       = [clamp(acc[stm]), clamp(acc[other]), extra features]
//   h2      = clamp(W2 x + b2), h3 = clamp(W3 h2 + b3)
//   out     = sigmoid(Wo[head] h3 + bo[head])
//
// The accumulators are updated incrementally from the parent position: a play
// changes at most a handful of thermometer features.
#pragma once

#include <string>
#include <vector>

#include "core/board.hpp"
#include "eval/features.hpp"
#include "eval/probs.hpp"

namespace bg {

constexpr int kMaxHidden = 512;

struct Network {
  int H = 256, E = kExtraFeatures, L2 = 32, L3 = 32, heads = kHeads;
  std::vector<float> W1, b1;  // [780][H], [H]
  std::vector<float> W2, b2;  // [L2][2H+E], [L2]
  std::vector<float> W3, b3;  // [L3][L2], [L3]
  std::vector<float> Wo, bo;  // [heads][5][L3], [heads][5]
  std::string id = "untrained";

  int in2() const { return 2 * H + E; }
  void init(int hidden, int l2, int l3, unsigned seed);
  bool load(const std::string& path, std::string* err = nullptr);
  bool save(const std::string& path) const;
  bool empty() const { return W1.empty(); }
};

struct alignas(64) Accumulator {
  float v[2][kMaxHidden];  // v[p] = accumulator of board side p's perspective
};

void acc_refresh(const Network& net, const Board& b, Accumulator& acc);
// `child` must have the same orientation as `parent` (no swap applied).
void acc_update(const Network& net, const Board& parent, const Accumulator& pa, const Board& child,
                Accumulator& ca);

// Evaluates with board side `stm` on roll. `extra` = extra_features of the board
// seen from stm. Returns sanitized probabilities from stm's view.
Probs net_forward(const Network& net, const Accumulator& acc, int stm, const float* extra, int head);

}  // namespace bg
