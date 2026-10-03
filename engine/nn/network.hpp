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
constexpr int kMaxDense = 128;  // max width of the two small dense layers

struct Network {
  int H = 256, E = kExtraFeatures, L2 = 32, L3 = 32, heads = kHeads;
  std::vector<float> W1, b1;  // [780][H], [H]
  std::vector<float> W2, b2;  // [L2][2H+E], [L2]
  std::vector<float> W3, b3;  // [L3][L2], [L3]
  std::vector<float> Wo, bo;  // [heads][5][L3], [heads][5]
  std::string id = "untrained";
  std::vector<float> W2T;     // [2H+E][L2], derived from W2 by prepare() for fast inference
  std::vector<float> W3T;     // [L2][L3], derived from W3

  // Optional cube head (file version 2): per class, [h3, 5 probs] -> C (clamped)
  // -> 3 outputs = cube efficiency for the cube states centered / owned by the
  // player on roll / owned by the opponent, x = kCubeXMin + kCubeXSpan * sigmoid(z).
  int C = 0;
  std::vector<float> Wc1, bc1;  // [heads][C][L3+5], [heads][C]
  std::vector<float> Wc2, bc2;  // [heads][3][C], [heads][3]
  int cin() const { return L3 + 5; }
  bool has_cube_head() const { return C > 0; }
  void init_cube_head(int hidden, unsigned seed);

  int in2() const { return 2 * H + E; }
  void init(int hidden, int l2, int l3, unsigned seed);
  bool load(const std::string& path, std::string* err = nullptr);
  bool save(const std::string& path) const;
  bool empty() const { return W1.empty(); }
  void prepare();  // must be called after changing W2

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
// If h3_out is given it receives the last hidden layer (L3 values).
Probs net_forward(const Network& net, const Accumulator& acc, int stm, const float* extra, int head,
                  float* h3_out = nullptr);

constexpr float kCubeXMin = 0.3f, kCubeXSpan = 0.9f;
// Cube efficiencies from the cube head for (centered, owned, opponent owns).
void net_cube_x(const Network& net, const float* h3, const Probs& p, int head, float x[3]);

}  // namespace bg
