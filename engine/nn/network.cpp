#include "network.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace bg {

namespace {

inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
inline float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

inline float dot(const float* __restrict a, const float* __restrict b, int n) {
  // Several independent partial sums so the compiler can keep all FMA units busy.
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4]; s5 += a[i + 5] * b[i + 5]; s6 += a[i + 6] * b[i + 6]; s7 += a[i + 7] * b[i + 7];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

inline void add_row(float* __restrict acc, const float* __restrict w, int n) {
  for (int i = 0; i < n; ++i) acc[i] += w[i];
}
inline void sub_row(float* __restrict acc, const float* __restrict w, int n) {
  for (int i = 0; i < n; ++i) acc[i] -= w[i];
}

inline int ctz128(u128 x) {
  u64 lo = u64(x);
  return lo ? __builtin_ctzll(lo) : 64 + __builtin_ctzll(u64(x >> 64));
}

constexpr u32 kMagic = 0x4E4E4742;  // "BGNN"
constexpr u32 kVersion = 1;

}  // namespace

void Network::prepare() {
  const int in = in2();
  W2T.resize(size_t(in) * L2);
  for (int k = 0; k < L2; ++k)
    for (int j = 0; j < in; ++j) W2T[size_t(j) * L2 + k] = W2[size_t(k) * in + j];
}

void Network::init(int hidden, int l2, int l3, unsigned seed) {
  H = hidden; L2 = l2; L3 = l3; E = kExtraFeatures; heads = kHeads;
  std::mt19937 rng(seed);
  auto fill = [&](std::vector<float>& v, size_t n, float a) {
    std::uniform_real_distribution<float> u(-a, a);
    v.resize(n);
    for (auto& x : v) x = u(rng);
  };
  fill(W1, size_t(kSparseFeatures) * H, 0.13f);
  b1.assign(H, 0.5f);
  fill(W2, size_t(L2) * in2(), std::sqrt(6.f / (in2() + L2)));
  b2.assign(L2, 0.f);
  fill(W3, size_t(L3) * L2, std::sqrt(6.f / (L2 + L3)));
  b3.assign(L3, 0.f);
  fill(Wo, size_t(heads) * 5 * L3, std::sqrt(6.f / (L3 + 5)));
  bo.assign(size_t(heads) * 5, 0.f);
  id = "random";
  prepare();
}

bool Network::save(const std::string& path) const {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  u32 hdr[2] = {kMagic, kVersion};
  int32_t dims[5] = {H, E, L2, L3, heads};
  u32 idlen = u32(id.size());
  std::fwrite(hdr, sizeof hdr, 1, f);
  std::fwrite(dims, sizeof dims, 1, f);
  std::fwrite(&idlen, sizeof idlen, 1, f);
  std::fwrite(id.data(), 1, idlen, f);
  for (const auto* v : {&W1, &b1, &W2, &b2, &W3, &b3, &Wo, &bo}) std::fwrite(v->data(), sizeof(float), v->size(), f);
  bool ok = std::ferror(f) == 0;
  std::fclose(f);
  return ok;
}

bool Network::load(const std::string& path, std::string* err) {
  auto fail = [&](const char* m) { if (err) *err = m; return false; };
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return fail("cannot open network file");
  u32 hdr[2];
  int32_t dims[5];
  u32 idlen = 0;
  bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == kMagic && hdr[1] == kVersion &&
            std::fread(dims, sizeof dims, 1, f) == 1 && std::fread(&idlen, sizeof idlen, 1, f) == 1 && idlen < 4096;
  if (!ok) { std::fclose(f); return fail("bad network header"); }
  if (dims[0] <= 0 || dims[0] > kMaxHidden || dims[1] != kExtraFeatures || dims[4] != kHeads) {
    std::fclose(f);
    return fail("unsupported network dimensions");
  }
  H = dims[0]; E = dims[1]; L2 = dims[2]; L3 = dims[3]; heads = dims[4];
  id.assign(idlen, ' ');
  ok = std::fread(id.data(), 1, idlen, f) == idlen;
  W1.resize(size_t(kSparseFeatures) * H); b1.resize(H);
  W2.resize(size_t(L2) * in2()); b2.resize(L2);
  W3.resize(size_t(L3) * L2); b3.resize(L3);
  Wo.resize(size_t(heads) * 5 * L3); bo.resize(size_t(heads) * 5);
  for (auto* v : {&W1, &b1, &W2, &b2, &W3, &b3, &Wo, &bo})
    ok = ok && std::fread(v->data(), sizeof(float), v->size(), f) == v->size();
  std::fclose(f);
  if (!ok) { W1.clear(); return fail("truncated network file"); }
  if (L2 > 64 || L3 > 64) { W1.clear(); return fail("unsupported layer sizes"); }
  prepare();
  return true;
}

void acc_refresh(const Network& net, const Board& b, Accumulator& acc) {
  const int H = net.H;
  int feats[2 * kCheckers * 2 + 64];
  for (int p = 0; p < 2; ++p) {
    float* a = acc.v[p];
    std::memcpy(a, net.b1.data(), sizeof(float) * H);
    int n = active_features(b, p, feats);
    for (int i = 0; i < n; ++i) add_row(a, &net.W1[size_t(feats[i]) * H], H);
  }
}

void acc_update(const Network& net, const Board& parent, const Accumulator& pa, const Board& child,
                Accumulator& ca) {
  const int H = net.H;
  std::memcpy(ca.v[0], pa.v[0], sizeof(float) * H);
  std::memcpy(ca.v[1], pa.v[1], sizeof(float) * H);
  for (int p = 0; p < 2; ++p) {
    const u128 a = parent.side[p], c = child.side[p];
    u128 diff = a ^ c;
    if (!diff) continue;
    auto toggle = [&](int slot, int lo, int hi, bool on) {  // thermometer k in (lo, hi]
      for (int k = lo + 1; k <= hi; ++k) {
        const float* w_own = &net.W1[size_t(feature_index(0, slot, k)) * H];
        const float* w_oth = &net.W1[size_t(feature_index(1, slot, k)) * H];
        if (on) { add_row(ca.v[p], w_own, H); add_row(ca.v[1 - p], w_oth, H); }
        else { sub_row(ca.v[p], w_own, H); sub_row(ca.v[1 - p], w_oth, H); }
      }
    };
    while (diff) {
      int slot = ctz128(diff) >> 2;
      diff &= ~(u128(15) << (4 * slot));
      int o = nibble(a, slot), n = nibble(c, slot);
      if (n > o) toggle(slot, o, n, true);
      else toggle(slot, n, o, false);
    }
    int off_a = kCheckers - popcount_checkers(a), off_c = kCheckers - popcount_checkers(c);
    if (off_c > off_a) toggle(25, off_a, off_c, true);
    else if (off_c < off_a) toggle(25, off_c, off_a, false);
  }
}

Probs net_forward(const Network& net, const Accumulator& acc, int stm, const float* extra, int head) {
  const int H = net.H, in = net.in2();
  alignas(64) float x[2 * kMaxHidden + 16];
  const float* a = acc.v[stm];
  const float* b = acc.v[1 - stm];
  for (int i = 0; i < H; ++i) x[i] = clamp01(a[i]);
  for (int i = 0; i < H; ++i) x[H + i] = clamp01(b[i]);
  for (int e = 0; e < net.E; ++e) x[2 * H + e] = extra[e];
  alignas(64) float h2[64], h3[64];
  // Input-major accumulation over the non-zero inputs only (the clamped
  // accumulators are sparse); the fixed 32-wide inner loop vectorises fully.
  const int L2 = net.L2;
  const float* w2t = net.W2T.data();
  int nnz[2 * kMaxHidden + 16];
  int cnt = 0;
  for (int j = 0; j < in; ++j) {
    nnz[cnt] = j;
    cnt += x[j] != 0.f;
  }
  if (L2 == 32) {
    float acc[32];
    for (int k = 0; k < 32; ++k) acc[k] = net.b2[k];
    for (int t = 0; t < cnt; ++t) {
      const float xj = x[nnz[t]];
      const float* w = w2t + size_t(nnz[t]) * 32;
      for (int k = 0; k < 32; ++k) acc[k] += xj * w[k];
    }
    for (int k = 0; k < 32; ++k) h2[k] = acc[k];
  } else {
    for (int k = 0; k < L2; ++k) h2[k] = net.b2[k];
    for (int t = 0; t < cnt; ++t) {
      const float xj = x[nnz[t]];
      const float* w = w2t + size_t(nnz[t]) * L2;
      for (int k = 0; k < L2; ++k) h2[k] += xj * w[k];
    }
  }
  for (int k = 0; k < L2; ++k) h2[k] = clamp01(h2[k]);
  for (int j = 0; j < net.L3; ++j) h3[j] = clamp01(net.b3[j] + dot(&net.W3[size_t(j) * net.L2], h2, net.L2));
  Probs p;
  const float* wo = &net.Wo[size_t(head) * 5 * net.L3];
  const float* bo = &net.bo[size_t(head) * 5];
  for (int o = 0; o < 5; ++o) p[o] = sigmoid(bo[o] + dot(wo + o * net.L3, h3, net.L3));
  p.sanitize();
  return p;
}

}  // namespace bg
