#include "network.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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
constexpr u32 kVersion = 2;  // 2 = optional cube head after the main weights

}  // namespace

void Network::prepare() {
  const int in = in2();
  W2T.resize(size_t(in) * L2);
  for (int k = 0; k < L2; ++k)
    for (int j = 0; j < in; ++j) W2T[size_t(j) * L2 + k] = W2[size_t(k) * in + j];
  W3T.resize(size_t(L2) * L3);
  for (int k = 0; k < L3; ++k)
    for (int j = 0; j < L2; ++j) W3T[size_t(j) * L3 + k] = W3[size_t(k) * L2 + j];
}

void Network::init_cube_head(int hidden, unsigned seed) {
  C = hidden;
  std::mt19937 rng(seed);
  auto fill = [&](std::vector<float>& v, size_t n, float a) {
    std::uniform_real_distribution<float> u(-a, a);
    v.resize(n);
    for (auto& x : v) x = u(rng);
  };
  fill(Wc1, size_t(heads) * C * cin(), std::sqrt(6.f / (cin() + C)));
  bc1.assign(size_t(heads) * C, 0.f);
  fill(Wc2, size_t(heads) * 3 * C, 0.05f);
  // Start at x = 0.76 (sigmoid^-1((0.76 - 0.3) / 0.9) ~ 0.045).
  bc2.assign(size_t(heads) * 3, 0.045f);
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
  int32_t c = C;
  std::fwrite(&c, sizeof c, 1, f);
  if (C > 0)
    for (const auto* v : {&Wc1, &bc1, &Wc2, &bc2}) std::fwrite(v->data(), sizeof(float), v->size(), f);
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
  bool ok = std::fread(hdr, sizeof hdr, 1, f) == 1 && hdr[0] == kMagic && (hdr[1] == 1 || hdr[1] == 2) &&
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
  C = 0;
  if (ok && hdr[1] >= 2) {
    int32_t c = 0;
    ok = std::fread(&c, sizeof c, 1, f) == 1 && c >= 0 && c <= 256;
    if (ok && c > 0) {
      C = c;
      Wc1.resize(size_t(heads) * C * cin()); bc1.resize(size_t(heads) * C);
      Wc2.resize(size_t(heads) * 3 * C); bc2.resize(size_t(heads) * 3);
      for (auto* v : {&Wc1, &bc1, &Wc2, &bc2})
        ok = ok && std::fread(v->data(), sizeof(float), v->size(), f) == v->size();
    }
  }
  std::fclose(f);
  if (!ok) { W1.clear(); return fail("truncated network file"); }
  if (L2 > kMaxDense || L3 > kMaxDense) { W1.clear(); return fail("unsupported layer sizes"); }
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
  // Collect the weight rows to add/subtract per perspective, then write each
  // child accumulator in a single fused pass.
  const float* add[2][64];
  const float* sub[2][64];
  int na[2] = {0, 0}, ns[2] = {0, 0};
  for (int p = 0; p < 2; ++p) {
    const u128 a = parent.side[p], c = child.side[p];
    u128 diff = a ^ c;
    if (!diff) continue;
    auto toggle = [&](int slot, int lo, int hi, bool on) {  // thermometer k in (lo, hi]
      for (int k = lo + 1; k <= hi; ++k) {
        const float* w_own = &net.W1[size_t(feature_index(0, slot, k)) * H];
        const float* w_oth = &net.W1[size_t(feature_index(1, slot, k)) * H];
        if (on) { add[p][na[p]++] = w_own; add[1 - p][na[1 - p]++] = w_oth; }
        else { sub[p][ns[p]++] = w_own; sub[1 - p][ns[1 - p]++] = w_oth; }
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
  for (int p = 0; p < 2; ++p) {
    const float* src = pa.v[p];
    float* dst = ca.v[p];
    if (na[p] + ns[p] > 60) {  // cannot happen for a single play, but stay safe
      acc_refresh(net, child, ca);
      return;
    }
    for (int i = 0; i < H; i += 16) {
      float t[16];
      for (int j = 0; j < 16; ++j) t[j] = src[i + j];
      for (int r = 0; r < na[p]; ++r) {
        const float* w = add[p][r] + i;
        for (int j = 0; j < 16; ++j) t[j] += w[j];
      }
      for (int r = 0; r < ns[p]; ++r) {
        const float* w = sub[p][r] + i;
        for (int j = 0; j < 16; ++j) t[j] -= w[j];
      }
      for (int j = 0; j < 16; ++j) dst[i + j] = t[j];
    }
  }
}

namespace {

// Indices of the positive entries of a[0..n) (n multiple of 4), offset by `base`.
inline int find_positive(const float* a, int n, int base, uint16_t* out) {
  int cnt = 0;
#if defined(__ARM_NEON)
  static const struct Lut {
    uint16_t idx[16][4];
    Lut() {
      for (int m = 0; m < 16; ++m) {
        int k = 0;
        for (int b = 0; b < 4; ++b)
          if (m & (1 << b)) idx[m][k++] = uint16_t(b);
        for (; k < 4; ++k) idx[m][k] = 0;
      }
    }
  } lut;
  const float32x4_t zero = vdupq_n_f32(0.f);
  const uint32x4_t bits = {1, 2, 4, 8};
  for (int i = 0; i < n; i += 4) {
    const uint32x4_t gt = vcgtq_f32(vld1q_f32(a + i), zero);
    const unsigned m = vaddvq_u32(vandq_u32(gt, bits));
    const uint16x4_t idx = vadd_u16(vld1_u16(lut.idx[m]), vdup_n_u16(uint16_t(base + i)));
    vst1_u16(out + cnt, idx);
    cnt += __builtin_popcount(m);
  }
#else
  for (int i = 0; i < n; ++i) {
    out[cnt] = uint16_t(base + i);
    cnt += a[i] > 0.f;
  }
#endif
  return cnt;
}

}  // namespace

namespace {

// First dense layer over the sparse accumulator inputs (input-major weights).
template <int N>
void layer2(const Network& net, const float* a, const float* b, const uint16_t* nnz, int cnt, const float* extra,
            float* h2) {
  const int H = net.H;
  const float* w2t = net.W2T.data();
  float t[N];
  for (int k = 0; k < N; ++k) t[k] = net.b2[k];
  for (int q = 0; q < cnt; ++q) {
    const int j = nnz[q];
    float xj = j < H ? a[j] : b[j - H];
    xj = xj < 1.f ? xj : 1.f;
    const float* w = w2t + size_t(j) * N;
    for (int k = 0; k < N; ++k) t[k] += xj * w[k];
  }
  for (int e = 0; e < net.E; ++e) {
    const float xj = extra[e];
    const float* w = w2t + size_t(2 * H + e) * N;
    for (int k = 0; k < N; ++k) t[k] += xj * w[k];
  }
  for (int k = 0; k < N; ++k) h2[k] = clamp01(t[k]);
}

void layer2_generic(const Network& net, const float* a, const float* b, const uint16_t* nnz, int cnt,
                    const float* extra, float* h2) {
  const int H = net.H, L2 = net.L2;
  const float* w2t = net.W2T.data();
  for (int k = 0; k < L2; ++k) h2[k] = net.b2[k];
  for (int q = 0; q < cnt; ++q) {
    const int j = nnz[q];
    float xj = j < H ? a[j] : b[j - H];
    xj = xj < 1.f ? xj : 1.f;
    const float* w = w2t + size_t(j) * L2;
    for (int k = 0; k < L2; ++k) h2[k] += xj * w[k];
  }
  for (int e = 0; e < net.E; ++e) {
    const float* w = w2t + size_t(2 * H + e) * L2;
    for (int k = 0; k < L2; ++k) h2[k] += extra[e] * w[k];
  }
  for (int k = 0; k < L2; ++k) h2[k] = clamp01(h2[k]);
}

// Second dense layer, input-major over the non-zero h2 entries.
template <int N>
void layer3(const Network& net, const float* h2, float* h3) {
  const float* w3t = net.W3T.data();
  float t[N];
  for (int k = 0; k < N; ++k) t[k] = net.b3[k];
  for (int j = 0; j < net.L2; ++j) {
    const float xj = h2[j];
    if (xj == 0.f) continue;
    const float* w = w3t + size_t(j) * N;
    for (int k = 0; k < N; ++k) t[k] += xj * w[k];
  }
  for (int k = 0; k < N; ++k) h3[k] = clamp01(t[k]);
}

}  // namespace

Probs net_forward(const Network& net, const Accumulator& acc, int stm, const float* extra, int head, float* h3_out) {
  const int H = net.H, L3 = net.L3;
  const float* a = acc.v[stm];
  const float* b = acc.v[1 - stm];
  // Only positive accumulator entries survive the clamp; collect their indices.
  alignas(64) uint16_t nnz[2 * kMaxHidden + 8];
  int cnt = find_positive(a, H, 0, nnz);
  cnt += find_positive(b, H, H, nnz + cnt);
  alignas(64) float h2[kMaxDense], h3[kMaxDense];
  switch (net.L2) {
    case 32: layer2<32>(net, a, b, nnz, cnt, extra, h2); break;
    case 64: layer2<64>(net, a, b, nnz, cnt, extra, h2); break;
    case 96: layer2<96>(net, a, b, nnz, cnt, extra, h2); break;
    case 128: layer2<128>(net, a, b, nnz, cnt, extra, h2); break;
    default: layer2_generic(net, a, b, nnz, cnt, extra, h2);
  }
  switch (L3) {
    case 32: layer3<32>(net, h2, h3); break;
    case 64: layer3<64>(net, h2, h3); break;
    case 96: layer3<96>(net, h2, h3); break;
    case 128: layer3<128>(net, h2, h3); break;
    default:
      for (int j = 0; j < L3; ++j) h3[j] = clamp01(net.b3[j] + dot(&net.W3[size_t(j) * net.L2], h2, net.L2));
  }
  if (h3_out) std::memcpy(h3_out, h3, sizeof(float) * L3);
  Probs p;
  const float* wo = &net.Wo[size_t(head) * 5 * L3];
  const float* bo = &net.bo[size_t(head) * 5];
  for (int o = 0; o < 5; ++o) p[o] = sigmoid(bo[o] + dot(wo + o * L3, h3, L3));
  p.sanitize();
  return p;
}

void net_cube_x(const Network& net, const float* h3, const Probs& p, int head, float x[3]) {
  const int C = net.C, in = net.cin(), L3 = net.L3;
  float u[256];
  const float* w1 = &net.Wc1[size_t(head) * C * in];
  const float* b1 = &net.bc1[size_t(head) * C];
  for (int k = 0; k < C; ++k) {
    const float* w = w1 + size_t(k) * in;
    float s = b1[k] + dot(w, h3, L3);
    for (int i = 0; i < 5; ++i) s += w[L3 + i] * p[i];
    u[k] = clamp01(s);
  }
  const float* w2 = &net.Wc2[size_t(head) * 3 * C];
  const float* b2 = &net.bc2[size_t(head) * 3];
  for (int o = 0; o < 3; ++o) x[o] = kCubeXMin + kCubeXSpan * sigmoid(b2[o] + dot(w2 + o * C, u, C));
}

}  // namespace bg
