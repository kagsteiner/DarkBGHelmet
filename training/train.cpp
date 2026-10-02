// bgtrain: dataset building and network training.
//
//   bgtrain build <out.bin> [--min-plies P] <label files...>
//   bgtrain train <data.bin> <out.bgnn> [--hidden 256] [--epochs 12] [--lr 1e-3]
//                 [--batch 2048] [--threads 4] [--init net.bgnn] [--val 0.01]
//   bgtrain stats <data.bin>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <mutex>
#include <vector>

#include "core/ids.hpp"
#include "dataset.hpp"
#include "eval/evaluator.hpp"
#include "search/search.hpp"
#include "eval/features.hpp"
#include "nn/network.hpp"

using namespace bg;

namespace {

struct Layout {
  size_t W1, b1, W2, b2, W3, b3, Wo, bo, total;
  explicit Layout(const Network& n) {
    size_t o = 0;
    W1 = o; o += n.W1.size();
    b1 = o; o += n.b1.size();
    W2 = o; o += n.W2.size();
    b2 = o; o += n.b2.size();
    W3 = o; o += n.W3.size();
    b3 = o; o += n.b3.size();
    Wo = o; o += n.Wo.size();
    bo = o; o += n.bo.size();
    total = o;
  }
};

std::vector<float*> tensors(Network& n) { return {n.W1.data(), n.b1.data(), n.W2.data(), n.b2.data(), n.W3.data(), n.b3.data(), n.Wo.data(), n.bo.data()}; }
std::vector<size_t> sizes(const Network& n) { return {n.W1.size(), n.b1.size(), n.W2.size(), n.b2.size(), n.W3.size(), n.b3.size(), n.Wo.size(), n.bo.size()}; }

inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
inline float sigm(float x) { return 1.f / (1.f + std::exp(-x)); }

struct Stats { double loss = 0, eq_se = 0; long n = 0; };

// Forward + backward for one record; accumulates gradients into g.
void sample_grad(const Network& net, const Layout& L, const Record& r, float* g, Stats& st) {
  const int H = net.H, in = net.in2(), L2 = net.L2, L3 = net.L3;
  int feats[2][128];
  int nf[2];
  float acc[2][kMaxHidden];
  for (int p = 0; p < 2; ++p) {
    nf[p] = active_features(r.board, p, feats[p]);
    std::memcpy(acc[p], net.b1.data(), sizeof(float) * H);
    for (int i = 0; i < nf[p]; ++i) {
      const float* w = &net.W1[size_t(feats[p][i]) * H];
      for (int j = 0; j < H; ++j) acc[p][j] += w[j];
    }
  }
  float x[2 * kMaxHidden + 16];
  for (int j = 0; j < H; ++j) { x[j] = clamp01(acc[0][j]); x[H + j] = clamp01(acc[1][j]); }
  extra_features(r.board, x + 2 * H);
  float z2[kMaxDense], h2[kMaxDense], z3[kMaxDense], h3[kMaxDense], y[5];
  for (int k = 0; k < L2; ++k) {
    const float* w = &net.W2[size_t(k) * in];
    float s = net.b2[k];
    for (int j = 0; j < in; ++j) s += w[j] * x[j];
    z2[k] = s; h2[k] = clamp01(s);
  }
  for (int k = 0; k < L3; ++k) {
    const float* w = &net.W3[size_t(k) * L2];
    float s = net.b3[k];
    for (int j = 0; j < L2; ++j) s += w[j] * h2[j];
    z3[k] = s; h3[k] = clamp01(s);
  }
  const int head = head_for(PosClass(r.cls));
  const float* wo = &net.Wo[size_t(head) * 5 * L3];
  const float* bo = &net.bo[size_t(head) * 5];
  float dzo[5];
  for (int o = 0; o < 5; ++o) {
    float s = bo[o];
    for (int j = 0; j < L3; ++j) s += wo[o * L3 + j] * h3[j];
    y[o] = sigm(s);
    float t = r.target[o];
    dzo[o] = (y[o] - t) * r.weight;
    const float yc = std::clamp(y[o], 1e-6f, 1.f - 1e-6f);
    st.loss += -(t * std::log(yc) + (1 - t) * std::log(1 - yc));
  }
  {
    float eq = 2 * y[0] - 1 + y[1] - y[3] + y[2] - y[4];
    const float* t = r.target;
    float et = 2 * t[0] - 1 + t[1] - t[3] + t[2] - t[4];
    st.eq_se += double(eq - et) * (eq - et);
    st.n++;
  }
  // Backward.
  float* gWo = g + L.Wo + size_t(head) * 5 * L3;
  float* gbo = g + L.bo + size_t(head) * 5;
  float gh3[kMaxDense] = {};
  for (int o = 0; o < 5; ++o) {
    gbo[o] += dzo[o];
    for (int j = 0; j < L3; ++j) { gWo[o * L3 + j] += dzo[o] * h3[j]; gh3[j] += dzo[o] * wo[o * L3 + j]; }
  }
  float gz3[kMaxDense], gh2[kMaxDense] = {};
  for (int k = 0; k < L3; ++k) gz3[k] = (z3[k] > 0.f && z3[k] < 1.f) ? gh3[k] : 0.f;
  for (int k = 0; k < L3; ++k) {
    if (gz3[k] == 0.f) continue;
    g[L.b3 + k] += gz3[k];
    float* gw = g + L.W3 + size_t(k) * L2;
    const float* w = &net.W3[size_t(k) * L2];
    for (int j = 0; j < L2; ++j) { gw[j] += gz3[k] * h2[j]; gh2[j] += gz3[k] * w[j]; }
  }
  float gz2[kMaxDense];
  for (int k = 0; k < L2; ++k) gz2[k] = (z2[k] > 0.f && z2[k] < 1.f) ? gh2[k] : 0.f;
  float gx[2 * kMaxHidden + 16] = {};
  for (int k = 0; k < L2; ++k) {
    if (gz2[k] == 0.f) continue;
    g[L.b2 + k] += gz2[k];
    float* gw = g + L.W2 + size_t(k) * in;
    const float* w = &net.W2[size_t(k) * in];
    const float a = gz2[k];
    for (int j = 0; j < in; ++j) { gw[j] += a * x[j]; gx[j] += a * w[j]; }
  }
  for (int p = 0; p < 2; ++p) {
    float gacc[kMaxHidden];
    for (int j = 0; j < H; ++j) gacc[j] = (acc[p][j] > 0.f && acc[p][j] < 1.f) ? gx[p * H + j] : 0.f;
    for (int j = 0; j < H; ++j) g[L.b1 + j] += gacc[j];
    for (int i = 0; i < nf[p]; ++i) {
      float* gw = g + L.W1 + size_t(feats[p][i]) * H;
      for (int j = 0; j < H; ++j) gw[j] += gacc[j];
    }
  }
}

struct Args {
  std::vector<std::string> pos;
  int hidden = 256, l2 = 32, l3 = 32, epochs = 12, batch = 2048, threads = 4, min_plies = -1;
  float lr = 1e-3f, val = 0.01f;
  std::string init;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (s == "--hidden") a.hidden = std::stoi(next());
    else if (s == "--l2") a.l2 = std::stoi(next());
    else if (s == "--l3") a.l3 = std::stoi(next());
    else if (s == "--epochs") a.epochs = std::stoi(next());
    else if (s == "--batch") a.batch = std::stoi(next());
    else if (s == "--threads") a.threads = std::stoi(next());
    else if (s == "--lr") a.lr = std::stof(next());
    else if (s == "--val") a.val = std::stof(next());
    else if (s == "--init") a.init = next();
    else if (s == "--min-plies") a.min_plies = std::stoi(next());
    else a.pos.push_back(s);
  }
  return a;
}

void validate(Network& net, const std::vector<Record>& val, const char* tag) {
  net.prepare();
  double se[5] = {}, all = 0;
  long n[5] = {};
  for (const Record& r : val) {
    Accumulator acc;
    acc_refresh(net, r.board, acc);
    float extra[kExtraFeatures];
    extra_features(r.board, extra);
    Probs p = net_forward(net, acc, 0, extra, head_for(PosClass(r.cls)));
    const float* t = r.target;
    float et = 2 * t[0] - 1 + t[1] - t[3] + t[2] - t[4];
    double d = p.equity() - et;
    se[r.cls] += d * d; n[r.cls]++; all += d * d;
  }
  std::printf("  [%s] val equity rmse %.4f", tag, std::sqrt(all / std::max<size_t>(1, val.size())));
  const char* names[5] = {"over", "bearoff", "race", "contact", "crashed"};
  for (int c = 2; c < 5; ++c)
    if (n[c]) std::printf("  %s %.4f", names[c], std::sqrt(se[c] / n[c]));
  std::printf("\n");
  std::fflush(stdout);
}

int cmd_train(const Args& a) {
  if (a.pos.size() < 2) { std::fprintf(stderr, "train <data.bin> <out.bgnn>\n"); return 1; }
  std::vector<Record> recs;
  if (!load_records(a.pos[0], recs)) { std::fprintf(stderr, "cannot load %s\n", a.pos[0].c_str()); return 1; }
  std::mt19937_64 rng(12345);
  std::shuffle(recs.begin(), recs.end(), rng);
  size_t nval = std::min<size_t>(size_t(recs.size() * a.val), 200000);
  std::vector<Record> val(recs.end() - nval, recs.end());
  recs.resize(recs.size() - nval);
  std::printf("train %zu  val %zu\n", recs.size(), val.size());

  Network net;
  if (!a.init.empty()) {
    std::string err;
    if (!net.load(a.init, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  } else {
    net.init(a.hidden, a.l2, a.l3, 1);
  }
  const Layout L(net);
  auto szs = sizes(net);
  std::vector<float> m(L.total, 0.f), v(L.total, 0.f);
  const int T = std::max(1, a.threads);
  std::vector<std::vector<float>> grads(T, std::vector<float>(L.total));
  const size_t steps_per_epoch = recs.size() / a.batch;
  const size_t total_steps = steps_per_epoch * a.epochs;
  const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  size_t step = 0;
  validate(net, val, "init");
  auto t0 = std::chrono::steady_clock::now();
  for (int ep = 0; ep < a.epochs; ++ep) {
    std::shuffle(recs.begin(), recs.end(), rng);
    Stats ep_st;
    for (size_t bi = 0; bi < steps_per_epoch; ++bi, ++step) {
      const size_t base = bi * a.batch;
      std::vector<Stats> st(T);
      std::vector<std::thread> th;
      for (int t = 0; t < T; ++t) {
        th.emplace_back([&, t]() {
          float* g = grads[t].data();
          std::memset(g, 0, sizeof(float) * L.total);
          for (size_t i = base + t; i < base + a.batch; i += T) sample_grad(net, L, recs[i], g, st[t]);
        });
      }
      for (auto& x : th) x.join();
      // Reduce gradients in parallel chunks.
      th.clear();
      for (int t = 0; t < T; ++t) {
        th.emplace_back([&, t]() {
          size_t lo = L.total * t / T, hi = L.total * (t + 1) / T;
          float* g0 = grads[0].data();
          for (int u = 1; u < T; ++u) {
            const float* gu = grads[u].data();
            for (size_t i = lo; i < hi; ++i) g0[i] += gu[i];
          }
        });
      }
      for (auto& x : th) x.join();
      for (auto& s : st) { ep_st.loss += s.loss; ep_st.eq_se += s.eq_se; ep_st.n += s.n; }
      // Adam with warmup + cosine decay.
      float frac = float(step) / float(std::max<size_t>(1, total_steps));
      float lr = a.lr * 0.5f * (1.f + std::cos(3.14159265f * frac));
      if (step < 500) lr *= float(step + 1) / 500.f;
      const float c1 = 1.f - std::pow(b1, float(step + 1)), c2 = 1.f - std::pow(b2, float(step + 1));
      const float scale = 1.f / a.batch;
      auto params = tensors(net);
      size_t off = 0;
      for (size_t k = 0; k < params.size(); ++k) {
        float* p = params[k];
        const float* g = grads[0].data() + off;
        float* mk = m.data() + off;
        float* vk = v.data() + off;
        for (size_t i = 0; i < szs[k]; ++i) {
          float gi = g[i] * scale;
          mk[i] = b1 * mk[i] + (1 - b1) * gi;
          vk[i] = b2 * vk[i] + (1 - b2) * gi * gi;
          p[i] -= lr * (mk[i] / c1) / (std::sqrt(vk[i] / c2) + eps);
        }
        off += szs[k];
      }
      if (bi % 1000 == 999) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  ep %d step %zu/%zu  loss %.4f  eq rmse %.4f  lr %.2e  %.0f samples/s\n", ep + 1, bi + 1,
                    steps_per_epoch, ep_st.loss / ep_st.n, std::sqrt(ep_st.eq_se / ep_st.n), lr,
                    double(step + 1) * a.batch / el);
        std::fflush(stdout);
      }
    }
    std::printf("epoch %d  train loss %.4f  eq rmse %.4f\n", ep + 1, ep_st.loss / std::max(1L, ep_st.n),
                std::sqrt(ep_st.eq_se / std::max(1L, ep_st.n)));
    char tag[32];
    std::snprintf(tag, sizeof tag, "ep %d", ep + 1);
    validate(net, val, tag);
    net.id = "dbgh-h" + std::to_string(net.H) + "-ep" + std::to_string(ep + 1);
    net.prepare();
    net.save(a.pos[1]);
  }
  return 0;
}

// Expert iteration: relabel positions with our own (deeper) search.
//   bgtrain relabel <out.txt> --net nets/latest.bgnn --plies 3 --threads 16 --max N <label files...>
int cmd_relabel(const Args& a, const std::string& net_path, int plies, size_t max_n) {
  if (a.pos.size() < 2) { std::fprintf(stderr, "relabel <out.txt> <label files...>\n"); return 1; }
  Evaluator ev;
  std::string err;
  if (!ev.init(net_path, "data", &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  std::vector<std::string> files(a.pos.begin() + 1, a.pos.end());
  std::vector<Record> recs = load_labels(files, -1);
  std::mt19937_64 rng(777);
  std::shuffle(recs.begin(), recs.end(), rng);
  if (max_n && recs.size() > max_n) recs.resize(max_n);
  std::printf("relabelling %zu positions at %d-ply with %s\n", recs.size(), plies, ev.network().id.c_str());
  FILE* out = std::fopen(a.pos[0].c_str(), "a");
  if (!out) return 1;
  std::mutex mu;
  std::atomic<size_t> next{0}, done{0};
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for (int t = 0; t < std::max(1, a.threads); ++t)
    th.emplace_back([&]() {
      Searcher s(ev, 1, 64);
      MatchState ms;
      std::string buf;
      for (size_t i; (i = next.fetch_add(1)) < recs.size();) {
        Probs p = s.evaluate(recs[i].board, ms, plies);
        char line[160];
        std::snprintf(line, sizeof line, "%s %d %.6f %.6f %.6f %.6f %.6f\n", position_id(recs[i].board).c_str(),
                      100 + plies, p[0], p[1], p[2], p[3], p[4]);
        buf += line;
        size_t d = done.fetch_add(1) + 1;
        if (buf.size() > 16384 || d == recs.size()) {
          std::lock_guard<std::mutex> g(mu);
          std::fputs(buf.c_str(), out);
          std::fflush(out);
          buf.clear();
        }
        if (d % 20000 == 0) {
          double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          std::printf("  %zu / %zu  (%.0f pos/s)\n", d, recs.size(), d / el);
          std::fflush(stdout);
        }
      }
      std::lock_guard<std::mutex> g(mu);
      std::fputs(buf.c_str(), out);
    });
  for (auto& x : th) x.join();
  std::fclose(out);
  return 0;
}

int cmd_build(const Args& a) {
  if (a.pos.size() < 2) { std::fprintf(stderr, "build <out.bin> files...\n"); return 1; }
  std::vector<std::string> files(a.pos.begin() + 1, a.pos.end());
  size_t parsed = 0;
  auto recs = load_labels(files, a.min_plies, &parsed);
  long cls[5] = {}, pl[8] = {};
  for (auto& r : recs) { cls[r.cls]++; if (r.plies >= 0 && r.plies < 8) pl[r.plies]++; }
  std::printf("parsed %zu lines -> %zu unique records (race %ld contact %ld crashed %ld; 0-ply %ld 1-ply %ld 2-ply %ld)\n",
              parsed, recs.size(), cls[2], cls[3], cls[4], pl[0], pl[1], pl[2]);
  return save_records(a.pos[0], recs) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bgtrain build|train ...\n");
    return 1;
  }
  std::string cmd = argv[1];
  Args a = parse(argc, argv);
  if (cmd == "build") return cmd_build(a);
  if (cmd == "train") return cmd_train(a);
  if (cmd == "relabel") {
    std::string net = "nets/latest.bgnn";
    int plies = 3;
    size_t max_n = 0;
    Args b = a;
    b.pos.clear();
    for (size_t i = 0; i < a.pos.size(); ++i) {
      if (a.pos[i] == "--net" && i + 1 < a.pos.size()) net = a.pos[++i];
      else if (a.pos[i] == "--plies" && i + 1 < a.pos.size()) plies = std::stoi(a.pos[++i]);
      else if (a.pos[i] == "--max" && i + 1 < a.pos.size()) max_n = std::stoul(a.pos[++i]);
      else b.pos.push_back(a.pos[i]);
    }
    return cmd_relabel(b, net, plies, max_n);
  }
  std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 1;
}
