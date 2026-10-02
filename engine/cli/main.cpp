// bgcli: developer command line.
//   bgcli moves <posid|xgid> <d1d2>     list legal plays
//   bgcli perft <depth>                  perft from the start position
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/ids.hpp"
#include "core/movegen.hpp"
#include "eval/evaluator.hpp"
#include <cmath>

using namespace bg;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: bgcli moves <id> <dice> | perft <depth>\n");
    return 1;
  }
  std::string cmd = argv[1];
  if (cmd == "moves" && argc >= 4) {
    Position p;
    std::string err;
    if (!parse_any(argv[2], p, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    std::vector<Move> moves;
    generate_moves(p.board, argv[3][0] - '0', argv[3][1] - '0', moves);
    for (const Move& m : moves) std::printf("%s\t%s\n", m.str().c_str(), position_id(m.after.swapped()).c_str());
    return 0;
  }
  if (cmd == "randpos" && argc >= 3) {
    // Random positions from random play: "<posid> <d1><d2>" per line.
    int n = std::atoi(argv[2]);
    u64 seed = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : 1;
    std::mt19937_64 rng(seed);
    std::vector<Move> moves;
    int printed = 0;
    while (printed < n) {
      Board b = Board::start();
      int plies = int(rng() % 80);
      for (int i = 0; i < plies && b.winner() < 0; ++i) {
        generate_moves(b, int(rng() % 6) + 1, int(rng() % 6) + 1, moves);
        b = moves[rng() % moves.size()].after.swapped();
      }
      if (b.winner() >= 0) continue;
      std::printf("%s %d%d\n", position_id(b).c_str(), int(rng() % 6) + 1, int(rng() % 6) + 1);
      ++printed;
    }
    return 0;
  }
  if (cmd == "checkmoves") {
    // Reads "RESULT <posid> <dice> <n>\t<m1>|<m2>|..." (gnubg_movelist.py) from stdin.
    char line[1 << 16];
    int ok = 0, bad = 0;
    std::vector<Move> moves;
    while (std::fgets(line, sizeof line, stdin)) {
      if (std::strncmp(line, "RESULT ", 7)) continue;
      char pid[32], dice[8];
      int n;
      if (std::sscanf(line + 7, "%31s %7s %d", pid, dice, &n) != 3) continue;
      Board b;
      if (!parse_position_id(pid, b)) { ++bad; continue; }
      generate_moves(b, dice[0] - '0', dice[1] - '0', moves);
      std::string rest = std::strchr(line, '\t') ? std::strchr(line, '\t') + 1 : "";
      while (!rest.empty() && (rest.back() == '\n' || rest.back() == '\r')) rest.pop_back();
      std::vector<std::string> gm;
      size_t a = 0;
      while (a < rest.size()) {
        size_t e = rest.find('|', a);
        if (e == std::string::npos) e = rest.size();
        gm.push_back(rest.substr(a, e - a));
        a = e + 1;
      }
      bool good = true;
      std::vector<bool> hit(moves.size(), false);
      int ours = (moves.size() == 1 && moves[0].n == 0) ? 0 : int(moves.size());
      if (ours != n) good = false;
      for (const auto& s : gm) {
        int i = find_move(moves, b, s);
        if (i < 0 || hit[i]) { good = false; std::printf("  unmatched gnubg move '%s'\n", s.c_str()); }
        else hit[i] = true;
      }
      if (good) ++ok;
      else {
        ++bad;
        std::printf("MISMATCH %s %s gnubg=%d ours=%d\n", pid, dice, n, ours);
        for (const Move& m : moves) std::printf("  ours: %s\n", m.str().c_str());
      }
    }
    std::printf("checked: %d ok, %d mismatches\n", ok, bad);
    return bad ? 2 : 0;
  }
  if (cmd == "compare") {
    // Reads "<posid> <plies> p0..p4" label lines on stdin, compares the static
    // evaluation: bgcli compare [net] < labels
    Evaluator ev;
    std::string err;
    if (!ev.init(argc >= 3 ? argv[2] : "", "data", &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    char pid[32];
    int plies;
    float q[5];
    double se = 0, ae = 0, aw = 0;
    long n = 0;
    double cls_se[5] = {}; long cls_n[5] = {};
    while (std::scanf("%31s %d %f %f %f %f %f", pid, &plies, &q[0], &q[1], &q[2], &q[3], &q[4]) == 7) {
      Board b;
      if (!parse_position_id(pid, b)) continue;
      Probs p = ev.evaluate(b);
      Probs t;
      for (int i = 0; i < 5; ++i) t[i] = q[i];
      double d = p.equity() - t.equity();
      se += d * d; ae += std::fabs(d); aw += std::fabs(p[kWin] - t[kWin]); ++n;
      int c = int(classify(b));
      cls_se[c] += d * d; cls_n[c]++;
    }
    if (!n) return 1;
    std::printf("n=%ld  equity rmse %.4f  mae %.4f  win mae %.4f\n", n, std::sqrt(se / n), ae / n, aw / n);
    const char* names[5] = {"over", "bearoff", "race", "contact", "crashed"};
    for (int c = 0; c < 5; ++c)
      if (cls_n[c]) std::printf("  %-8s n=%-8ld rmse %.4f\n", names[c], cls_n[c], std::sqrt(cls_se[c] / cls_n[c]));
    return 0;
  }
  if (cmd == "eval" && argc >= 3) {
    Position p;
    std::string err;
    if (!parse_any(argv[2], p, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Evaluator ev;
    if (!ev.init(argc >= 4 ? argv[3] : "", "data", &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Probs r = ev.evaluate(p.board);
    std::printf("%s", p.board.ascii().c_str());
    std::printf("win %.4f wg %.4f wbg %.4f lg %.4f lbg %.4f  equity %+.4f\n", r[0], r[1], r[2], r[3], r[4], r.equity());
    return 0;
  }
  if (cmd == "perft" && argc >= 3) {
    int depth = std::atoi(argv[2]);
    auto t0 = std::chrono::steady_clock::now();
    u64 n = perft(Board::start(), depth);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("perft %d = %llu  (%.3f s, %.1f M/s)\n", depth, (unsigned long long)n, s, n / s / 1e6);
    return 0;
  }
  std::fprintf(stderr, "unknown command\n");
  return 1;
}
