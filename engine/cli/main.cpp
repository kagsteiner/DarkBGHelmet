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
#include <map>
#include "eval/cube.hpp"
#include "search/search.hpp"

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
  if (cmd == "checkcube") {
    // Compares our cube model with gnubg (tools/gnubg_cubecheck.py output on stdin).
    char pid[32];
    int len, s0, s1, cube, owner, crw;
    float q[5], gnd, gdt, gdp;
    struct Acc { double nd = 0, dt = 0; long n = 0, agree = 0; };
    std::map<std::string, Acc> by;
    while (std::scanf(" C %31s %d %d %d %d %d %d %f %f %f %f %f %f %f %f", pid, &len, &s0, &s1, &cube, &owner, &crw,
                      &q[0], &q[1], &q[2], &q[3], &q[4], &gnd, &gdt, &gdp) == 15) {
      Board b;
      if (!parse_position_id(pid, b)) continue;
      MatchState ms;
      ms.length = len;
      ms.score[0] = s0;
      ms.score[1] = s1;
      ms.cube = cube;
      ms.owner = owner < 0 ? CubeOwner::Center : (owner == 0 ? CubeOwner::Me : CubeOwner::Opp);
      ms.crawford = crw != 0;
      ms.beavers = false;
      Probs p;
      for (int i = 0; i < 5; ++i) p[i] = q[i];
      CubeAnalysis a = analyze_cube(p, ms, cube_efficiency(b));
      CubeModel cm(ms);
      // gnubg reports the optimal value (max of ND and the doubled branch), not ND.
      double dnd = cm.to_emg(a.can_double ? std::max(a.nd, std::min(a.dt, a.dp)) : a.nd) - gnd;
      double ddt = cm.to_emg(a.dt) - gdt;
      // gnubg decision from its numbers with the same rule
      auto decide = [](float nd, float dt, float dp) {
        float d = std::min(dt, dp);
        if (d > nd) return dt <= dp ? 1 : 2;
        if (nd > dp) return 3;
        return 0;
      };
      int ours = a.can_double ? decide(cm.to_emg(a.nd), cm.to_emg(a.dt), cm.to_emg(a.dp)) : -1;
      // From gnubg's optimal value: doubled if it equals min(DT,DP).
      int theirs = -1;
      if (a.can_double) {
        float d = std::min(gdt, gdp);
        bool dbl = std::fabs(gnd - d) < 1e-5f && gnd >= 0.f - 10.f;
        theirs = dbl ? (gdt <= gdp ? 1 : 2) : (ours == 3 ? 3 : 0);
        if (ours == 3 && !dbl && gnd <= gdp) theirs = 0;
      }
      if (ours == 3) ours = 0;  // gnubg's optimal value cannot distinguish too-good from no double
      if (theirs == 3) theirs = 0;
      char key[64];
      std::snprintf(key, sizeof key, "len %2d score %2d-%2d cube %d owner %2d crw %d", len, s0, s1, cube, owner, crw);
      if (std::getenv("BG_DEBUG") && std::fabs(ddt) > 0.04)
        std::printf("DBG %s %s cls %d x %.3f pips %d/%d w %.4f ourDT %.4f gnubgDT %.4f\n", key, pid, int(classify(b)),
                    cube_efficiency(b), b.pips(0), b.pips(1), p[0], cm.to_emg(a.dt), gdt);
      Acc& ac = by[key];
      ac.nd += dnd * dnd; ac.dt += ddt * ddt; ac.n++; ac.agree += ours == theirs;
    }
    for (auto& [k, ac] : by)
      std::printf("%s  n=%ld  OPT rmse %.4f  DT rmse %.4f  decision agree %.1f%%\n", k.c_str(), ac.n,
                  std::sqrt(ac.nd / ac.n), std::sqrt(ac.dt / ac.n), 100.0 * ac.agree / ac.n);
    return 0;
  }
  if ((cmd == "move" || cmd == "cube") && argc >= 3) {
    // bgcli move <id> <dice> [plies] [net] [threads] | bgcli cube <id> [plies] [net] [threads]
    Position p;
    std::string err;
    if (!parse_any(argv[2], p, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    int ai = 3;
    int d1 = 0, d2 = 0;
    if (cmd == "move") {
      if (argc < 4) return 1;
      d1 = argv[3][0] - '0';
      d2 = argv[3][1] - '0';
      ai = 4;
    }
    int plies = argc > ai ? std::atoi(argv[ai]) : 2;
    std::string net = argc > ai + 1 ? argv[ai + 1] : "nets/latest.bgnn";
    int threads = argc > ai + 2 ? std::atoi(argv[ai + 2]) : 0;
    Evaluator ev;
    if (!ev.init(net, "data", &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    Searcher s(ev, threads);
    SearchConfig cfg = SearchConfig::for_plies(plies);
    if (const char* e = std::getenv("BG_SAMPLED")) cfg.sampled_rolls = std::atoi(e);
    if (const char* e = std::getenv("BG_FULL")) cfg.full_layers = std::atoi(e);
    std::printf("%s", p.board.ascii().c_str());
    if (cmd == "move") {
      MoveAnalysis a = s.analyze_moves(p.board, d1, d2, p.match, cfg);
      for (size_t i = 0; i < a.moves.size() && i < 12; ++i) {
        const MoveEval& m = a.moves[i];
        std::printf("%2zu. %-24s %d-ply  emg %+.4f (%+.4f)  w %.3f g %.3f bg %.3f | l %.3f g %.3f bg %.3f\n", i + 1,
                    m.move.str().c_str(), m.plies, m.emg, m.emg - a.moves[0].emg, m.probs[0], m.probs[1], m.probs[2],
                    1 - m.probs[0], m.probs[3], m.probs[4]);
      }
      std::printf("%llu nodes, %.3f s\n", (unsigned long long)a.nodes, a.seconds);
    } else {
      CubeResult r = s.analyze_cube(p.board, p.match, cfg);
      const CubeAnalysis& a = r.analysis;
      std::printf("%d-ply  w %.3f g %.3f bg %.3f | l %.3f g %.3f bg %.3f\n", r.plies, a.probs[0], a.probs[1],
                  a.probs[2], 1 - a.probs[0], a.probs[3], a.probs[4]);
      std::printf("No double %+.4f  Double/take %+.4f  Double/pass %+.4f   => %s\n", r.nd_emg, r.dt_emg, r.dp_emg,
                  to_string(a.decision, p.match.owner == CubeOwner::Me));
      std::printf("%llu nodes, %.3f s\n", (unsigned long long)r.nodes, r.seconds);
    }
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
