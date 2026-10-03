// bgselfplay: the engine plays against itself (with the cube) and writes the
// games in Jellyfish .mat format, as exported by BGBlitz/gnubg/XG.
//
//   bgselfplay [--match 7 | --games 10] [--plies 3] [--cube-extra 1] [--net nets/latest.bgnn]
//              [--threads 4] [--seed N] [--name1 X] [--name2 Y] [--out file.mat] [--no-jacoby]
//
// --match L plays one L-point match (Crawford rule); --games N plays N money
// games (shown as a "0 point match", like BGBlitz does).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

#include "core/movegen.hpp"
#include "eval/cube.hpp"
#include "eval/evaluator.hpp"
#include "search/search.hpp"

using namespace bg;

namespace {

// One checker step per entry, highest starting point first (the order BGBlitz,
// gnubg and XG use): "25/20* 20/15 13/8"; bar = 25, off = 0. Hits are marked on
// the first step that lands on the blot, replaying the steps on `before`.
std::string mat_move(const Board& before, const Move& m) {
  int idx[4] = {0, 1, 2, 3};
  std::sort(idx, idx + m.n, [&](int x, int y) {
    if (m.from[x] != m.from[y]) return m.from[x] > m.from[y];
    return m.to[x] > m.to[y];
  });
  u128 opp = before.side[1];
  std::string s;
  for (int k = 0; k < m.n; ++k) {
    const int i = idx[k];
    if (!s.empty()) s += ' ';
    const int from = m.from[i] == kBar ? 25 : m.from[i] + 1;
    const int to = m.to[i] < 0 ? 0 : m.to[i] + 1;
    s += std::to_string(from) + "/" + std::to_string(to);
    if (m.to[i] >= 0 && nibble(opp, 23 - m.to[i]) == 1) {
      opp -= nibble_bit(23 - m.to[i]);
      s += '*';
    }
  }
  return s;
}

std::string pad(std::string s, size_t w) {
  if (s.size() < w) s.append(w - s.size(), ' ');
  return s;
}

// Collects the events of a game and lays them out in two columns.
struct GameLog {
  struct Line { std::string left, right; bool has_left = false, has_right = false; };
  std::vector<Line> lines;
  void add(int player, const std::string& text) {
    if (lines.empty() || lines.back().has_right || (player == 0 && lines.back().has_left)) lines.push_back({});
    Line& l = lines.back();
    if (player == 0) { l.left = text; l.has_left = true; }
    else { l.right = text; l.has_right = true; }
  }
  void write(FILE* f) const {
    int n = 0;
    for (const Line& l : lines)
      std::fprintf(f, "%3d) %s%s\n", ++n, pad(l.left, 28).c_str(), pad(l.right, 28).c_str());
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string net = "nets/latest.bgnn", out = "", name[2] = {"DarkBGHelmet_A", "DarkBGHelmet_B"};
  int plies = 3, cube_extra = 1, match = 0, games = 10, threads = 4;
  bool jacoby = true;
  unsigned seed = unsigned(std::chrono::system_clock::now().time_since_epoch().count());
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--net") net = next();
    else if (a == "--plies") plies = std::stoi(next());
    else if (a == "--cube-extra") cube_extra = std::max(0, std::stoi(next()));
    else if (a == "--match") match = std::stoi(next());
    else if (a == "--games") games = std::stoi(next());
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--seed") seed = unsigned(std::stoul(next()));
    else if (a == "--name1") name[0] = next();
    else if (a == "--name2") name[1] = next();
    else if (a == "--out") out = next();
    else if (a == "--no-jacoby") jacoby = false;
    else {
      std::fprintf(stderr,
                   "usage: bgselfplay [--match L | --games N] [--plies n] [--cube-extra n] [--net file] [--threads n]\n"
                   "                  [--seed n] [--name1 s] [--name2 s] [--out file.mat] [--no-jacoby]\n");
      return 1;
    }
  }
  if (match < 0 || match > kMaxMatch) { std::fprintf(stderr, "match length must be 1..25\n"); return 1; }
  Evaluator ev;
  std::string err;
  if (!ev.init(net, "data", &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  Searcher search(ev, threads);
  const SearchConfig cfg = SearchConfig::for_plies(plies);
  const SearchConfig cube_cfg = SearchConfig::for_plies(std::min(7, plies + cube_extra));
  std::mt19937 rng(seed);

  if (out.empty()) {
    char buf[64];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof buf, "%Y-%m-%d_%H%M%S", std::localtime(&t));
    out = std::string("selfplay_") + (match ? std::to_string(match) + "pt_" : "money_") + buf + ".mat";
  }
  FILE* f = std::fopen(out.c_str(), "w");
  if (!f) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return 1; }
  std::fprintf(f, "; [Crawford \"On\"]\n; [Jacoby \"%s\"]\n; [Beaver \"Off\"]\n", match == 0 && jacoby ? "On" : "Off");
  std::fprintf(f, " %d point match\n\n", match);

  int score[2] = {0, 0};
  bool crawford_done = false;
  const auto t0 = std::chrono::steady_clock::now();
  for (int game = 1;; ++game) {
    if (match == 0 && game > games) break;
    // Match state from player 0's view.
    MatchState ms;
    ms.length = match;
    ms.score[0] = score[0];
    ms.score[1] = score[1];
    ms.jacoby = match == 0 && jacoby;
    ms.beavers = false;
    ms.crawford = match > 0 && !crawford_done && (match - score[0] == 1 || match - score[1] == 1);
    if (ms.crawford) crawford_done = true;

    std::fprintf(f, " Game %d\n %s : %d", game, name[0].c_str(), score[0]);
    std::fprintf(f, "%*s%s : %d\n", int(std::max<size_t>(1, 28 - name[0].size() - 3 - std::to_string(score[0]).size())), "",
                 name[1].c_str(), score[1]);

    GameLog log;
    int d1, d2;
    do {
      d1 = int(rng() % 6) + 1;
      d2 = int(rng() % 6) + 1;
    } while (d1 == d2);
    int p = d1 > d2 ? 0 : 1;  // player 0's die vs player 1's die
    Board b = Board::start();  // side 0 = player to move (p)
    bool opening = true, turned = false;
    int winner = -1, points = 0;
    while (winner < 0) {
      MatchState mp = p == 0 ? ms : ms.swapped();
      // Cube action.
      if (!opening && mp.owner != CubeOwner::Opp && CubeModel(mp).can_double(0, mp.cube)) {
        CubeResult r = search.analyze_cube(b, mp, cube_cfg);
        const CubeDecision d = r.analysis.decision;
        if (d == CubeDecision::DoubleTake || d == CubeDecision::DoublePass) {
          log.add(p, " Doubles => " + std::to_string(2 * ms.cube));
          if (!r.analysis.take) {
            log.add(1 - p, " Drops");
            winner = p;
            points = ms.cube;
            break;
          }
          log.add(1 - p, " Takes");
          ms.cube *= 2;
          ms.owner = p == 0 ? CubeOwner::Opp : CubeOwner::Me;
          turned = true;
          mp = p == 0 ? ms : ms.swapped();
        }
      }
      int a, c;
      if (opening) { a = std::max(d1, d2); c = std::min(d1, d2); opening = false; }
      else { a = int(rng() % 6) + 1; c = int(rng() % 6) + 1; }
      MoveAnalysis an = search.analyze_moves(b, a, c, mp, cfg);
      const Move& m = an.moves[0].move;
      const int hi = std::max(a, c), lo = std::min(a, c);
      std::string text = std::to_string(hi) + std::to_string(lo) + ":";
      if (m.n > 0) text += " " + mat_move(b, m);
      log.add(p, text);
      if (m.after.side[0] == 0) {
        winner = p;
        const int k = int(m.after.result_for_winner(0));
        points = (ms.jacoby && !turned ? 1 : k) * ms.cube;
        break;
      }
      b = m.after.swapped();
      p = 1 - p;
    }
    log.write(f);
    score[winner] += points;
    const bool match_over = match > 0 && score[winner] >= match;
    std::fprintf(f, "%*s Wins %d point%s%s\n\n", winner == 0 ? 5 : 33, "", points, points == 1 ? "" : "s",
                 match_over ? " and the match" : "");
    std::fflush(f);
    std::printf("game %d: %s wins %d  (score %d-%d, %.0fs)\n", game, name[winner].c_str(), points, score[0], score[1],
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fflush(stdout);
    if (match_over) break;
  }
  std::fclose(f);
  std::printf("written %s\n", out.c_str());
  return 0;
}
