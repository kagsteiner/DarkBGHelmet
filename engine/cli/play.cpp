// bgplay: play against the engine in a terminal.
//
//   bgplay [--net nets/latest.bgnn] [--plies 3] [--cube-extra 2] [--match 7] [--dice manual] [--threads 4]
//   --cube-extra n: cube decisions search n plies deeper than checker play (0 = off)
//
// You are X (moving from your 24 point down to your 1 point). Moves use gnubg
// notation from the mover's own side ("13/7 8/7", "bar/22*", "6/off(2)"); the
// engine's moves are printed in its own numbering, as gnubg and XG do.
#include <cstdio>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/ids.hpp"
#include "core/movegen.hpp"
#include "eval/cube.hpp"
#include "eval/evaluator.hpp"
#include "search/search.hpp"

using namespace bg;

namespace {

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string ask(const std::string& prompt) {
  std::cout << prompt << std::flush;
  std::string line;
  if (!std::getline(std::cin, line)) std::exit(0);
  return line;
}

// Board printed from the human's view: human = X at the bottom.
void show(const Board& human_view, const MatchState& ms_human, int cube) {
  std::cout << "\n" << human_view.ascii();
  std::cout << "Cube " << cube << " ("
            << (ms_human.owner == CubeOwner::Center ? "centered" : ms_human.owner == CubeOwner::Me ? "yours" : "engine's")
            << ")";
  if (!ms_human.money())
    std::cout << "   Match to " << ms_human.length << ": you " << ms_human.score[0] << ", engine " << ms_human.score[1]
              << (ms_human.crawford ? "  (Crawford game)" : "");
  Position p{human_view, ms_human, {0, 0}};
  std::cout << "\n" << position_id(human_view) << ":" << match_id(ms_human) << "   " << to_xgid(p) << "\n";
}

// Rewrites a move from the mover's numbering into the opponent's (point p -> 25 - p).
std::string mirror(const std::string& m) {
  std::string out, num;
  auto flush = [&]() {
    if (!num.empty()) {
      int v = std::stoi(num);
      out += (v >= 1 && v <= 24) ? std::to_string(25 - v) : num;
      num.clear();
    }
  };
  for (size_t i = 0; i < m.size(); ++i) {
    char c = m[i];
    bool in_paren = !out.empty() && out.back() == '(';
    if (std::isdigit(static_cast<unsigned char>(c)) && !in_paren) num += c;
    else { flush(); out += c; }
  }
  flush();
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string net = "nets/latest.bgnn";
  int plies = 3, match = 0, threads = 4, cube_extra = 2;
  bool manual_dice = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--net") net = next();
    else if (a == "--plies") plies = std::stoi(next());
    else if (a == "--cube-extra") cube_extra = std::max(0, std::stoi(next()));
    else if (a == "--match") match = std::stoi(next());
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--dice") manual_dice = next() == "manual";
    else {
      std::fprintf(stderr, "usage: bgplay [--net file] [--plies n] [--cube-extra n] [--match n] [--dice manual|auto] [--threads n]\n");
      return 1;
    }
  }
  Evaluator ev;
  std::string err;
  if (!ev.init(net, "data", &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  Searcher search(ev, threads);
  const SearchConfig cfg = SearchConfig::for_plies(plies);
  const SearchConfig cube_cfg = SearchConfig::for_plies(std::min(7, plies + cube_extra));
  std::mt19937 rng(std::random_device{}());

  std::cout << "DarkBGHelmet (" << ev.network().id << ", " << plies << "-ply checker play, "
            << std::min(7, plies + cube_extra) << "-ply cube decisions). You are X.\n"
            << "Commands on your turn: roll / double (before rolling), a move like '13/7 8/7', 'moves' (list legal),\n"
            << "'hint' (engine's suggestion), 'resign', 'quit'.\n";

  int score_h = 0, score_e = 0;
  bool crawford_done = false;
  for (int game = 1;; ++game) {
    // Match state from the human's view; the engine's view is swapped().
    MatchState msh;
    msh.length = match;
    msh.score[0] = score_h;
    msh.score[1] = score_e;
    msh.jacoby = match == 0;
    msh.beavers = false;
    msh.crawford = match > 0 && !crawford_done && (match - score_h == 1 || match - score_e == 1);
    if (msh.crawford) crawford_done = true;
    Board hb = Board::start();  // human's view (X = human = side 0)
    // Opening roll decides who starts.
    int d1, d2;
    do {
      d1 = int(rng() % 6) + 1;
      d2 = int(rng() % 6) + 1;
    } while (d1 == d2 && !manual_dice);
    bool human_turn;
    if (manual_dice) {
      std::string s = ask("Opening roll: who starts? (me/engine): ");
      human_turn = lower(s).rfind("m", 0) == 0;
      d1 = d2 = 0;
    } else {
      human_turn = d1 > d2;
      std::cout << "\n=== Game " << game << " === Opening roll: you " << d1 << ", engine " << d2 << " -> "
                << (human_turn ? "you start" : "engine starts") << " with " << std::max(d1, d2) << std::min(d1, d2)
                << "\n";
    }
    bool opening = !manual_dice;
    int winner = -1, points = 0;  // winner 0 = human, 1 = engine
    while (winner < 0) {
      auto roll = [&](int& a, int& b) {
        if (opening) { a = std::max(d1, d2); b = std::min(d1, d2); opening = false; return; }
        if (manual_dice) {
          for (;;) {
            std::string s = ask("Enter dice (e.g. 52): ");
            if (s.size() >= 2 && s[0] >= '1' && s[0] <= '6' && s[1] >= '1' && s[1] <= '6') { a = s[0] - '0'; b = s[1] - '0'; return; }
          }
        }
        a = int(rng() % 6) + 1;
        b = int(rng() % 6) + 1;
      };
      if (human_turn) {
        show(hb, msh, msh.cube);
        bool rolled = opening;
        while (!rolled) {
          bool can_double = msh.owner != CubeOwner::Opp && CubeModel(msh).can_double(0, msh.cube);
          std::string c = lower(ask(can_double ? "Your turn: roll or double? " : "Your turn (press Enter to roll): "));
          if (c == "quit") return 0;
          if (c == "resign") { winner = 1; points = msh.cube; break; }
          if (can_double && (c == "double" || c == "d")) {
            CubeResult r = search.analyze_cube(hb, msh, cube_cfg);  // from your view; 'take' = engine's answer
            if (!r.analysis.take) {
              std::cout << "Engine passes.\n";
              winner = 0;
              points = msh.cube;
              break;
            }
            std::cout << "Engine takes.\n";
            msh.cube *= 2;
            msh.owner = CubeOwner::Opp;
            show(hb, msh, msh.cube);
            continue;
          }
          if (c.empty() || c == "roll" || c == "r") rolled = true;
        }
        if (winner >= 0) break;
        int a, b;
        roll(a, b);
        std::vector<Move> moves;
        generate_moves(hb, a, b, moves);
        std::cout << "You rolled " << a << b << ".\n";
        Board after;
        if (moves.size() == 1 && moves[0].n == 0) {
          std::cout << "No legal move.\n";
          after = hb;
        } else {
          for (;;) {
            std::string s = ask("Your move: ");
            std::string l = lower(s);
            if (l == "quit") return 0;
            if (l == "moves") {
              for (const Move& m : moves) std::cout << "  " << m.str() << "\n";
              continue;
            }
            if (l == "hint") {
              MoveAnalysis an = search.analyze_moves(hb, a, b, msh, cfg);
              for (size_t i = 0; i < an.moves.size() && i < 3; ++i)
                std::printf("  %-24s %+.3f\n", an.moves[i].move.str().c_str(), an.moves[i].emg);
              continue;
            }
            int k = find_move(moves, hb, s);
            if (k < 0) {
              std::cout << "Not a legal move (type 'moves' to list them).\n";
              continue;
            }
            after = moves[k].after;
            break;
          }
        }
        if (after.side[0] == 0) {
          winner = 0;
          points = int(after.result_for_winner(0)) * msh.cube;
          if (msh.jacoby && msh.owner == CubeOwner::Center) points = msh.cube;
          break;
        }
        hb = after;  // still human's view; engine moves next
        human_turn = false;
      } else {
        Board eb = hb.swapped();
        MatchState mse = msh.swapped();
        if (!opening && mse.owner != CubeOwner::Opp && CubeModel(mse).can_double(0, mse.cube)) {
          CubeResult r = search.analyze_cube(eb, mse, cube_cfg);
          const CubeDecision d = r.analysis.decision;
          if (d == CubeDecision::DoubleTake || d == CubeDecision::DoublePass) {
            show(hb, msh, msh.cube);
            std::string s = lower(ask("Engine doubles to " + std::to_string(2 * msh.cube) + ". Take or pass? "));
            if (s.rfind("p", 0) == 0) {
              winner = 1;
              points = msh.cube;
              break;
            }
            msh.cube *= 2;
            msh.owner = CubeOwner::Me;
            mse = msh.swapped();
          }
        }
        int a, b;
        roll(a, b);
        MoveAnalysis an = search.analyze_moves(eb, a, b, mse, cfg);
        const Move& m = an.moves[0].move;
        if (m.n == 0) std::printf("\nEngine rolls %d%d: no legal move.\n", a, b);
        else
          std::printf("\nEngine rolls %d%d and plays: %s   (in your numbering: %s; engine equity %+.3f)\n", a, b,
                      m.str().c_str(), mirror(m.str()).c_str(), an.moves[0].emg);
        if (m.after.side[0] == 0) {
          winner = 1;
          points = int(m.after.result_for_winner(0)) * msh.cube;
          if (msh.jacoby && msh.owner == CubeOwner::Center) points = msh.cube;
          hb = m.after.swapped();
          break;
        }
        hb = m.after.swapped();
        human_turn = true;
      }
    }
    std::cout << "\n" << (winner == 0 ? "You win " : "Engine wins ") << points << " point" << (points == 1 ? "" : "s")
              << ".\n";
    if (winner == 0) score_h += points;
    else score_e += points;
    if (match > 0) {
      std::cout << "Score: you " << score_h << ", engine " << score_e << " (match to " << match << ")\n";
      if (score_h >= match || score_e >= match) {
        std::cout << (score_h >= match ? "You win the match!\n" : "The engine wins the match.\n");
        return 0;
      }
    } else {
      std::cout << "Total: you " << score_h << ", engine " << score_e << "\n";
    }
    if (lower(ask("Another game? (y/n) ")).rfind("n", 0) == 0) return 0;
  }
}
