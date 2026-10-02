// bgei: line-based, UCI-like text protocol on stdin/stdout ("BackGammon Engine
// Interface"). See docs/API.md for the full description.
//
//   bgei                      -> id name ..., option lines, bgeiok
//   isready                   -> readyok
//   setoption name <Threads|Plies|Net> value <v>
//   position start | position xgid <XGID> | position gnubg <posid>[:<matchid>]
//   match length <n> score <a> <b> cube <c> owner <center|onroll|opponent> crawford <0|1>
//   go move dice <d1d2> [plies n]   -> info ... lines, then "bestmove <move>"
//   go cube [plies n]               -> "bestcube <decision>; nd .. dt .. dp .."
//   go eval [plies n]               -> "eval win .. wg .. wbg .. lg .. lbg .. equity .."
//   go rollout [dice d1d2] [trials n] [truncate t] -> "rollout <move|position> ..." lines
//   json <request>                  -> one line of JSON (the HTTP /v1/analyze answer)
//   quit
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "api/engine.hpp"

using nlohmann::json;

int main(int argc, char** argv) {
  bg::EngineOptions o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--net" && i + 1 < argc) o.net_path = argv[++i];
    else if (a == "--threads" && i + 1 < argc) o.threads = std::stoi(argv[++i]);
    else if (a == "--data" && i + 1 < argc) o.data_dir = argv[++i];
  }
  bg::Engine engine;
  std::string err;
  if (!engine.init(o, &err)) {
    std::cout << "error " << err << std::endl;
    return 1;
  }
  json pos = {{"xgid", bg::to_xgid(bg::Position{bg::Board::start(), {}, {0, 0}})}};
  json match = json::object();
  int plies = o.default_plies;

  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream is(line);
    std::string cmd;
    is >> cmd;
    try {
      if (cmd == "bgei") {
        std::cout << "id name " << bg::Engine::kName << " " << bg::Engine::kVersion << "\n"
                  << "id network " << engine.net_id() << "\n"
                  << "option name Threads type spin default 0 min 0 max 256\n"
                  << "option name Plies type spin default " << o.default_plies << " min 1 max 7\n"
                  << "option name Net type string default " << o.net_path << "\n"
                  << "bgeiok" << std::endl;
      } else if (cmd == "isready") {
        std::cout << "readyok" << std::endl;
      } else if (cmd == "setoption") {
        std::string tok, name, value;
        while (is >> tok) {
          if (tok == "name") is >> name;
          else if (tok == "value") std::getline(is >> std::ws, value);
        }
        if (name == "Threads") engine.set_threads(std::stoi(value));
        else if (name == "Plies") plies = std::stoi(value);
        else if (name == "Net") {
          bg::EngineOptions no = engine.options();
          no.net_path = value;
          if (!engine.init(no, &err)) std::cout << "error " << err << std::endl;
        } else std::cout << "error unknown option " << name << std::endl;
      } else if (cmd == "position") {
        std::string kind, arg;
        is >> kind;
        std::getline(is >> std::ws, arg);
        if (kind == "start") pos = {{"xgid", bg::to_xgid(bg::Position{bg::Board::start(), {}, {0, 0}})}};
        else if (kind == "xgid") pos = {{"xgid", arg}};
        else if (kind == "gnubg") pos = {{"gnubg", arg}};
        else throw std::invalid_argument("position start|xgid|gnubg");
        match = json::object();
      } else if (cmd == "match") {
        std::string k;
        while (is >> k) {
          if (k == "length") { int v; is >> v; match["length"] = v; }
          else if (k == "score") { int a, b; is >> a >> b; match["score"] = {a, b}; }
          else if (k == "cube") { int v; is >> v; match["cube"] = v; }
          else if (k == "owner") { std::string v; is >> v; match["owner"] = v; }
          else if (k == "crawford") { int v; is >> v; match["crawford"] = v != 0; }
          else if (k == "jacoby") { int v; is >> v; match["jacoby"] = v != 0; }
        }
      } else if (cmd == "go") {
        std::string what, k;
        is >> what;
        json req = {{"position", pos}, {"options", {{"plies", plies}}}};
        if (!match.empty()) req["match"] = match;
        while (is >> k) {
          if (k == "dice") { std::string d; is >> d; req["dice"] = {d[0] - '0', d[1] - '0'}; }
          else if (k == "plies") { int v; is >> v; req["options"]["plies"] = v; }
          else if (k == "trials") { int v; is >> v; req["options"]["trials"] = v; }
          else if (k == "truncate") { int v; is >> v; req["options"]["truncate"] = v; }
        }
        if (what == "move") {
          json r = engine.move(req);
          int i = 0;
          for (const auto& m : r["moves"]) {
            std::printf("info rank %d move %s equity %.4f diff %.4f plies %d\n", ++i,
                        m["move"].get<std::string>().c_str(), m["equity"].get<double>(), m["diff"].get<double>(),
                        m["plies"].get<int>());
          }
          std::printf("bestmove %s\n", r["best"].get<std::string>().c_str());
        } else if (what == "cube") {
          json r = engine.cube(req);
          std::printf("bestcube %s; take %s; nd %.4f dt %.4f dp %.4f\n", r["decision"].get<std::string>().c_str(),
                      r["take"].get<std::string>().c_str(), r["equity"]["no_double"].get<double>(),
                      r["equity"]["double_take"].get<double>(), r["equity"]["double_pass"].get<double>());
        } else if (what == "rollout") {
          json r = engine.rollout(req);
          auto line = [](const json& x, const std::string& tag) {
            const json& p = x["probabilities"];
            std::printf("rollout %s win %.4f wg %.4f wbg %.4f lg %.4f lbg %.4f cubeless %.4f cubeful %.4f se %.4f trials %d\n",
                        tag.c_str(), p["win"].get<double>(), p["win_gammon"].get<double>(),
                        p["win_backgammon"].get<double>(), p["lose_gammon"].get<double>(),
                        p["lose_backgammon"].get<double>(), x.value("cubeless_equity", 0.0),
                        x["cubeful_equity"].get<double>(), x.contains("std_error") ? x["std_error"]["cubeful"].get<double>() : 0.0,
                        x["trials"].get<int>());
          };
          if (r.contains("moves")) for (const auto& m : r["moves"]) line(m, m["move"].get<std::string>());
          else line(r, "position");
        } else if (what == "eval") {
          json r = engine.evaluate(req);
          const json& p = r["probabilities"];
          std::printf("eval win %.4f wg %.4f wbg %.4f lg %.4f lbg %.4f cubeless %.4f cubeful %.4f\n",
                      p["win"].get<double>(), p["win_gammon"].get<double>(), p["win_backgammon"].get<double>(),
                      p["lose_gammon"].get<double>(), p["lose_backgammon"].get<double>(),
                      r["cubeless_equity"].get<double>(), r["cubeful_equity"].get<double>());
        } else {
          throw std::invalid_argument("go move|cube|eval");
        }
        std::fflush(stdout);
      } else if (cmd == "json") {
        std::string rest;
        std::getline(is >> std::ws, rest);
        std::cout << engine.analyze(json::parse(rest)).dump() << std::endl;
      } else if (cmd == "quit") {
        break;
      } else if (!cmd.empty()) {
        std::cout << "error unknown command " << cmd << std::endl;
      }
    } catch (const std::exception& e) {
      std::cout << "error " << e.what() << std::endl;
    }
  }
  return 0;
}
