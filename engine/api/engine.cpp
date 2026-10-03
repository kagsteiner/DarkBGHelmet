#include "engine.hpp"

#include <stdexcept>
#include <thread>

#include "search/rollout.hpp"

using nlohmann::json;

namespace bg {

namespace {

[[noreturn]] void bad(const std::string& m) { throw std::invalid_argument(m); }

CubeOwner parse_owner(const std::string& s) {
  if (s == "center" || s == "centered" || s == "none") return CubeOwner::Center;
  if (s == "onroll" || s == "on_roll" || s == "me" || s == "player") return CubeOwner::Me;
  if (s == "opponent" || s == "opp") return CubeOwner::Opp;
  bad("cube owner must be center, onroll or opponent");
}

const char* owner_name(CubeOwner o) {
  return o == CubeOwner::Center ? "center" : (o == CubeOwner::Me ? "onroll" : "opponent");
}

}  // namespace

bool Engine::init(const EngineOptions& o, std::string* err) {
  opts_ = o;
  if (!ev_.init(o.net_path, o.data_dir, err)) return false;
  search_ = std::make_unique<Searcher>(ev_, o.threads, o.hash_mb);
  return true;
}

void Engine::set_threads(int t) {
  std::lock_guard<std::mutex> g(mutex_);
  opts_.threads = t;
  search_->set_threads(t);
}

json Engine::info() const {
  return {{"name", kName},
          {"version", kVersion},
          {"network", ev_.network().id},
          {"threads", search_ ? search_->threads() : 0},
          {"default_plies", opts_.default_plies},
          {"cube_extra_plies", opts_.cube_extra_plies},
          {"max_plies", opts_.max_plies},
          {"ply_convention", "XG (1-ply = static evaluation)"},
          {"position_formats", {"xgid", "gnubg", "points"}},
          {"match_lengths", "0 (money) or 1..25"}};
}

// Position input:
//   {"xgid": "XGID=..."}                         -- includes match/cube/dice
//   {"gnubg": "<posid>:<matchid>"} or {"gnubg": "<posid>"}
//   {"points": [26 ints]}  index 1..24 = points of the player on roll (positive) /
//                          opponent (negative); 25 = on-roll player's bar,
//                          0 = opponent's bar (count, sign ignored)
Position Engine::parse_request(const json& req) const {
  if (!req.is_object()) bad("request must be a JSON object");
  Position p;
  std::string err;
  if (!req.contains("position")) bad("missing 'position'");
  const json& pos = req["position"];
  if (pos.is_string()) {
    if (!parse_any(pos.get<std::string>(), p, &err)) bad("position: " + err);
  } else if (pos.contains("xgid")) {
    if (!parse_xgid(pos["xgid"].get<std::string>(), p, &err)) bad("xgid: " + err);
  } else if (pos.contains("gnubg")) {
    if (!parse_any(pos["gnubg"].get<std::string>(), p, &err)) bad("gnubg: " + err);
  } else if (pos.contains("points")) {
    const json& a = pos["points"];
    if (!a.is_array() || a.size() != 26) bad("points must be an array of 26 integers");
    Board b;
    for (int i = 1; i <= 24; ++i) {
      int v = a[i].get<int>();
      if (v > 0) b.set(0, i - 1, v);
      if (v < 0) b.set(1, 24 - i, -v);
    }
    b.set(0, kBar, std::abs(a[25].get<int>()));
    b.set(1, kBar, std::abs(a[0].get<int>()));
    if (!b.valid(&err)) bad("points: " + err);
    p.board = b;
  } else {
    bad("position needs one of xgid, gnubg, points");
  }
  if (req.contains("match")) {
    const json& m = req["match"];
    MatchState& ms = p.match;
    if (m.contains("length")) ms.length = m["length"].get<int>();
    if (m.contains("score")) {
      if (!m["score"].is_array() || m["score"].size() != 2) bad("score must be [onroll, opponent]");
      ms.score[0] = m["score"][0].get<int>();
      ms.score[1] = m["score"][1].get<int>();
    }
    if (m.contains("cube")) ms.cube = m["cube"].get<int>();
    if (m.contains("owner")) ms.owner = parse_owner(m["owner"].get<std::string>());
    if (m.contains("crawford")) ms.crawford = m["crawford"].get<bool>();
    if (m.contains("jacoby")) ms.jacoby = m["jacoby"].get<bool>();
    if (m.contains("beavers")) ms.beavers = m["beavers"].get<bool>();
    if (ms.cube == 1 && ms.owner != CubeOwner::Center) bad("a 1-cube must be centered");
  }
  if (!p.match.valid(&err)) bad("match: " + err);
  if (req.contains("dice")) {
    const json& d = req["dice"];
    if (!d.is_array() || d.size() != 2) bad("dice must be [d1, d2]");
    p.dice[0] = d[0].get<int>();
    p.dice[1] = d[1].get<int>();
    if (p.dice[0] < 1 || p.dice[0] > 6 || p.dice[1] < 1 || p.dice[1] > 6) bad("dice must be 1..6");
  }
  if (p.board.winner() >= 0) bad("the game is already over in this position");
  return p;
}

SearchConfig Engine::parse_options(const json& req) const {
  int plies = opts_.default_plies;
  json o = req.contains("options") ? req["options"] : json::object();
  if (o.contains("plies")) plies = o["plies"].get<int>();
  if (plies < 1 || plies > opts_.max_plies) bad("plies must be 1.." + std::to_string(opts_.max_plies));
  SearchConfig cfg = SearchConfig::for_plies(plies);
  if (o.contains("cubeful")) cfg.cubeful = o["cubeful"].get<bool>();
  if (o.contains("candidates")) cfg.max_candidates = o["candidates"].get<int>();
  if (o.contains("sampled_rolls")) cfg.sampled_rolls = std::clamp(o["sampled_rolls"].get<int>(), 1, 21);
  return cfg;
}

SearchConfig Engine::parse_cube_options(const json& req) const {
  SearchConfig base = parse_options(req);
  json o = req.contains("options") ? req["options"] : json::object();
  int extra = o.contains("cube_extra_plies") ? o["cube_extra_plies"].get<int>() : opts_.cube_extra_plies;
  int plies = o.contains("cube_plies") ? o["cube_plies"].get<int>() : base.plies + std::max(0, extra);
  plies = std::clamp(plies, 1, opts_.max_plies);
  SearchConfig cfg = SearchConfig::for_plies(plies);
  cfg.cubeful = base.cubeful;
  if (o.contains("sampled_rolls")) cfg.sampled_rolls = base.sampled_rolls;
  return cfg;
}

json Engine::position_json(const Position& p) const {
  return {{"gnubg", position_id(p.board) + ":" + match_id(p.match, p.dice[0], p.dice[1])}, {"xgid", to_xgid(p)}};
}

json Engine::probs_json(const Probs& p) const {
  return {{"win", p[kWin]},   {"win_gammon", p[kWinG]},   {"win_backgammon", p[kWinBG]},
          {"lose", 1 - p[kWin]}, {"lose_gammon", p[kLoseG]}, {"lose_backgammon", p[kLoseBG]}};
}

json Engine::cube_json(const CubeResult& r, const Position& p) const {
  const CubeAnalysis& a = r.analysis;
  const bool match = !p.match.money();
  json j = {{"decision", to_string(a.decision, p.match.owner == CubeOwner::Me)},
            {"take", a.take ? (a.beaver ? "beaver" : "take") : "pass"},
            {"cube_available", a.can_double},
            {"equity", {{"no_double", r.nd_emg}, {"double_take", r.dt_emg}, {"double_pass", r.dp_emg}}},
            {"cube_efficiency", a.x}};
  if (match) j["mwc"] = {{"no_double", a.nd}, {"double_take", a.dt}, {"double_pass", a.dp}};
  return j;
}

json Engine::evaluate(const json& req) {
  Position p = parse_request(req);
  SearchConfig cfg = parse_cube_options(req);
  std::lock_guard<std::mutex> g(mutex_);
  CubeResult r = search_->analyze_cube(p.board, p.match, cfg);
  CubeModel cm(p.match);
  const bool jac = p.match.money() && p.match.jacoby && p.match.owner == CubeOwner::Center && p.match.cube == 1;
  const float cubeless = cm.dead(r.analysis.probs, p.match.cube, jac);
  json out = {{"plies", r.plies},
              {"probabilities", probs_json(r.analysis.probs)},
              {"cubeless_equity", r.analysis.probs.equity()},
              {"cubeful_equity", cm.to_emg(r.analysis.best())},
              {"cube", cube_json(r, p)},
              {"position", position_json(p)},
              {"nodes", r.nodes},
              {"time_ms", r.seconds * 1000}};
  if (!p.match.money()) {
    out["match_equity"] = {{"cubeless", cubeless}, {"cubeful", r.analysis.best()}};
    out["match"] = {{"length", p.match.length},
                    {"away", {p.match.away(0), p.match.away(1)}},
                    {"crawford", p.match.crawford},
                    {"cube", p.match.cube},
                    {"owner", owner_name(p.match.owner)}};
  }
  return out;
}

json Engine::cube(const json& req) {
  Position p = parse_request(req);
  SearchConfig cfg = parse_cube_options(req);
  std::lock_guard<std::mutex> g(mutex_);
  CubeResult r = search_->analyze_cube(p.board, p.match, cfg);
  json out = cube_json(r, p);
  out["plies"] = r.plies;
  out["probabilities"] = probs_json(r.analysis.probs);
  out["nodes"] = r.nodes;
  out["time_ms"] = r.seconds * 1000;
  return out;
}

json Engine::move(const json& req) {
  Position p = parse_request(req);
  if (!p.dice[0]) bad("missing 'dice' (or dice in the position ID)");
  SearchConfig cfg = parse_options(req);
  int limit = 20;
  if (req.contains("options") && req["options"].contains("max_moves")) limit = req["options"]["max_moves"].get<int>();
  std::lock_guard<std::mutex> g(mutex_);
  MoveAnalysis a = search_->analyze_moves(p.board, p.dice[0], p.dice[1], p.match, cfg);
  json moves = json::array();
  const bool match = !p.match.money();
  for (size_t i = 0; i < a.moves.size() && int(i) < limit; ++i) {
    const MoveEval& m = a.moves[i];
    Position after;
    after.board = m.move.after.swapped();
    after.match = p.match.swapped();
    json j = {{"move", m.move.str()},
              {"equity", m.emg},
              {"diff", m.emg - a.moves[0].emg},
              {"plies", m.plies},
              {"probabilities", probs_json(m.probs)},
              {"position_after", position_json(after)}};
    if (match) j["mwc"] = m.value;
    moves.push_back(j);
  }
  return {{"dice", {p.dice[0], p.dice[1]}},
          {"plies", cfg.plies},
          {"legal_moves", a.moves.size()},
          {"moves", moves},
          {"best", a.moves.empty() ? "" : a.moves[0].move.str()},
          {"nodes", a.nodes},
          {"time_ms", a.seconds * 1000}};
}

json Engine::analyze(const json& req) {
  json out;
  json creq = req;
  creq.erase("dice");
  out["cube"] = cube(creq);
  Position p = parse_request(req);
  if (p.dice[0]) out["checker"] = move(req);
  out["position"] = position_json(p);
  return out;
}

}  // namespace bg

namespace bg {

// Rolls out the position (before the roll) or, with dice, every candidate play
// among the best `candidates` at 1-ply.
json Engine::rollout(const json& req) {
  Position p = parse_request(req);
  json o = req.contains("options") ? req["options"] : json::object();
  RolloutConfig rc;
  if (o.contains("trials")) rc.trials = o["trials"].get<int>();
  if (o.contains("truncate")) rc.truncate = o["truncate"].get<int>();
  if (o.contains("cubeful")) rc.cubeful = o["cubeful"].get<bool>();
  if (o.contains("variance_reduction")) rc.variance_reduction = o["variance_reduction"].get<bool>();
  if (o.contains("seed")) rc.seed = o["seed"].get<uint64_t>();
  if (rc.trials < 1 || rc.trials > 1000000) bad("trials must be 1..1000000");
  std::lock_guard<std::mutex> g(mutex_);
  CubeModel cm(p.match);
  auto result_json = [&](const RolloutResult& r, const CubeModel& m) {
    json j = {{"probabilities", probs_json(r.probs)},
              {"cubeless_equity", m.to_emg(r.cubeless)},
              {"cubeful_equity", m.to_emg(r.cubeful)},
              {"std_error", {{"cubeless", r.se_cubeless}, {"cubeful", r.se_cubeful}}},
              {"trials", r.trials},
              {"time_ms", r.seconds * 1000}};
    if (!p.match.money()) j["mwc"] = {{"cubeless", r.cubeless}, {"cubeful", r.cubeful}};
    return j;
  };
  if (!p.dice[0]) {
    RolloutResult r = bg::rollout(ev_, p.board, p.match, rc, search_->pool());
    json j = result_json(r, cm);
    j["note"] = "cubeful value assumes the player on roll does not double now";
    return j;
  }
  int cands = o.contains("candidates") ? o["candidates"].get<int>() : 4;
  SearchConfig sc = SearchConfig::for_plies(1);
  MoveAnalysis a = search_->analyze_moves(p.board, p.dice[0], p.dice[1], p.match, sc);
  json moves = json::array();
  const MatchState opp = p.match.swapped();
  CubeModel ocm(opp);
  for (int i = 0; i < int(a.moves.size()) && i < cands; ++i) {
    const MoveEval& m = a.moves[i];
    json j;
    if (m.move.after.side[0] == 0) {
      j = {{"probabilities", probs_json(m.probs)}, {"cubeful_equity", m.emg}, {"trials", 0}};
    } else {
      RolloutResult r = bg::rollout(ev_, m.move.after.swapped(), opp, rc, search_->pool());
      // Seen from the mover: invert the opponent's result.
      RolloutResult mine = r;
      mine.probs = r.probs.inverted();
      mine.cubeless = p.match.money() ? -r.cubeless : 1 - r.cubeless;
      mine.cubeful = p.match.money() ? -r.cubeful : 1 - r.cubeful;
      j = result_json(mine, cm);
      j["note"] = "the opponent may double before its first roll in the cubeful value";
    }
    j["move"] = m.move.str();
    moves.push_back(j);
  }
  std::stable_sort(moves.begin(), moves.end(), [](const json& x, const json& y) {
    return x["cubeful_equity"].get<double>() > y["cubeful_equity"].get<double>();
  });
  return {{"dice", {p.dice[0], p.dice[1]}}, {"moves", moves}};
}

}  // namespace bg
