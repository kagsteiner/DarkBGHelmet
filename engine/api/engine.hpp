// Engine facade shared by the HTTP server and the BGEI text protocol.
// Requests and responses are JSON (see docs/API.md).
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/ids.hpp"
#include "eval/evaluator.hpp"
#include "search/search.hpp"

#include "json.hpp"

namespace bg {

struct EngineOptions {
  std::string net_path = "nets/latest.bgnn";
  std::string data_dir = "data";
  int threads = 0;        // 0 = all cores
  size_t hash_mb = 256;
  int default_plies = 3;
  int cube_extra_plies = 2;  // cube decisions search this many plies deeper (0 = same depth)
  int max_plies = 7;
  // Additional networks, selectable per request by name ("network" option).
  // The network from net_path is always available as "default".
  std::vector<std::pair<std::string, std::string>> extra_nets;  // name, file
};

class Engine {
 public:
  static constexpr const char* kName = "DarkBGHelmet";
  static constexpr const char* kVersion = "0.1.0";

  bool init(const EngineOptions& o, std::string* err);
  const EngineOptions& options() const { return opts_; }
  std::string net_id() const { return nets_.empty() ? "" : nets_[0]->ev->network().id; }

  // Loads another network under `name` (replacing one with the same name).
  bool add_network(const std::string& name, const std::string& path, std::string* err);
  std::vector<std::string> network_names() const;

  nlohmann::json info() const;
  // Each handler validates the request and throws std::invalid_argument on bad input.
  nlohmann::json evaluate(const nlohmann::json& req);
  nlohmann::json move(const nlohmann::json& req);
  nlohmann::json cube(const nlohmann::json& req);
  nlohmann::json analyze(const nlohmann::json& req);
  nlohmann::json rollout(const nlohmann::json& req);

  void set_threads(int t);
  void stop() { for (auto& n : nets_) n->search->stop(); }

  // Parses position + match + dice from a request.
  Position parse_request(const nlohmann::json& req) const;
  SearchConfig parse_options(const nlohmann::json& req) const;
  // Depth for cube decisions: options.cube_plies, else plies + cube_extra_plies.
  SearchConfig parse_cube_options(const nlohmann::json& req) const;
  void set_cube_extra_plies(int n) { opts_.cube_extra_plies = std::max(0, n); }

 private:
  nlohmann::json position_json(const Position& p) const;
  nlohmann::json probs_json(const Probs& p) const;
  nlohmann::json cube_json(const CubeResult& r, const Position& p) const;

  struct Slot {
    std::string name, path;
    std::unique_ptr<Evaluator> ev;
    std::unique_ptr<Searcher> search;
  };
  // Network chosen by the request's "network" option (default: the first);
  // applies the "cube_head" option for the duration of the request.
  Slot& select(const nlohmann::json& req);
  struct HeadGuard {
    Evaluator* ev = nullptr;
    bool prev = true;
    ~HeadGuard() { if (ev) ev->set_use_cube_head(prev); }
  };
  void apply_cube_head(Slot& s, const nlohmann::json& req, HeadGuard& g);

  EngineOptions opts_;
  std::vector<std::unique_ptr<Slot>> nets_;
  std::mutex mutex_;  // one analysis at a time (each uses all threads)
};

}  // namespace bg
