// Engine facade shared by the HTTP server and the BGEI text protocol.
// Requests and responses are JSON (see docs/API.md).
#pragma once

#include <memory>
#include <mutex>
#include <string>

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
  int max_plies = 7;
};

class Engine {
 public:
  static constexpr const char* kName = "DarkBGHelmet";
  static constexpr const char* kVersion = "0.1.0";

  bool init(const EngineOptions& o, std::string* err);
  const EngineOptions& options() const { return opts_; }
  std::string net_id() const { return ev_.network().id; }

  nlohmann::json info() const;
  // Each handler validates the request and throws std::invalid_argument on bad input.
  nlohmann::json evaluate(const nlohmann::json& req);
  nlohmann::json move(const nlohmann::json& req);
  nlohmann::json cube(const nlohmann::json& req);
  nlohmann::json analyze(const nlohmann::json& req);

  void set_threads(int t);
  void stop() { if (search_) search_->stop(); }

  // Parses position + match + dice from a request.
  Position parse_request(const nlohmann::json& req) const;
  SearchConfig parse_options(const nlohmann::json& req) const;

 private:
  nlohmann::json position_json(const Position& p) const;
  nlohmann::json probs_json(const Probs& p) const;
  nlohmann::json cube_json(const CubeResult& r, const Position& p) const;

  EngineOptions opts_;
  Evaluator ev_;
  std::unique_ptr<Searcher> search_;
  std::mutex mutex_;  // one analysis at a time (each uses all threads)
};

}  // namespace bg
