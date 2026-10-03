// bgserver: HTTP JSON API for the engine.
//   bgserver [--port 8080] [--host 127.0.0.1] [--net nets/latest.bgnn] [--threads N] [--plies 3] [--cube-extra 2]
#include <cstdio>
#include <cstring>
#include <string>

#include "api/engine.hpp"
#include "httplib.h"

using nlohmann::json;

int main(int argc, char** argv) {
  bg::EngineOptions o;
  int port = 8080;
  std::string host = "127.0.0.1";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--port") port = std::stoi(next());
    else if (a == "--host") host = next();
    else if (a == "--net") o.net_path = next();
    else if (a == "--data") o.data_dir = next();
    else if (a == "--threads") o.threads = std::stoi(next());
    else if (a == "--plies") o.default_plies = std::stoi(next());
    else if (a == "--cube-extra") o.cube_extra_plies = std::stoi(next());
    else if (a == "--hash") o.hash_mb = std::stoul(next());
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 1;
    }
  }
  bg::Engine engine;
  std::string err;
  if (!engine.init(o, &err)) {
    std::fprintf(stderr, "engine init failed: %s\n", err.c_str());
    return 1;
  }

  httplib::Server srv;
  auto reply = [](httplib::Response& res, int status, const json& j) {
    res.status = status;
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_content(j.dump(2), "application/json");
  };
  auto handler = [&](json (bg::Engine::*fn)(const json&)) {
    return [&, fn](const httplib::Request& req, httplib::Response& res) {
      try {
        json body = json::parse(req.body);
        reply(res, 200, (engine.*fn)(body));
      } catch (const json::exception& e) {
        reply(res, 400, {{"error", std::string("invalid JSON: ") + e.what()}});
      } catch (const std::invalid_argument& e) {
        reply(res, 400, {{"error", e.what()}});
      } catch (const std::exception& e) {
        reply(res, 500, {{"error", e.what()}});
      }
    };
  };
  srv.Get("/v1/info", [&](const httplib::Request&, httplib::Response& res) { reply(res, 200, engine.info()); });
  srv.Get("/v1/health", [&](const httplib::Request&, httplib::Response& res) { reply(res, 200, {{"ok", true}}); });
  srv.Post("/v1/evaluate", handler(&bg::Engine::evaluate));
  srv.Post("/v1/move", handler(&bg::Engine::move));
  srv.Post("/v1/cube", handler(&bg::Engine::cube));
  srv.Post("/v1/analyze", handler(&bg::Engine::analyze));
  srv.Post("/v1/rollout", handler(&bg::Engine::rollout));
  srv.Options(R"(/v1/.*)", [](const httplib::Request&, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.status = 204;
  });
  std::printf("%s %s (net %s) listening on http://%s:%d\n", bg::Engine::kName, bg::Engine::kVersion,
              engine.net_id().c_str(), host.c_str(), port);
  std::fflush(stdout);
  if (!srv.listen(host, port)) {
    std::fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port);
    return 1;
  }
  return 0;
}
