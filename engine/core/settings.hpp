// Local settings from the environment or a .env file in the working directory.
//
//   THREADS=12        default number of search threads for bgplay, bgselfplay,
//                     bgei and bgserver (command-line --threads still wins)
//
// The process environment takes precedence over the .env file.
#pragma once

#include <cstdlib>
#include <fstream>
#include <string>

namespace bg {

// Value of `key` from the environment or ./.env, or "" if not set.
inline std::string local_setting(const std::string& key) {
  if (const char* v = std::getenv(key.c_str())) return v;
  std::ifstream f(".env");
  std::string line;
  while (std::getline(f, line)) {
    const size_t start = line.find_first_not_of(" \t");
    if (start == std::string::npos || line[start] == '#') continue;
    const size_t eq = line.find('=', start);
    if (eq == std::string::npos) continue;
    std::string k = line.substr(start, eq - start);
    while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
    if (k != key) continue;
    std::string v = line.substr(eq + 1);
    const size_t vs = v.find_first_not_of(" \t\"'");
    if (vs == std::string::npos) return "";
    v = v.substr(vs);
    while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '\r' || v.back() == '"' || v.back() == '\''))
      v.pop_back();
    return v;
  }
  return "";
}

// THREADS from the environment or .env, else `fallback` (0 = all cores).
inline int default_threads(int fallback) {
  const std::string v = local_setting("THREADS");
  if (v.empty()) return fallback;
  const int n = std::atoi(v.c_str());
  return n >= 0 ? n : fallback;
}

}  // namespace bg
