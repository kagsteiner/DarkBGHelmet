#include "dataset.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include "core/ids.hpp"

namespace bg {

std::vector<Record> load_labels(const std::vector<std::string>& files, int min_plies, size_t* parsed) {
  std::vector<Record> out;
  std::unordered_map<u64, size_t> seen;
  seen.reserve(1 << 24);
  size_t n_parsed = 0;
  char line[256];
  for (const auto& path : files) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); continue; }
    while (std::fgets(line, sizeof line, f)) {
      char pid[32];
      int plies;
      float q[5];
      if (std::sscanf(line, "%31s %d %f %f %f %f %f", pid, &plies, &q[0], &q[1], &q[2], &q[3], &q[4]) != 7) continue;
      ++n_parsed;
      if (plies < min_plies) continue;
      Record r;
      if (!parse_position_id(pid, r.board)) continue;
      PosClass c = classify(r.board);
      if (c == PosClass::Over || c == PosClass::Bearoff) continue;
      r.cls = int8_t(c);
      r.plies = int8_t(plies);
      std::memcpy(r.target, q, sizeof q);
      u64 h = r.board.hash();
      auto it = seen.find(h);
      if (it != seen.end()) {
        Record& o = out[it->second];
        if (o.board == r.board) {
          if (r.plies > o.plies) o = r;
          continue;
        }
      }
      seen[h] = out.size();
      out.push_back(r);
    }
    std::fclose(f);
  }
  if (parsed) *parsed = n_parsed;
  return out;
}

bool save_records(const std::string& path, const std::vector<Record>& recs) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  u64 n = recs.size();
  std::fwrite(&n, sizeof n, 1, f);
  std::fwrite(recs.data(), sizeof(Record), recs.size(), f);
  std::fclose(f);
  return true;
}

bool load_records(const std::string& path, std::vector<Record>& recs) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  u64 n = 0;
  if (std::fread(&n, sizeof n, 1, f) != 1) { std::fclose(f); return false; }
  recs.resize(n);
  bool ok = std::fread(recs.data(), sizeof(Record), n, f) == n;
  std::fclose(f);
  return ok;
}

}  // namespace bg
