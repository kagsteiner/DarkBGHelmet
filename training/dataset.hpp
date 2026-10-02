// Training records: a position (side 0 on roll) and its target probabilities.
#pragma once

#include <string>
#include <vector>

#include "core/board.hpp"
#include "eval/features.hpp"

namespace bg {

struct Record {
  Board board;
  float target[5];
  int8_t cls;    // PosClass
  int8_t plies;  // label depth (gnubg plies, -1 unknown)
  int16_t weight = 1;
};

// Parses label text files ("<posid> <plies> p0..p4"), drops invalid, finished
// and bearoff-database positions, and de-duplicates (deeper label wins).
std::vector<Record> load_labels(const std::vector<std::string>& files, int min_plies, size_t* parsed = nullptr);

bool save_records(const std::string& path, const std::vector<Record>& recs);
bool load_records(const std::string& path, std::vector<Record>& recs);

}  // namespace bg
