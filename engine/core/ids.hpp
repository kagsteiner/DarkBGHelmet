// Position formats: gnubg Position ID / Match ID and eXtreme Gammon XGID.
#pragma once

#include <string>

#include "match.hpp"

namespace bg {

std::string position_id(const Board& b);
bool parse_position_id(const std::string& id, Board& out);

std::string match_id(const MatchState& m, int d1 = 0, int d2 = 0);
bool parse_match_id(const std::string& id, MatchState& m, int& d1, int& d2);

// XGID, e.g. "XGID=-b----E-C---eE---c-e----B-:0:0:1:52:0:0:0:0:10".
// The player on roll becomes board side 0.
bool parse_xgid(const std::string& xgid, Position& out, std::string* err = nullptr);
std::string to_xgid(const Position& p);

// Accepts "XGID=...", "posid:matchid", or a bare 14-char position ID.
bool parse_any(const std::string& text, Position& out, std::string* err = nullptr);

}  // namespace bg
