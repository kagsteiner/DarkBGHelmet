# Specification of a backgammon engine

Your task is to build a state of the art backgammon engine that relies on a NNUE or similar neuronal network for position analysis and an innovative search algorithm - seems monte carlo tree search is standard, but maybe there are other options like alpha beta?

## Task

The engine shall be written in C, C++ or Java, with Performance in mind. 

First please thoroughly search the web on articles / discussions to find about efficient BG engine programming.

Then make a plan how to build the engine. Features it must support is:

* Decide on doubling, on taking a cube
* Decide on the best move, taking into account that doubles are 4 moves
* Support money games, and matches from 1 to 25 points.
* Find an efficient board+match standing representation similar to Chess bitboards
* Find a way to represent a position analysis that returns the current match equity, game equity and what the correct cube action is in the position, in an NNUE or similar network architecture.
* Suggest a great search algorithm that shall support up to 6 ply search depth (current apps only support up to 5 ply, so let's be daring)
* Find out whether there is a standard format for an engine like UCI in chess. If it isn't suggest an API that we can use for an endpoint of a http server that serves the engine. 

After going through the plan with me, implement the plan.