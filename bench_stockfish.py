"""
bench_stockfish.py -- OPTIONAL external benchmark.

Plays our engine against Stockfish at a capped skill level, to put an
independent number on how strong the thing actually is.

    NOTE FOR THE REPORT: this is a supporting measurement, not the headline
    result.  The headline result is compare.py -- GA-tuned vs untuned baseline
    -- because that is the one that isolates what the genetic algorithm did.

DEPENDENCY NOTE
    This is the ONE script that needs something outside python-chess: the
    Stockfish executable itself.  No new Python library is involved --
    python-chess already speaks the UCI protocol (chess.engine).
    If Stockfish is not installed, this script says so and exits; nothing else
    in the project depends on it, and the lab demo does not need it.

GETTING STOCKFISH
    Windows : winget install Stockfish.Stockfish
              or download from https://stockfishchess.org/download/
              and put stockfish.exe in this folder
    macOS   : brew install stockfish
    Linux   : sudo apt install stockfish

USAGE
    python bench_stockfish.py                     # tuned weights, skill 0
    python bench_stockfish.py --skill 3 --games 10
    python bench_stockfish.py --baseline          # benchmark the control group
    python bench_stockfish.py --engine-path C:/path/to/stockfish.exe

Stockfish skill levels run 0-20.  Level 0 is roughly 1300-1400 Elo, which is
about the right order of magnitude for a depth-3 course-project engine.
"""

import argparse
import glob
import os
import shutil
import sys
import time

import chess

try:
    import chess.engine
except ImportError:                                   # pragma: no cover
    print("python-chess is missing.  Run:  pip install chess")
    sys.exit(1)

import engine as our_engine

DEFAULT_SKILL = 0
DEFAULT_GAMES = 10
DEFAULT_DEPTH = 3
DEFAULT_MOVETIME = 0.05        # seconds Stockfish is allowed per move
MAX_PLIES = 160

CANDIDATE_NAMES = ["stockfish", "stockfish.exe"]
CANDIDATE_DIRS = [
    ".",
    os.path.expanduser("~"),
    r"C:\Program Files\Stockfish",
    "/usr/games", "/usr/local/bin", "/opt/homebrew/bin",
]

# Where `winget install Stockfish.Stockfish` puts the executable.  It is worth
# checking explicitly, because winget only adds it to PATH for NEW shells --
# so right after installing, shutil.which() still cannot see it.
WINGET_PATTERNS = [
    os.path.join(os.environ.get("LOCALAPPDATA", ""),
                 "Microsoft", "WinGet", "Links", "stockfish.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""),
                 "Microsoft", "WinGet", "Packages", "Stockfish.Stockfish*",
                 "**", "stockfish*.exe"),
]


def find_stockfish(explicit_path=None):
    """Locate a Stockfish executable, or return None."""
    if explicit_path:
        return explicit_path if os.path.exists(explicit_path) else None

    for name in CANDIDATE_NAMES:
        found = shutil.which(name)
        if found:
            return found

    for directory in CANDIDATE_DIRS:
        for name in CANDIDATE_NAMES:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate

    for pattern in WINGET_PATTERNS:
        if not pattern:
            continue
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def play_against_stockfish(stockfish, weights, our_colour, depth, movetime):
    """One game.  Returns (points_for_us, plies, termination)."""
    board = chess.Board()
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        if board.turn == our_colour:
            move = our_engine.choose_move(board, weights, depth=depth)
        else:
            result = stockfish.play(board, chess.engine.Limit(time=movetime))
            move = result.move
        if move is None:
            break
        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Move cap: adjudicate on material, like compare.py does.
        material = our_engine._material_balance(board)
        ours = material if our_colour == chess.WHITE else -material
        if ours >= 1.5:
            return 1.0, plies, "adjudicated"
        if ours <= -1.5:
            return 0.0, plies, "adjudicated"
        return 0.5, plies, "adjudicated"

    reason = outcome.termination.name.lower()
    if outcome.winner is None:
        return 0.5, plies, reason
    return (1.0 if outcome.winner == our_colour else 0.0), plies, reason


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the engine against Stockfish at capped skill.")
    parser.add_argument("--skill", type=int, default=DEFAULT_SKILL,
                        help="Stockfish skill level 0-20 (default {})"
                             .format(DEFAULT_SKILL))
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                        help="number of games, colours alternate (default {})"
                             .format(DEFAULT_GAMES))
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help="our search depth (default {})".format(DEFAULT_DEPTH))
    parser.add_argument("--movetime", type=float, default=DEFAULT_MOVETIME,
                        help="seconds per move for Stockfish (default {})"
                             .format(DEFAULT_MOVETIME))
    parser.add_argument("--baseline", action="store_true",
                        help="benchmark the untuned baseline instead of the "
                             "GA-tuned weights")
    parser.add_argument("--engine-path", default=None,
                        help="explicit path to the Stockfish executable")
    args = parser.parse_args()

    path = find_stockfish(args.engine_path)
    if path is None:
        print("Stockfish was not found on this machine.")
        print()
        print("This benchmark is OPTIONAL -- nothing else in the project needs it.")
        print("To install it:")
        print("    Windows : winget install Stockfish.Stockfish")
        print("    macOS   : brew install stockfish")
        print("    Linux   : sudo apt install stockfish")
        print("Or download from https://stockfishchess.org/download/ and pass")
        print("    python bench_stockfish.py --engine-path <path to the exe>")
        sys.exit(1)

    if args.baseline:
        weights = list(our_engine.BASELINE_WEIGHTS)
        label = "untuned baseline"
    else:
        weights = our_engine.load_weights()
        if weights is None:
            print("{} not found -- run  python train_ga.py  first, or pass "
                  "--baseline.".format(our_engine.WEIGHTS_FILE))
            sys.exit(1)
        label = "GA-tuned"

    print("=" * 70)
    print("BENCHMARK: our {} engine  vs  Stockfish (skill {})"
          .format(label, args.skill))
    print("=" * 70)
    print("stockfish : {}".format(path))
    print("games     : {} (colours alternate)".format(args.games))
    print("our depth : {}".format(args.depth))
    print("sf per-move time: {}s".format(args.movetime))
    print("-" * 70, flush=True)

    wins = draws = losses = 0
    started = time.time()

    with chess.engine.SimpleEngine.popen_uci(path) as stockfish:
        stockfish.configure({"Skill Level": max(0, min(20, args.skill))})

        for index in range(args.games):
            our_colour = chess.WHITE if index % 2 == 0 else chess.BLACK
            points, plies, reason = play_against_stockfish(
                stockfish, weights, our_colour, args.depth, args.movetime)

            if points == 1.0:
                wins += 1
                result = "WIN "
            elif points == 0.5:
                draws += 1
                result = "DRAW"
            else:
                losses += 1
                result = "LOSS"

            print("game {:>3}/{:<3}  we play {:<5}  {}   "
                  "running: {}W {}D {}L   ({} plies, {})".format(
                      index + 1, args.games,
                      "white" if our_colour == chess.WHITE else "black",
                      result, wins, draws, losses, plies, reason), flush=True)

    score = wins + 0.5 * draws
    elapsed = time.time() - started

    print("-" * 70)
    print("RESULT vs Stockfish skill {}:  {}W {}D {}L   score {:.1f}/{} "
          "({:.1f}%)".format(args.skill, wins, draws, losses, score,
                             args.games, 100 * score / args.games))
    print("total time: {:.1f} min".format(elapsed / 60))


if __name__ == "__main__":
    main()
