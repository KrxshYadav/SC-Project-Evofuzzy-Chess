"""
make_positions.py -- build positions.csv for the GA's "positions" fitness mode.

Only needed if you set FITNESS_MODE = "positions" in train_ga.py.
Fitness mode "games" needs no data at all.

WHAT THIS PRODUCES
    positions.csv with two columns:
        fen        the position, as text
        best_move  the move a strong player actually chose, in UCI

    The GA then measures how many of those moves its evaluation function
    reproduces at 1 ply.  This is the fitness function of Paper 2
    (Vazquez-Fernandez, Coello Coello & Sagols Troncoso, 2013).

WHERE THE DATA COMES FROM
    The Lichess Open Database puzzle file:
        https://database.lichess.org/#puzzles
        https://database.lichess.org/lichess_db_puzzle.csv.zst
    Each puzzle row already contains a FEN and the solution moves, so there is
    no PGN parsing to do.

    DO NOT download a full monthly game archive -- those are ~20 GB compressed.
    The puzzle file is a few hundred MB and 10,000-50,000 positions is plenty
    for a course project.

IMPORTANT DETAIL ABOUT THE PUZZLE FORMAT
    In a Lichess puzzle the FEN is the position BEFORE the opponent's blunder,
    and the first move in the "Moves" column is that opponent move.  The move
    the SOLVER is meant to find is the SECOND one.  So this script plays the
    first move onto the board and records (resulting FEN, second move).

USAGE
    1. Download the file (browser, or curl):
           curl -O https://database.lichess.org/lichess_db_puzzle.csv.zst
    2. Decompress it.  .zst needs the zstd tool (https://github.com/facebook/zstd
       or `winget install Facebook.Zstandard` on Windows):
           zstd -d lichess_db_puzzle.csv.zst
    3. Build the training file:
           python make_positions.py lichess_db_puzzle.csv

    Options:
           python make_positions.py <input.csv> --limit 20000 --min-rating 1800

    The script also accepts an already-decompressed CSV in any of the two
    layouts below, and will tell you clearly if the columns are not what it
    expects.

FORMAT PIPELINE (this is the pipeline described in the report)
    PGN (text) -> FEN (text, one position) -> [3, 12, -1, 0, 5, 2, 0.5]
    (list of numbers, Module 1) -> evaluation function (Module 4)

    The data is fed in as PLAIN NUMBERS, not binary.  Binary board encodings
    (8x8x12 planes) are only for neural-network approaches, which this project
    does not use.
"""

import argparse
import csv
import os
import sys

import chess

OUTPUT_FILE = "positions.csv"
DEFAULT_LIMIT = 20000
DEFAULT_MIN_RATING = 1500

# The Lichess puzzle CSV header, as published.
LICHESS_COLUMNS = ["PuzzleId", "FEN", "Moves", "Rating"]


def looks_like_lichess_puzzles(fieldnames):
    return fieldnames is not None and "FEN" in fieldnames and "Moves" in fieldnames


def looks_like_plain_fen_move(fieldnames):
    return fieldnames is not None and "fen" in fieldnames and "best_move" in fieldnames


def rows_from_lichess(reader, limit, min_rating):
    """Convert Lichess puzzle rows into (fen, best_move) training rows."""
    kept = 0
    skipped_rating = 0
    skipped_bad = 0

    for row in reader:
        if kept >= limit:
            break

        try:
            rating = int(row.get("Rating") or 0)
        except ValueError:
            rating = 0
        if rating < min_rating:
            skipped_rating += 1
            continue

        fen = (row.get("FEN") or "").strip()
        moves = (row.get("Moves") or "").split()
        if not fen or len(moves) < 2:
            skipped_bad += 1
            continue

        try:
            board = chess.Board(fen)
            # Move 1 is the opponent's blunder; play it.  Move 2 is the move
            # the strong player found -- that is our label.
            opponent_move = chess.Move.from_uci(moves[0])
            if opponent_move not in board.legal_moves:
                skipped_bad += 1
                continue
            board.push(opponent_move)

            solution = chess.Move.from_uci(moves[1])
            if solution not in board.legal_moves:
                skipped_bad += 1
                continue
        except ValueError:
            skipped_bad += 1
            continue

        kept += 1
        yield board.fen(), solution.uci()

    print("  kept {} positions (skipped {} below the rating cut, {} unusable)"
          .format(kept, skipped_rating, skipped_bad))


def rows_from_plain(reader, limit):
    """Pass through a CSV that is already in fen,best_move form (validating it)."""
    kept = 0
    skipped_bad = 0
    for row in reader:
        if kept >= limit:
            break
        fen = (row.get("fen") or "").strip()
        uci = (row.get("best_move") or "").strip()
        try:
            board = chess.Board(fen)
            if chess.Move.from_uci(uci) not in board.legal_moves:
                skipped_bad += 1
                continue
        except ValueError:
            skipped_bad += 1
            continue
        kept += 1
        yield fen, uci
    print("  kept {} positions ({} unusable)".format(kept, skipped_bad))


def main():
    parser = argparse.ArgumentParser(
        description="Build positions.csv for the GA's 'positions' fitness mode.")
    parser.add_argument("input_csv", nargs="?",
                        help="decompressed Lichess puzzle CSV "
                             "(lichess_db_puzzle.csv)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="how many positions to keep (default {})"
                             .format(DEFAULT_LIMIT))
    parser.add_argument("--min-rating", type=int, default=DEFAULT_MIN_RATING,
                        help="skip puzzles below this rating (default {})"
                             .format(DEFAULT_MIN_RATING))
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help="output file (default {})".format(OUTPUT_FILE))
    args = parser.parse_args()

    if not args.input_csv:
        print(__doc__)
        print("No input file given.  See the USAGE section above.")
        sys.exit(1)

    if not os.path.exists(args.input_csv):
        print("ERROR: {} does not exist.".format(args.input_csv))
        if args.input_csv.endswith(".zst"):
            print("That looks like the compressed file -- decompress it first:")
            print("    zstd -d {}".format(args.input_csv))
        sys.exit(1)

    if args.input_csv.endswith(".zst"):
        print("ERROR: {} is still compressed.".format(args.input_csv))
        print("Decompress it first:  zstd -d {}".format(args.input_csv))
        sys.exit(1)

    print("reading  : {}".format(args.input_csv))
    print("limit    : {} positions".format(args.limit))
    print("min rating: {}".format(args.min_rating))

    with open(args.input_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        if looks_like_lichess_puzzles(reader.fieldnames):
            print("format   : Lichess puzzle database")
            rows = rows_from_lichess(reader, args.limit, args.min_rating)
        elif looks_like_plain_fen_move(reader.fieldnames):
            print("format   : plain fen,best_move")
            rows = rows_from_plain(reader, args.limit)
        else:
            print("ERROR: unrecognised columns: {}".format(reader.fieldnames))
            print("Expected either the Lichess puzzle header ({}) or a plain "
                  "'fen,best_move' header.".format(", ".join(LICHESS_COLUMNS)))
            sys.exit(1)

        with open(args.output, "w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["fen", "best_move"])
            written = 0
            for fen, uci in rows:
                writer.writerow([fen, uci])
                written += 1

    print()
    print("wrote {} with {} positions.".format(args.output, written))
    print()
    print("Now set  FITNESS_MODE = \"positions\"  at the top of train_ga.py "
          "and run:")
    print("    python train_ga.py")


if __name__ == "__main__":
    main()
