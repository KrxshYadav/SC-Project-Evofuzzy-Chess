"""
compare.py -- THE EXPERIMENT FOR THE REPORT.

    GA-tuned engine  vs  untuned baseline engine.
    How much did the genetic algorithm actually improve play?

Both sides use exactly the same code (Modules 1-4 and 6) and the same search
depth.  The ONLY difference is the 7 evaluation weights:

    tuned    = best_weights.txt   (produced by train_ga.py, Module 5)
    baseline = engine.BASELINE_WEIGHTS  (the control group -- never delete it)

Every opening in the book is played TWICE, once with each engine as White, so
the result cannot be a colour bias.

Outputs:
    results.txt   human-readable summary table for the report
    results.csv   one row per game, for an appendix or a plot

Run:  python compare.py
"""

import concurrent.futures
import csv
import math
import os
import sys
import time

import chess

import engine

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

NUM_GAMES = 50          # 25 openings x 2 colours
DEPTH = 3               # play strength depth (training used 2, for speed)
MAX_PLIES = 120         # unfinished games are adjudicated on material
ADJUDICATION_MARGIN = 1.5

RESULTS_TXT = "results.txt"
RESULTS_CSV = "results.csv"

# The games are independent of each other, so they can be played on several CPU
# cores at once.  This is pure speed -- the games themselves are deterministic,
# so the result is identical either way.  Set WORKERS = 1 for a plain
# sequential loop.
WORKERS = max(1, min(10, (os.cpu_count() or 1) - 2))

# ---------------------------------------------------------------------------
# Opening book: 25 standard lines.  The engines are deterministic, so without
# an opening book every game would be an identical replay of the first one.
# ---------------------------------------------------------------------------

OPENING_BOOK = [
    ("Start position",        []),
    ("Open Game",             ["e4", "e5", "Nf3", "Nc6"]),
    ("Ruy Lopez",             ["e4", "e5", "Nf3", "Nc6", "Bb5"]),
    ("Italian Game",          ["e4", "e5", "Nf3", "Nc6", "Bc4"]),
    ("Petrov Defence",        ["e4", "e5", "Nf3", "Nf6"]),
    ("Sicilian",              ["e4", "c5"]),
    ("Sicilian, d6",          ["e4", "c5", "Nf3", "d6"]),
    ("Sicilian, Nc6",         ["e4", "c5", "Nf3", "Nc6"]),
    ("French Defence",        ["e4", "e6", "d4", "d5"]),
    ("Caro-Kann",             ["e4", "c6", "d4", "d5"]),
    ("Scandinavian",          ["e4", "d5"]),
    ("Alekhine Defence",      ["e4", "Nf6"]),
    ("Pirc Defence",          ["e4", "d6"]),
    ("Modern Defence",        ["e4", "g6"]),
    ("Queen's Gambit",        ["d4", "d5", "c4"]),
    ("QGD",                   ["d4", "d5", "c4", "e6"]),
    ("Slav Defence",          ["d4", "d5", "c4", "c6"]),
    ("Indian Defence",        ["d4", "Nf6", "c4", "e6"]),
    ("King's Indian",         ["d4", "Nf6", "c4", "g6"]),
    ("Queen's Indian",        ["d4", "Nf6", "c4", "e6", "Nf3", "b6"]),
    ("Dutch Defence",         ["d4", "f5"]),
    ("Symmetrical d4/Nf3",    ["d4", "d5", "Nf3", "Nf6"]),
    ("English Opening",       ["c4", "e5"]),
    ("Symmetrical English",   ["c4", "c5"]),
    ("Reti Opening",          ["Nf3", "d5", "g3", "Nf6"]),
]


def opening_board(line):
    board = chess.Board()
    for san in line:
        board.push_san(san)
    return board


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def play_game(white_weights, black_weights, opening_line, depth=DEPTH):
    """Returns (white_points, plies, final_material, how_it_ended)."""
    board = opening_board(opening_line)
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        weights = white_weights if board.turn == chess.WHITE else black_weights
        move = engine.choose_move(board, weights, depth=depth)
        if move is None:
            break
        board.push(move)
        plies += 1

    material = engine._material_balance(board)
    outcome = board.outcome(claim_draw=True)

    if outcome is not None:
        reason = outcome.termination.name.lower()
        if outcome.winner is None:
            return 0.5, plies, material, reason
        return (1.0 if outcome.winner == chess.WHITE else 0.0), plies, material, reason

    # Move cap reached -- adjudicate on material, as a tournament arbiter would.
    if material >= ADJUDICATION_MARGIN:
        return 1.0, plies, material, "adjudicated"
    if material <= -ADJUDICATION_MARGIN:
        return 0.0, plies, material, "adjudicated"
    return 0.5, plies, material, "adjudicated"


def run_one_game(job):
    """Play game number `index` of the match.  A plain module-level function
    taking one tuple, so it can also be handed to a worker process."""
    index, tuned, baseline = job
    opening_name, opening_line = OPENING_BOOK[(index // 2) % len(OPENING_BOOK)]
    tuned_is_white = (index % 2 == 0)

    started = time.time()
    if tuned_is_white:
        white_points, plies, material, reason = play_game(tuned, baseline,
                                                          opening_line)
        tuned_points = white_points
    else:
        white_points, plies, material, reason = play_game(baseline, tuned,
                                                          opening_line)
        tuned_points = 1.0 - white_points

    return {
        "game": index + 1,
        "opening": opening_name,
        "tuned_colour": "white" if tuned_is_white else "black",
        "result_for_tuned": ("win" if tuned_points == 1.0
                             else "draw" if tuned_points == 0.5 else "loss"),
        "points_for_tuned": tuned_points,
        "plies": plies,
        "final_material_white": round(material, 1),
        "termination": reason,
        "seconds": round(time.time() - started, 1),
    }


def elo_difference(score_rate):
    """Convert a score rate (0..1) into an approximate Elo difference.

    Standard logistic formula:  dElo = -400 * log10(1/score - 1).
    Undefined at 0% and 100%, so those are reported as bounds.
    """
    if score_rate <= 0.0:
        return float("-inf")
    if score_rate >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / score_rate - 1.0)


def format_elo(score_rate):
    value = elo_difference(score_rate)
    if value == float("inf"):
        return ">+800 (won every game)"
    if value == float("-inf"):
        return "<-800 (lost every game)"
    return "{:+.1f}".format(value)


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def main():
    tuned = engine.load_weights()
    if tuned is None:
        print("ERROR: {} not found or unreadable.".format(engine.WEIGHTS_FILE))
        print("Run  python train_ga.py  first -- it produces the tuned weights.")
        sys.exit(1)

    baseline = engine.BASELINE_WEIGHTS

    header = []
    header.append("=" * 74)
    header.append("EXPERIMENT: GA-TUNED ENGINE  vs  UNTUNED BASELINE")
    header.append("=" * 74)
    header.append("games        : {}  ({} openings, each played twice with "
                  "colours reversed)".format(NUM_GAMES, (NUM_GAMES + 1) // 2))
    header.append("search depth : {} (identical for both sides)".format(DEPTH))
    header.append("move cap     : {} plies, then adjudicated on material"
                  .format(MAX_PLIES))
    header.append("cpu workers  : {}".format(WORKERS))
    header.append("")
    header.append("{:<16} {:>8} {:>8}".format("feature", "tuned", "baseline"))
    for name, t, b in zip(engine.FEATURE_NAMES, tuned, baseline):
        header.append("{:<16} {:>8.3f} {:>8.3f}".format(name, t, b))
    header.append("-" * 74)
    print("\n".join(header), flush=True)

    wins = draws = losses = 0
    rows = []
    started = time.time()

    jobs = [(index, tuned, baseline) for index in range(NUM_GAMES)]
    if WORKERS > 1:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS)
        finished = pool.map(run_one_game, jobs)      # yields in game order
    else:
        pool = None
        finished = (run_one_game(job) for job in jobs)

    # Tally in game order, so the running score reads the same however many
    # cores the match was played on.  Each game is appended to results.csv the
    # moment it finishes, so a long match that gets interrupted still leaves
    # every game it did play on disk.
    csv_handle = open(RESULTS_CSV, "w", encoding="utf-8", newline="")
    writer = None
    try:
        for row in finished:
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(csv_handle, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            csv_handle.flush()

            if row["result_for_tuned"] == "win":
                wins += 1
                label = "WIN "
            elif row["result_for_tuned"] == "draw":
                draws += 1
                label = "DRAW"
            else:
                losses += 1
                label = "LOSS"
            played = row["game"]
            score = wins + 0.5 * draws
            print("game {:>3}/{:<3}  {:<22} tuned as {:<5}  {}   "
                  "running: {:>2}W {:>2}D {:>2}L  score {:.1f}/{:<3}  "
                  "({:.0f}s, {} plies, {})".format(
                      played, NUM_GAMES, row["opening"], row["tuned_colour"],
                      label, wins, draws, losses, score, played, row["seconds"],
                      row["plies"], row["termination"]), flush=True)
    finally:
        csv_handle.close()
        if pool is not None:
            pool.shutdown()

    if not rows:
        print("No games completed.")
        sys.exit(1)
    games_played = len(rows)

    elapsed = time.time() - started
    score = wins + 0.5 * draws
    score_rate = score / games_played

    summary = []
    summary.append("-" * 74)
    summary.append("RESULT  (from the GA-tuned engine's point of view)")
    summary.append("-" * 74)
    summary.append("{:<24}{:>10}".format("games played", games_played))
    summary.append("{:<24}{:>10}".format("wins", wins))
    summary.append("{:<24}{:>10}".format("draws", draws))
    summary.append("{:<24}{:>10}".format("losses", losses))
    summary.append("{:<24}{:>10}".format("score",
                                         "{:.1f} / {}".format(score, games_played)))
    summary.append("{:<24}{:>10}".format("score rate",
                                         "{:.1f}%".format(100 * score_rate)))
    summary.append("{:<24}{:>10}".format("approx. Elo gain", format_elo(score_rate)))
    summary.append("")
    summary.append("total time: {:.1f} min ({:.1f}s per game average)"
                   .format(elapsed / 60, elapsed / games_played))
    summary.append("")
    if score_rate > 0.5:
        summary.append("The GA-tuned weights outperform the untuned baseline.")
    elif score_rate == 0.5:
        summary.append("Dead level -- the GA has not separated itself from the "
                       "baseline yet.  Train for more generations.")
    else:
        summary.append("The baseline is still ahead.  Per project documentation: train for "
                       "more generations, the baseline weights are decent.")
    summary_text = "\n".join(summary)
    print(summary_text, flush=True)

    with open(RESULTS_TXT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        for row in rows:
            handle.write("game {:>3}  {:<22} tuned as {:<5}  {:<5} "
                         "{:>4} plies  {}\n".format(
                             row["game"], row["opening"], row["tuned_colour"],
                             row["result_for_tuned"], row["plies"],
                             row["termination"]))
        handle.write(summary_text + "\n")

    print()
    print("wrote {} and {}".format(RESULTS_TXT, RESULTS_CSV))


if __name__ == "__main__":
    main()
