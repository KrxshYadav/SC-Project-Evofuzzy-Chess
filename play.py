"""
play.py -- the human interface.  Terminal only, no GUI library.

You play against the engine (Modules 1-4 and 6) using either the GA-tuned
weights from best_weights.txt or the untuned baseline, so you can feel the
difference the genetic algorithm made.

Type moves in either notation:
    SAN   e4, Nf3, O-O, exd5, e8=Q
    UCI   e2e4, g1f3, e1g1, e7e8q

Commands:
    help    this list            board   redraw the board
    moves   list legal moves     eval    show features, fuzzy phase and score
    fen     print the FEN        undo    take back your last move
    flip    flip the board view  resign  end the game
    quit    exit

Run:  python play.py
"""

import sys
import time

import chess

import engine

DEFAULT_DEPTH = 3


# ---------------------------------------------------------------------------
# Small terminal helpers
# ---------------------------------------------------------------------------

def ask(prompt, valid, default):
    """Ask until the answer is in `valid`.  Empty input takes the default."""
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return default
        if answer == "":
            return default
        if answer in valid:
            return answer
        print("Please answer one of: {}".format(", ".join(valid)))


def ask_int(prompt, low, high, default):
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            return default
        if answer == "":
            return default
        try:
            value = int(answer)
        except ValueError:
            print("Please type a number.")
            continue
        if low <= value <= high:
            return value
        print("Please choose a number between {} and {}.".format(low, high))


def show_position(board, unicode_pieces, flipped, weights, human_msg="", engine_msg=""):
    print()
    board_lines = engine.board_to_text(board, unicode_pieces, flipped).split('\n')
    opening, middlegame, endgame = engine.game_phase(board)
    
    info_lines = [
        "",
        "    Move {}".format(board.fullmove_number),
        "    Material on board: {:.0f}/78".format(engine.total_material(board)),
        "    Fuzzy phase: Opening {:.0f}% | Middlegame {:.0f}% | Endgame {:.0f}%".format(100 * opening, 100 * middlegame, 100 * endgame),
        "    Static evaluation: {:+.2f} (+ = White better)".format(engine.evaluate_white(board, weights)),
        "",
        "    " + human_msg if human_msg else "",
        "    " + engine_msg if engine_msg else "",
    ]
    
    for i in range(max(len(board_lines), len(info_lines))):
        left = board_lines[i] if i < len(board_lines) else " " * 30
        right = info_lines[i] if i < len(info_lines) else ""
        print(left + right)
    print()


def show_evaluation_detail(board, weights):
    """MODULE 1 + 4 + 6, laid out term by term.  Useful in the lab demo."""
    features = engine.extract_features(board)
    phase = engine.game_phase(board)
    adjusted = engine.fuzzy_adjust_weights(weights, phase)
    print()
    print("  fuzzy phase: opening {:.0f}%  middlegame {:.0f}%  endgame {:.0f}%"
          .format(*[100 * p for p in phase]))
    print("  {:<16} {:>9} {:>9} {:>9} {:>9}".format(
        "feature", "value", "base w", "fuzzy w", "term"))
    total = 0.0
    for name, value, base, fuzzy_weight in zip(engine.FEATURE_NAMES, features,
                                               weights, adjusted):
        term = value * fuzzy_weight
        total += term
        print("  {:<16} {:>9.3f} {:>9.3f} {:>9.3f} {:>9.3f}".format(
            name, value, base, fuzzy_weight, term))
    print("  {:<16} {:>39.3f}".format("TOTAL (White)", total))
    print()


def parse_move(board, text):
    """Accept SAN or UCI.  Returns a legal Move, or None."""
    try:
        move = board.parse_san(text)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    try:
        move = chess.Move.from_uci(text.lower())
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    return None


def announce_result(board, human_colour):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return
    print()
    print("=" * 46)
    if outcome.winner is None:
        print("  GAME OVER -- draw ({})".format(outcome.termination.name.lower()))
    elif outcome.winner == human_colour:
        print("  GAME OVER -- you win by {}!".format(
            outcome.termination.name.lower()))
    else:
        print("  GAME OVER -- the engine wins by {}.".format(
            outcome.termination.name.lower()))
    print("  final position: {}".format(board.fen()))
    print("=" * 46)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 66)
    print("  SOFT COMPUTING CHESS  --  GA-tuned + Fuzzy Logic evaluation")
    print("=" * 66)

    tuned = engine.load_weights()
    if tuned is None:
        print("No {} found -- playing with the UNTUNED baseline weights."
              .format(engine.WEIGHTS_FILE))
        print("(Run  python train_ga.py  to produce the tuned weights.)")
        weights = list(engine.BASELINE_WEIGHTS)
        which = "baseline (untuned)"
    else:
        choice = ask("Play against [t]uned weights or [b]aseline? [t]: ",
                     ("t", "b"), "t")
        if choice == "t":
            weights = tuned
            which = "GA-tuned"
        else:
            weights = list(engine.BASELINE_WEIGHTS)
            which = "baseline (untuned)"

    print()
    print("Engine weights in use ({}):".format(which))
    for name, value in zip(engine.FEATURE_NAMES, weights):
        print("    {:<16} {:.3f}".format(name, value))
    print()

    colour_choice = ask("Play as [w]hite or [b]lack? [w]: ", ("w", "b"), "w")
    human_colour = chess.WHITE if colour_choice == "w" else chess.BLACK
    depth = ask_int("Search depth 1-4 [{}]: ".format(DEFAULT_DEPTH), 1, 4,
                    DEFAULT_DEPTH)
    unicode_choice = ask("Use unicode chess pieces? [y]/n: ", ("y", "n"), "y")
    unicode_pieces = (unicode_choice == "y")

    board = chess.Board()
    flipped = (human_colour == chess.BLACK)

    print()
    print("You are {}.  Engine depth {}.  Type 'help' for commands."
          .format("White" if human_colour == chess.WHITE else "Black", depth))

    human_msg = ""
    engine_msg = ""

    while not board.is_game_over(claim_draw=True):
        show_position(board, unicode_pieces, flipped, weights, human_msg, engine_msg)

        if board.turn == human_colour:
            # ---------------- human turn ----------------
            try:
                text = input("your move> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye.")
                return
            if text == "":
                continue
            command = text.lower()

            if command in ("quit", "exit"):
                print("bye.")
                return
            if command == "resign":
                print("You resign.  The engine wins.")
                return
            if command == "help":
                print(__doc__)
                continue
            if command == "board":
                continue
            if command == "fen":
                print(board.fen())
                continue
            if command == "flip":
                flipped = not flipped
                continue
            if command == "moves":
                human_msg = "Legal moves: " + " ".join(board.san(m) for m in board.legal_moves)
                continue
            if command == "eval":
                show_evaluation_detail(board, weights)
                continue
            if command == "undo":
                if len(board.move_stack) >= 2:
                    board.pop()      # engine's reply
                    board.pop()      # your move
                    human_msg = "Took back one full move."
                    engine_msg = ""
                else:
                    human_msg = "Nothing to take back."
                continue

            move = parse_move(board, text)
            if move is None:
                human_msg = "'{}' is not a legal move. Type 'moves' for help.".format(text)
                continue
            
            human_msg = "You played: {}".format(board.san(move))
            board.push(move)

        else:
            # ---------------- engine turn ----------------
            # Clear previous messages when engine thinks
            human_msg = ""
            print("engine thinking (depth {})...".format(depth), end="\r", flush=True)
            started = time.time()
            move, score, stats = engine.search(board, weights, depth=depth)
            elapsed = time.time() - started
            if move is None:
                break
            san = board.san(move)
            board.push(move)
            
            # Pad the string with spaces to overwrite the "engine thinking" line perfectly
            engine_msg = "Engine plays: {}  (score {:+.2f}, {:.1f}s)".format(san, score, elapsed)
            print(" " * 50, end="\r", flush=True)

    show_position(board, unicode_pieces, flipped, weights, human_msg, engine_msg)
    announce_result(board, human_colour)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye.")
        sys.exit(0)
