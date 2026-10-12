"""
engine.py -- Soft Computing chess engine.

Contains ARCHITECTURE MODULES 1, 2, 3, 4 and 6:

    MODULE 1  Board Input .......... position -> 7 numeric features
    MODULE 2  Move Generator ....... legal moves (delegated to python-chess)
    MODULE 3  Search ............... minimax / negamax + alpha-beta + quiescence
    MODULE 4  Evaluation Function .. score = w1*f1 + w2*f2 + ... + w7*f7
    MODULE 6  Fuzzy Logic .......... Mamdani phase detection, adjusts the weights

MODULE 5 (the Genetic Algorithm) lives in train_ga.py.

Only external dependency: python-chess.
Sign convention everywhere in this file: POSITIVE = WHITE IS BETTER.
"""

import math
import chess

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,          # kings are never counted as material
}

FEATURE_NAMES = [
    "material",
    "mobility",
    "king_safety",
    "king_activity",
    "doubled_pawns",
    "passed_pawns",
    "centre_control",
]
NUM_FEATURES = len(FEATURE_NAMES)

# The untuned baseline weights.  THIS IS THE CONTROL GROUP for the experiment
# in compare.py -- do not delete it, without it there is no result to report.
BASELINE_WEIGHTS = [1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 0.5]

# GA chromosome bounds (Module 5 searches inside this box).
WEIGHT_MIN = 0.0
WEIGHT_MAX = 3.0

CENTRE_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

MATE_SCORE = 100000.0

# Maximum material on the board, kings excluded:
#   16 pawns*1 + 4 knights*3 + 4 bishops*3 + 4 rooks*5 + 2 queens*9 = 78
MAX_MATERIAL = 78.0


# ===========================================================================
# MODULE 1 -- BOARD INPUT
# Converts a position into a list of 7 numbers.  Positive = White is better.
# Every feature is pre-scaled so that all 7 live on a comparable numeric
# range; that is what makes a single GA weight range of 0..3 meaningful.
# ===========================================================================

def _material_balance(board):
    """Feature 1: material, in pawn units.  White total minus Black total."""
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        if value == 0.0:
            continue
        score += value * len(board.pieces(piece_type, chess.WHITE))
        score -= value * len(board.pieces(piece_type, chess.BLACK))
    return score


def _legal_move_count(board, colour):
    """Number of legal moves colour would have if it were that colour's turn.

    Flipping board.turn is safe here: in any legal position the side that is
    NOT to move cannot have its king in check, so the flipped position is
    still a valid one to generate moves from.
    """
    original = board.turn
    board.turn = colour
    count = board.legal_moves.count()
    board.turn = original
    return count


def _mobility(board):
    """Feature 2: mobility.  (White legal moves - Black legal moves) / 10."""
    return (_legal_move_count(board, chess.WHITE)
            - _legal_move_count(board, chess.BLACK)) / 10.0


def _king_zone(square):
    """The king square plus every square adjacent to it."""
    zone = [square]
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            f, r = file_index + df, rank_index + dr
            if 0 <= f <= 7 and 0 <= r <= 7 and not (df == 0 and dr == 0):
                zone.append(chess.square(f, r))
    return zone


def _king_safety_one_side(board, colour):
    """Pawn shield around the king, minus enemy pressure on the king zone."""
    king_square = board.king(colour)
    if king_square is None:                      # cannot happen in a real game
        return 0.0
    zone = _king_zone(king_square)
    shield = sum(
        1 for sq in zone
        if board.piece_type_at(sq) == chess.PAWN and board.color_at(sq) == colour
    )
    pressure = sum(1 for sq in zone if board.is_attacked_by(not colour, sq))
    return shield - pressure


def _king_safety(board):
    """Feature 3: king safety.  (White safety - Black safety) / 4."""
    return (_king_safety_one_side(board, chess.WHITE)
            - _king_safety_one_side(board, chess.BLACK)) / 4.0


def _king_activity_one_side(board, colour):
    """How centralised the king is.  Matters only in the endgame -- Module 6
    is what switches this feature on and off."""
    king_square = board.king(colour)
    if king_square is None:
        return 0.0
    distance = min(chess.square_distance(king_square, sq) for sq in CENTRE_SQUARES)
    return 6.0 - distance


def _king_activity(board):
    """Feature 4: king activity.  (White - Black) / 3."""
    return (_king_activity_one_side(board, chess.WHITE)
            - _king_activity_one_side(board, chess.BLACK)) / 3.0


def _doubled_pawn_count(board, colour):
    """Surplus pawns sharing a file (2 pawns on one file counts as 1)."""
    surplus = 0
    for file_index in range(8):
        pawns_on_file = sum(
            1 for sq in board.pieces(chess.PAWN, colour)
            if chess.square_file(sq) == file_index
        )
        if pawns_on_file > 1:
            surplus += pawns_on_file - 1
    return surplus


def _doubled_pawns(board):
    """Feature 5: doubled pawns.  Black's surplus minus White's, so that --
    like every other feature -- a POSITIVE value means White is better."""
    return float(_doubled_pawn_count(board, chess.BLACK)
                 - _doubled_pawn_count(board, chess.WHITE))


def _passed_pawn_count(board, colour):
    """A pawn is passed if no enemy pawn stands on its file or an adjacent
    file anywhere ahead of it."""
    enemy_pawns = board.pieces(chess.PAWN, not colour)
    direction = 1 if colour == chess.WHITE else -1
    passed = 0
    for square in board.pieces(chess.PAWN, colour):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        blocked = False
        for enemy in enemy_pawns:
            if abs(chess.square_file(enemy) - file_index) > 1:
                continue
            ahead = (chess.square_rank(enemy) - rank_index) * direction
            if ahead > 0:
                blocked = True
                break
        if not blocked:
            passed += 1
    return passed


def _passed_pawns(board):
    """Feature 6: passed pawns.  White passed pawns minus Black's."""
    return float(_passed_pawn_count(board, chess.WHITE)
                 - _passed_pawn_count(board, chess.BLACK))


def _centre_control(board):
    """Feature 7: centre control.  Attackers of d4/e4/d5/e5, White - Black,
    divided by 4."""
    score = 0
    for square in CENTRE_SQUARES:
        score += len(board.attackers(chess.WHITE, square))
        score -= len(board.attackers(chess.BLACK, square))
    return score / 4.0


def extract_features(board):
    """MODULE 1 entry point.  Position -> list of 7 numbers."""
    return [
        _material_balance(board),
        _mobility(board),
        _king_safety(board),
        _king_activity(board),
        _doubled_pawns(board),
        _passed_pawns(board),
        _centre_control(board),
    ]


def total_material(board):
    """Total non-king material still on the board, 0..78.  Fuzzy input #1."""
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        if value == 0.0:
            continue
        total += value * (len(board.pieces(piece_type, chess.WHITE))
                          + len(board.pieces(piece_type, chess.BLACK)))
    return total


# ===========================================================================
# MODULE 6 -- FUZZY LOGIC  (soft computing technique #2)
#
# Two crisp inputs -> fuzzification -> Mamdani rules (AND = min, OR = max)
# -> three phase membership degrees that sum to 1.0.
#
# Why fuzzy instead of "if pieces < 10 then endgame": a hard rule makes the
# weights jump at one arbitrary threshold.  Fuzzy lets a position be 30%
# middlegame and 70% endgame at the same time, so the weights shift smoothly.
# ===========================================================================

def _triangle(x, left, peak, right):
    """Triangular membership function."""
    if x <= left or x >= right:
        return 0.0
    if x == peak:
        return 1.0
    if x < peak:
        return (x - left) / (peak - left)
    return (right - x) / (right - peak)


def _shoulder_down(x, full_until, zero_at):
    """Left shoulder: 1.0 up to full_until, then ramps down to 0 at zero_at."""
    if x <= full_until:
        return 1.0
    if x >= zero_at:
        return 0.0
    return (zero_at - x) / (zero_at - full_until)


def _shoulder_up(x, zero_until, full_at):
    """Right shoulder: 0.0 up to zero_until, then ramps up to 1 at full_at."""
    if x <= zero_until:
        return 0.0
    if x >= full_at:
        return 1.0
    return (x - zero_until) / (full_at - zero_until)


def fuzzify_material(material):
    """Crisp material (0..78) -> LOW / MEDIUM / HIGH membership degrees."""
    return {
        "LOW": _shoulder_down(material, 0.0, 30.0),
        "MEDIUM": _triangle(material, 20.0, 45.0, 70.0),
        "HIGH": _shoulder_up(material, 40.0, 80.0),
    }


def fuzzify_moves(move_number):
    """Crisp move number -> EARLY / MIDDLE / LATE membership degrees."""
    return {
        "EARLY": _shoulder_down(move_number, 8.0, 24.0),
        "MIDDLE": _triangle(move_number, 5.0, 18.0, 40.0),
        "LATE": _shoulder_up(move_number, 30.0, 45.0),
    }


def fire_rules(material_mf, moves_mf):
    """The Mamdani rule base, AND = min, OR = max.

        R1  IF material HIGH   AND moves EARLY  THEN OPENING
        R2  IF material MEDIUM AND moves MIDDLE THEN MIDDLEGAME
        R3  IF material LOW    OR  moves LATE   THEN ENDGAME

    Returns the three un-normalised rule strengths.
    """
    opening = min(material_mf["HIGH"], moves_mf["EARLY"])        # R1
    middlegame = min(material_mf["MEDIUM"], moves_mf["MIDDLE"])  # R2
    endgame = max(material_mf["LOW"], moves_mf["LATE"])          # R3
    return opening, middlegame, endgame


def phase_from_inputs(material, move_number):
    """Fuzzy phase for raw crisp inputs (used by the self-test and the report)."""
    opening, middlegame, endgame = fire_rules(fuzzify_material(material),
                                              fuzzify_moves(move_number))
    total = opening + middlegame + endgame
    if total == 0.0:
        # No rule fired at all (a rare corner of the input space, e.g. a full
        # board still standing at move 30).  Fall back to "purely middlegame"
        # rather than dividing by zero.
        return (0.0, 1.0, 0.0)
    return (opening / total, middlegame / total, endgame / total)


def game_phase(board):
    """MODULE 6 entry point.

    Returns (opening, middlegame, endgame) membership degrees summing to 1.0.
    """
    return phase_from_inputs(total_material(board), board.fullmove_number)


# Phase-specific multipliers, one row per feature, columns = opening / middle /
# endgame.  These encode ordinary chess knowledge; the GA tunes the BASE
# weights, the fuzzy system decides how much of each column to apply.
PHASE_MULTIPLIERS = {
    "material":       (1.0, 1.0, 1.0),
    "mobility":       (1.0, 1.2, 0.8),
    "king_safety":    (1.2, 1.4, 0.2),   # decisive in the middlegame, irrelevant in the endgame
    "king_activity":  (0.0, 0.3, 1.5),   # off in the opening, the king marches in the endgame
    "doubled_pawns":  (0.6, 1.0, 1.3),
    "passed_pawns":   (0.4, 0.8, 2.0),
    "centre_control": (1.3, 1.0, 0.4),
}


def fuzzy_adjust_weights(base_weights, phase):
    """Blend the phase multipliers with the fuzzy memberships, then apply them
    to the GA-learned base weights."""
    opening, middlegame, endgame = phase
    adjusted = []
    for index, name in enumerate(FEATURE_NAMES):
        m_open, m_mid, m_end = PHASE_MULTIPLIERS[name]
        multiplier = opening * m_open + middlegame * m_mid + endgame * m_end
        adjusted.append(base_weights[index] * multiplier)
    return adjusted


# ===========================================================================
# MODULE 4 -- EVALUATION FUNCTION
#   score = w1*f1 + w2*f2 + ... + w7*f7
# The weights are what Modules 5 and 6 exist to produce.
# ===========================================================================

def evaluate_white(board, weights, use_fuzzy=True):
    """Static evaluation from WHITE's point of view, in pawn units."""
    features = extract_features(board)
    if use_fuzzy:
        weights = fuzzy_adjust_weights(weights, game_phase(board))
    return sum(w * f for w, f in zip(weights, features))


def evaluate(board, weights, use_fuzzy=True):
    """Static evaluation from the SIDE TO MOVE's point of view.

    Negamax needs this sign convention; everything else in the project uses
    the White-positive convention of evaluate_white().
    """
    score = evaluate_white(board, weights, use_fuzzy)
    return score if board.turn == chess.WHITE else -score


# ===========================================================================
# MODULE 2 -- MOVE GENERATOR
# python-chess handles legality, check, mate, stalemate, castling, en passant
# and promotion.  We do not reimplement any of it; we only ORDER the moves,
# because trying captures first makes alpha-beta prune far more.
# ===========================================================================

def _move_score_for_ordering(board, move):
    """MVV-LVA style key: most valuable victim, least valuable attacker."""
    if board.is_capture(move):
        if board.is_en_passant(move):
            victim = 1.0
        else:
            captured = board.piece_type_at(move.to_square)
            victim = PIECE_VALUES.get(captured, 0.0)
        attacker = PIECE_VALUES.get(board.piece_type_at(move.from_square), 0.0)
        return 1000.0 + victim * 10.0 - attacker
    if move.promotion:
        return 900.0
    return 0.0


def ordered_moves(board, captures_only=False):
    """MODULE 2 entry point.  Legal moves, best-looking ones first."""
    if captures_only:
        moves = [m for m in board.legal_moves
                 if board.is_capture(m) or m.promotion]
    else:
        moves = list(board.legal_moves)
    moves.sort(key=lambda m: _move_score_for_ordering(board, m), reverse=True)
    return moves


# ===========================================================================
# MODULE 3 -- SEARCH
# Negamax-style minimax with alpha-beta pruning, plus a quiescence search at
# the leaves so the engine does not hang pieces exactly at the depth limit
# (the classic "horizon effect").
# ===========================================================================

class SearchStats:
    """Small counter object so play.py / compare.py can report node counts."""

    def __init__(self):
        self.nodes = 0
        self.quiescence_nodes = 0


def _terminal_score(board, ply):
    """Score of a finished game from the side-to-move's point of view, or None
    if the game is not over.  Mates found sooner score higher."""
    if board.is_checkmate():
        return -(MATE_SCORE - ply)
    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
            or board.is_repetition(3)):
        return 0.0
    return None


def quiescence(board, alpha, beta, weights, ply, stats, use_fuzzy=True,
               max_extra_ply=6):
    """Search only captures and promotions until the position is quiet."""
    stats.quiescence_nodes += 1

    terminal = _terminal_score(board, ply)
    if terminal is not None:
        return terminal

    stand_pat = evaluate(board, weights, use_fuzzy)
    if max_extra_ply <= 0:
        return stand_pat
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    for move in ordered_moves(board, captures_only=True):
        board.push(move)
        score = -quiescence(board, -beta, -alpha, weights, ply + 1, stats,
                            use_fuzzy, max_extra_ply - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def negamax(board, depth, alpha, beta, weights, ply, stats, use_fuzzy=True,
            use_quiescence=True):
    """Minimax in negamax form with alpha-beta pruning."""
    stats.nodes += 1

    terminal = _terminal_score(board, ply)
    if terminal is not None:
        return terminal

    if depth <= 0:
        if use_quiescence:
            return quiescence(board, alpha, beta, weights, ply, stats, use_fuzzy)
        return evaluate(board, weights, use_fuzzy)

    best = -math.inf
    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, weights, ply + 1,
                         stats, use_fuzzy, use_quiescence)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break                                   # alpha-beta cutoff
    return best


def search(board, weights, depth=3, use_fuzzy=True, use_quiescence=True):
    """MODULE 3 entry point.

    Returns (best_move, score, stats).  score is from the side-to-move's point
    of view.  Deterministic: move ordering is deterministic and the first
    strictly-best move wins, so the same position always produces the same
    move.  That is what makes the GA and the comparison experiment repeatable.
    """
    stats = SearchStats()
    best_move = None
    best_score = -math.inf
    alpha = -math.inf
    beta = math.inf

    for move in ordered_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, weights, 1, stats,
                         use_fuzzy, use_quiescence)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    return best_move, best_score, stats


def choose_move(board, weights, depth=3, use_fuzzy=True, use_quiescence=True):
    """Convenience wrapper used by play.py, train_ga.py and compare.py."""
    move, _score, _stats = search(board, weights, depth, use_fuzzy, use_quiescence)
    return move


def evaluate_all_moves(board, weights, depth=2, use_fuzzy=True, use_quiescence=True):
    """Score every legal move in the position.
    Returns a list of dicts sorted from best to worst move:
      [{"move": Move, "uci": str, "san": str, "score": float, "white_score": float}, ...]
    score is from the side-to-move's perspective; white_score is from White's perspective.
    """
    stats = SearchStats()
    scored = []
    is_white = (board.turn == chess.WHITE)
    for move in ordered_moves(board):
        san = board.san(move)
        board.push(move)
        if depth <= 1:
            if use_quiescence:
                score = -quiescence(board, -math.inf, math.inf, weights, 1, stats, use_fuzzy)
            else:
                score = -evaluate(board, weights, use_fuzzy)
        else:
            score = -negamax(board, depth - 1, -math.inf, math.inf, weights, 1, stats,
                             use_fuzzy, use_quiescence)
        board.pop()
        white_score = score if is_white else -score
        scored.append({
            "move": move,
            "uci": move.uci(),
            "san": san,
            "score": score,
            "white_score": white_score,
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def get_top_moves(board, weights, n=3, depth=2, use_fuzzy=True, use_quiescence=True):
    """Returns the top `n` legal moves and their evaluations."""
    all_moves = evaluate_all_moves(board, weights, depth, use_fuzzy, use_quiescence)
    return all_moves[:n]


def classify_move(best_score, move_score):
    """
    Classify a move compared to the engine's best move.
    best_score: score of the top move (from player's perspective)
    move_score: score of the chosen move (from player's perspective)
    Returns a dict with badge, label, icon, delta, description.
    """
    if best_score >= MATE_SCORE - 100:
        if move_score >= MATE_SCORE - 100:
            return {
                "badge": "best",
                "label": "Best Move",
                "icon": "⭐",
                "delta": 0.0,
                "description": "Maintains checkmate!",
            }
        else:
            return {
                "badge": "blunder",
                "label": "Blunder",
                "icon": "🔴",
                "delta": round(best_score - move_score, 2),
                "description": "Missed checkmate!",
            }

    delta = max(0.0, round(best_score - move_score, 2))

    if delta <= 0.05:
        return {
            "badge": "best",
            "label": "Best Move",
            "icon": "⭐",
            "delta": delta,
            "description": "Best move in this position.",
        }
    elif delta <= 0.25:
        return {
            "badge": "excellent",
            "label": "Excellent",
            "icon": "🟢",
            "delta": delta,
            "description": "Almost as strong as the best move.",
        }
    elif delta <= 0.55:
        return {
            "badge": "good",
            "label": "Good",
            "icon": "👍",
            "delta": delta,
            "description": "Solid practical move.",
        }
    elif delta <= 1.20:
        return {
            "badge": "inaccuracy",
            "label": "Inaccuracy",
            "icon": "🟡",
            "delta": delta,
            "description": "A slight inaccuracy that gives up some advantage.",
        }
    elif delta <= 2.20:
        return {
            "badge": "mistake",
            "label": "Mistake",
            "icon": "🟠",
            "delta": delta,
            "description": "A mistake that significantly worsens the position.",
        }
    else:
        return {
            "badge": "blunder",
            "label": "Blunder",
            "icon": "🔴",
            "delta": delta,
            "description": "A major blunder that loses material or advantage.",
        }


def get_evaluation_summary(board, weights, depth=2, use_fuzzy=True):
    """
    Computes comprehensive evaluation details:
    - white_score: float (+ = White advantage)
    - side_score: float (+ = side-to-move advantage)
    - leader: "White", "Black", or "Equal"
    - leading_text: e.g. "White is leading (+1.25)"
    - eval_percent: 0-100% (for UI eval bar, >50% = White)
    - ascii_bar: e.g. "[████████████░░░░░░░░]"
    - phase: (opening, middlegame, endgame)
    - phase_percents: {"opening": 60, "middlegame": 40, "endgame": 0}
    - features: dict of feature names and values
    """
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            white_score = -MATE_SCORE
            side_score = -MATE_SCORE
            leader = "Black"
            leading_text = "Black wins by checkmate!"
            eval_percent = 0.0
        else:
            white_score = MATE_SCORE
            side_score = -MATE_SCORE
            leader = "White"
            leading_text = "White wins by checkmate!"
            eval_percent = 100.0
    elif (board.is_stalemate() or board.is_insufficient_material() or
          board.is_fifty_moves() or board.is_repetition(3)):
        white_score = 0.0
        side_score = 0.0
        leader = "Equal"
        leading_text = "Drawn position"
        eval_percent = 50.0
    else:
        white_score = evaluate_white(board, weights, use_fuzzy)
        side_score = white_score if board.turn == chess.WHITE else -white_score

        eval_percent = round(50.0 + 50.0 * math.tanh(white_score / 5.0), 1)
        eval_percent = max(1.0, min(99.0, eval_percent))

        if abs(white_score) < 0.20:
            leader = "Equal"
            leading_text = "Equal position (0.00)"
        elif white_score > 0:
            leader = "White"
            leading_text = f"White is leading (+{white_score:.2f})"
        else:
            leader = "Black"
            leading_text = f"Black is leading (+{abs(white_score):.2f})"

    white_blocks = int(round((eval_percent / 100.0) * 20))
    white_blocks = max(0, min(20, white_blocks))
    black_blocks = 20 - white_blocks
    ascii_bar = "[" + "#" * white_blocks + "-" * black_blocks + "]"

    op, mid, end = game_phase(board)
    raw_features = extract_features(board)
    features_dict = {name: round(val, 3) for name, val in zip(FEATURE_NAMES, raw_features)}

    return {
        "white_score": round(white_score, 2),
        "side_score": round(side_score, 2),
        "leader": leader,
        "leading_text": leading_text,
        "eval_percent": eval_percent,
        "ascii_bar": ascii_bar,
        "phase": (op, mid, end),
        "phase_percents": {
            "opening": round(op * 100),
            "middlegame": round(mid * 100),
            "endgame": round(end * 100),
        },
        "features": features_dict,
    }



# ---------------------------------------------------------------------------
# Weight file helpers (train_ga.py writes it; play.py and compare.py read it)
# ---------------------------------------------------------------------------

WEIGHTS_FILE = "best_weights.txt"


def save_weights(weights, path=WEIGHTS_FILE):
    with open(path, "w", encoding="utf-8") as handle:
        for name, value in zip(FEATURE_NAMES, weights):
            handle.write("{:.6f}  # {}\n".format(value, name))


def load_weights(path=WEIGHTS_FILE):
    """Returns the 7 saved weights, or None if the file is missing/unreadable."""
    try:
        values = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#")[0].strip()
                if line:
                    values.append(float(line))
        if len(values) != NUM_FEATURES:
            return None
        return values
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Terminal board drawing (used by play.py; kept here so every module can use it)
# ---------------------------------------------------------------------------

def board_to_text(board, unicode_pieces=False, flipped=False):
    """A text chessboard.  No GUI library anywhere in this project."""
    # SWAPPED UNICODE SYMBOLS:
    # White pieces (uppercase) now map to filled shapes
    # Black pieces (lowercase) now map to hollow shapes
    symbols = {
        "P": "♟", "N": "♞", "B": "♝",
        "R": "♜", "Q": "♛", "K": "♚",
        "p": "♙", "n": "♘", "b": "♗",
        "r": "♖", "q": "♕", "k": "♔",
    }
    ranks = range(8) if flipped else range(7, -1, -1)
    files = range(7, -1, -1) if flipped else range(8)
    
    # 2x bigger board: 48 dashes horizontally
    lines = ["   +" + "-" * 48 + "+"]
    for rank in ranks:
        row1 = "   |"  # Blank padding row
        row2 = " {} |".format(rank + 1)  # Piece row
        for file_index in files:
            piece = board.piece_at(chess.square(file_index, rank))
            row1 += "      "
            if piece is None:
                row2 += "  .   "
            else:
                letter = piece.symbol()
                row2 += "  {}   ".format(symbols[letter] if unicode_pieces else letter)
        row1 += "|"
        row2 += "|"
        lines.append(row1)
        lines.append(row2)
        
    lines.append("   |" + "      " * 8 + "|")
    lines.append("   +" + "-" * 48 + "+")
    header = "       h     g     f     e     d     c     b     a" if flipped else "       a     b     c     d     e     f     g     h"
    lines.append(header)
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-check: prints the fuzzy behaviour quoted in README.md plus a sample
    # evaluation, so this file can be sanity-checked on its own.
    print("Feature names :", FEATURE_NAMES)
    print("Baseline      :", BASELINE_WEIGHTS)
    print()
    print("MODULE 6 verification (expected: 100% opening / 30-70 / 100% endgame)")
    for material, move_number in [(78, 1), (52, 14), (12, 45)]:
        opening, middlegame, endgame = phase_from_inputs(material, move_number)
        print("  material {:>3}  move {:>3}  ->  opening {:5.1f}%  "
              "middlegame {:5.1f}%  endgame {:5.1f}%".format(
                  material, move_number, 100 * opening,
                  100 * middlegame, 100 * endgame))
    print()
    b = chess.Board()
    print(board_to_text(b))
    print("features :", [round(f, 3) for f in extract_features(b)])
    print("eval     :", round(evaluate_white(b, BASELINE_WEIGHTS), 3))
