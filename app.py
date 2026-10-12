"""
app.py -- Soft Computing Chess Web UI.
A standalone local web interface styled like Chess.com:
- Interactive chessboard with drag-and-drop & click-to-move
- Smooth vertical evaluation bar & leading player banner
- Soft Computing Architecture: GA-Tuned weights with dynamic Mamdani Fuzzy phase modulation
- Replay / History navigation with < and > controls
- Futuristic AI reasoning & thinking dashboard for presentation
- No external database or login required; runs 100% locally.
"""

import math
import os
import sys
import threading
import time
import webbrowser
import chess
from flask import Flask, jsonify, render_template, request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import engine

app = Flask(__name__)


class ChessGame:
    def __init__(self):
        self.reset()

    def reset(self, mode="normal", human_color=chess.WHITE, depth=3, weights_type="tuned"):
        self.board = chess.Board()
        self.mode = mode
        self.human_color = human_color
        self.depth = depth
        self.weights_type = weights_type
        self.weights = self._load_weights(weights_type)
        self.history = []
        self.last_move_analysis = None
        self.last_engine_stats = None
        self.snapshots = [self._snapshot_current(last_move=None)]

    def _load_weights(self, weights_type):
        if weights_type == "tuned":
            w = engine.load_weights()
            if w is not None:
                return w
        return list(engine.BASELINE_WEIGHTS)

    def _snapshot_current(self, last_move=None):
        phase = engine.game_phase(self.board)
        adjusted_weights = engine.fuzzy_adjust_weights(self.weights, phase)
        eval_summary = engine.get_evaluation_summary(
            self.board, self.weights, depth=min(2, self.depth)
        )
        live_weights = [
            {
                "name": name,
                "base": round(base_w, 3),
                "adjusted": round(adj_w, 3),
                "multiplier": round(adj_w / base_w, 2) if base_w > 0.0001 else 1.0,
            }
            for name, base_w, adj_w in zip(engine.FEATURE_NAMES, self.weights, adjusted_weights)
        ]
        return {
            "ply": len(self.board.move_stack),
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "last_move": last_move,
            "eval": eval_summary,
            "live_weights": live_weights,
            "in_check": self.board.is_check(),
            "is_game_over": self.board.is_game_over(claim_draw=True),
        }

    def get_dict(self):
        is_over = self.board.is_game_over(claim_draw=True)
        game_over_result = None
        winner = None
        if is_over:
            outcome = self.board.outcome(claim_draw=True)
            if outcome is not None:
                if outcome.winner is None:
                    winner = "draw"
                    game_over_result = f"Draw by {outcome.termination.name.replace('_', ' ').title()}"
                elif outcome.winner == chess.WHITE:
                    winner = "white"
                    game_over_result = f"White Wins by {outcome.termination.name.replace('_', ' ').title()}!"
                else:
                    winner = "black"
                    game_over_result = f"Black Wins by {outcome.termination.name.replace('_', ' ').title()}!"

        in_check = self.board.is_check()
        king_sq = chess.square_name(self.board.king(self.board.turn)) if in_check else None
        ep_sq = chess.square_name(self.board.ep_square) if self.board.ep_square is not None else None

        eval_summary = engine.get_evaluation_summary(
            self.board, self.weights, depth=min(2, self.depth)
        )

        phase = engine.game_phase(self.board)
        adjusted_weights = engine.fuzzy_adjust_weights(self.weights, phase)
        live_weights = [
            {
                "name": name,
                "base": round(base_w, 3),
                "adjusted": round(adj_w, 3),
                "multiplier": round(adj_w / base_w, 2) if base_w > 0.0001 else 1.0,
            }
            for name, base_w, adj_w in zip(engine.FEATURE_NAMES, self.weights, adjusted_weights)
        ]

        top_moves = []
        if self.mode == "learning" and not is_over and self.board.turn == self.human_color:
            raw_top = engine.get_top_moves(
                self.board, self.weights, n=3, depth=min(2, self.depth)
            )
            for m in raw_top:
                uci_str = m["uci"]
                from_sq = uci_str[:2]
                to_sq = uci_str[2:4]
                piece_obj = self.board.piece_at(chess.parse_square(from_sq))
                piece_sym = piece_obj.symbol() if piece_obj else ""
                top_moves.append({
                    "uci": uci_str,
                    "san": m["san"],
                    "from": from_sq,
                    "to": to_sq,
                    "piece": piece_sym,
                    "score": round(m["score"], 2),
                    "white_score": round(m["white_score"], 2),
                })

        legal_moves = [
            {
                "from": chess.square_name(m.from_square),
                "to": chess.square_name(m.to_square),
                "uci": m.uci(),
                "san": self.board.san(m),
            }
            for m in self.board.legal_moves
        ]

        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "human_color": "white" if self.human_color == chess.WHITE else "black",
            "mode": self.mode,
            "depth": self.depth,
            "weights_type": self.weights_type,
            "is_game_over": is_over,
            "winner": winner,
            "game_over_result": game_over_result,
            "in_check": in_check,
            "king_square": king_sq,
            "en_passant": ep_sq,
            "eval": eval_summary,
            "live_weights": live_weights,
            "top_moves": top_moves,
            "legal_moves": legal_moves,
            "history": self.history,
            "last_move_analysis": self.last_move_analysis,
            "last_engine_stats": self.last_engine_stats,
            "snapshots": self.snapshots,
        }


game = ChessGame()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def get_state():
    return jsonify(game.get_dict())


@app.route("/api/move", methods=["POST"])
def make_move():
    data = request.json or {}
    from_str = data.get("from")
    to_str = data.get("to")
    promo_str = data.get("promotion", "q")

    if not from_str or not to_str:
        return jsonify({"success": False, "error": "Missing move squares"}), 400

    from_sq = chess.parse_square(from_str)
    to_sq = chess.parse_square(to_str)

    # Detect promotion
    piece = game.board.piece_at(from_sq)
    promo_piece = None
    if piece and piece.piece_type == chess.PAWN:
        if (piece.color == chess.WHITE and chess.square_rank(to_sq) == 7) or \
           (piece.color == chess.BLACK and chess.square_rank(to_sq) == 0):
            promo_piece = chess.Piece.from_symbol(promo_str).piece_type if promo_str else chess.QUEEN

    move = chess.Move(from_sq, to_sq, promotion=promo_piece)

    if move not in game.board.legal_moves:
        return jsonify({"success": False, "error": "Illegal move"}), 400

    # In Learning Mode, analyze the move quality before applying it
    if game.mode == "learning":
        all_moves = engine.evaluate_all_moves(
            game.board, game.weights, depth=min(2, game.depth)
        )
        best_move_dict = all_moves[0] if all_moves else None
        player_move_dict = next((m for m in all_moves if m["move"] == move), None)

        if best_move_dict and player_move_dict:
            res = engine.classify_move(best_move_dict["score"], player_move_dict["score"])
            san_before = game.board.san(move)
            game.last_move_analysis = {
                "badge": res["badge"],
                "label": res["label"],
                "icon": res["icon"],
                "delta": res["delta"],
                "description": res["description"],
                "player_move": san_before,
                "best_move": best_move_dict["san"],
                "player_score": round(player_move_dict["score"], 2),
                "best_score": round(best_move_dict["score"], 2),
            }
        else:
            game.last_move_analysis = None
    else:
        game.last_move_analysis = None

    san = game.board.san(move)
    turn_color = "white" if game.board.turn == chess.WHITE else "black"
    game.board.push(move)
    move_entry = {"san": san, "uci": move.uci(), "turn": turn_color}
    game.history.append(move_entry)
    game.snapshots.append(game._snapshot_current(last_move=move_entry))

    return jsonify({"success": True, "state": game.get_dict()})


@app.route("/api/engine_move", methods=["POST"])
def engine_move():
    if game.board.is_game_over(claim_draw=True):
        return jsonify({"success": False, "error": "Game is over", "state": game.get_dict()})

    if game.board.turn == game.human_color:
        return jsonify({"success": False, "error": "Not engine's turn", "state": game.get_dict()})

    started = time.time()
    move, score, stats = engine.search(game.board, game.weights, depth=game.depth)
    elapsed = time.time() - started

    if move is None:
        return jsonify({"success": False, "error": "No legal moves", "state": game.get_dict()})

    san = game.board.san(move)
    turn_color = "white" if game.board.turn == chess.WHITE else "black"
    game.board.push(move)
    move_entry = {"san": san, "uci": move.uci(), "turn": turn_color}
    game.history.append(move_entry)
    game.snapshots.append(game._snapshot_current(last_move=move_entry))

    engine_stats = {
        "move": san,
        "score": round(score, 2),
        "elapsed": round(elapsed, 2),
        "nodes": stats.nodes,
        "quiescence_nodes": stats.quiescence_nodes,
        "depth": game.depth,
    }
    game.last_engine_stats = engine_stats

    return jsonify({
        "success": True,
        "move": san,
        "score": round(score, 2),
        "elapsed": round(elapsed, 2),
        "nodes": stats.nodes,
        "quiescence_nodes": stats.quiescence_nodes,
        "state": game.get_dict()
    })


@app.route("/api/new_game", methods=["POST"])
def new_game():
    data = request.json or {}
    mode = data.get("mode", "normal")
    color_str = data.get("color", "white")
    human_color = chess.WHITE if color_str == "white" else chess.BLACK
    depth = int(data.get("depth", 3))
    weights_type = data.get("weights", "tuned")

    game.reset(mode=mode, human_color=human_color, depth=depth, weights_type=weights_type)
    return jsonify({"success": True, "state": game.get_dict()})


@app.route("/api/revert_to_ply", methods=["POST"])
def revert_to_ply():
    data = request.json or {}
    ply = data.get("ply")
    if ply is None or ply < 0 or ply > len(game.board.move_stack):
        return jsonify({"success": False, "error": "Invalid ply index"}), 400

    while len(game.board.move_stack) > ply:
        game.board.pop()
    game.history = game.history[:ply]
    game.snapshots = game.snapshots[:ply + 1]
    game.last_move_analysis = None
    return jsonify({"success": True, "state": game.get_dict()})


@app.route("/api/undo", methods=["POST"])
def undo():
    # If it's human's turn, engine already replied, so undo both engine and human move
    if len(game.board.move_stack) >= 2:
        game.board.pop()
        game.board.pop()
        game.history = game.history[:-2]
        game.snapshots = game.snapshots[:-2]
    elif len(game.board.move_stack) == 1:
        game.board.pop()
        game.history = game.history[:-1]
        game.snapshots = game.snapshots[:-1]

    game.last_move_analysis = None
    return jsonify({"success": True, "state": game.get_dict()})


@app.route("/api/set_mode", methods=["POST"])
def set_mode():
    data = request.json or {}
    mode = data.get("mode", "normal")
    game.mode = mode
    return jsonify({"success": True, "state": game.get_dict()})


def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    print("=" * 70)
    print("  SOFT COMPUTING CHESS - WEB UI")
    print("  GA-Tuned & Fuzzy Logic Evaluation Engine")
    print("  Starting local server on: http://127.0.0.1:5000")
    print("  (No login or database required. Press Ctrl+C in terminal to stop)")
    print("=" * 70)

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
