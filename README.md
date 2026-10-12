# README.md — Soft Computing Chess Project

## What this project is

A college **Soft Computing** course project. We are building a chess-playing
engine whose evaluation function is tuned by a **Genetic Algorithm** and
adjusted by a **Fuzzy Logic** system. A human can play against it.

The point of the project is NOT to beat Stockfish. The point is to
demonstrate and measure two soft-computing techniques working on a real
problem. The headline result is: **GA-tuned engine vs untuned baseline —
how much did tuning improve it?**

Deliverables: working code + a report + a lab demo.

---

## Hard constraints — do not violate these

- **Python only.** Runs on a normal student laptop, CPU only, no GPU.
- **Terminal interface only.** No Pygame, no web app, no GUI library.
  A text chessboard printed to the terminal is the interface.
  (Reason: fewer dependencies = nothing breaks during the lab demo.)
- **One external library: `chess` (python-chess).** Do not add numpy,
  DEAP, scikit-fuzzy, torch, or anything else. The GA and the fuzzy
  system are hand-written so they can be explained line-by-line in a viva.
- **No neural networks.** The soft-computing content is GA + Fuzzy Logic.
  AlphaZero is cited as background only.
- **Keep code readable over clever.** This is coursework — a teacher will
  read it. Comments should say WHICH ARCHITECTURE MODULE each block is.
- **Must finish in one semester.** Prefer the simple working version over
  the impressive fragile one.

---

## Architecture — 6 modules

```
HUMAN (terminal)
      |
      v
1. BOARD INPUT ......... read position, extract numeric features
      |
      v
2. MOVE GENERATOR ...... legal moves via python-chess
      |
      v
3. SEARCH .............. minimax + alpha-beta pruning, depth 3-4
      |
      v
4. EVALUATION FUNCTION . score = w1*f1 + w2*f2 + ... + wn*fn
      ^                                    ^
      |                                    |
5. GENETIC ALGORITHM              6. FUZZY LOGIC MODULE
   learns the base weights           adjusts them by game phase
      |
      v
OUTPUT: best move
```

**Module 1 — Board Input.** Converts a position into a list of 7 numbers:
material, mobility, king_safety, king_activity, doubled_pawns,
passed_pawns, centre_control. Positive = White is better.

**Module 2 — Move Generator.** `python-chess` handles legality, check,
mate, stalemate, castling, en passant, promotion. Do not reimplement this.

**Module 3 — Search.** Negamax-style minimax with alpha-beta pruning.
Captures are tried first so pruning works better. Depth 3 for play,
depth 1–2 during GA training for speed.

**Module 4 — Evaluation Function.** Weighted sum of the 7 features.
The weights are what everything else exists to produce.

**Module 5 — Genetic Algorithm (soft computing #1).**
- Chromosome = the 7 evaluation weights, real-valued, range 0–3
- Population 12, generations 6 (tunable down if slow)
- Selection: tournament, size 3
- Crossover: BLX blend, rate 0.8
- Mutation: Gaussian, rate 0.15, strength 0.25
- Elitism: 2 individuals carried over unchanged
- Two fitness modes:
  - `"games"` — play games vs the baseline. Works with no data download.
  - `"positions"` — count how many grandmaster moves it reproduces at
    1-ply. Faster. Needs `positions.csv`. This is the method from Paper 2.

**Module 6 — Fuzzy Logic (soft computing #2).**
- Inputs: total material left on board (0–78), current move number
- Fuzzification: material → LOW / MEDIUM / HIGH (triangular and shoulder
  membership functions); move number → EARLY / MIDDLE / LATE
- Rules (Mamdani, AND = min):
  - IF material HIGH AND moves EARLY THEN OPENING
  - IF material MEDIUM AND moves MIDDLE THEN MIDDLEGAME
  - IF material LOW OR moves LATE THEN ENDGAME
- Output: three membership degrees that sum to 1.0
- Effect: blends between phase-specific weight multipliers. Example —
  king_safety multiplier is 1.4 in the middlegame but 0.2 in the endgame;
  king_activity is 0.0 in the opening but 1.5 in the endgame.

**Why fuzzy and not an if-statement:** a hard rule like "if fewer than 10
pieces then endgame" jumps suddenly. Fuzzy lets a position be 30%
middlegame and 70% endgame at once, so the weights shift smoothly.
Verified behaviour: move 1 full material → 100% opening; move 14 with 52
material → 30/70 opening/middle; move 45 with 12 material → 100% endgame.

---

## Files

| File | Contains | Status |
|---|---|---|
| `app.py` | Chess.com-style Web UI (Flask, local, no DB/login) | working, browser UI |
| `engine.py` | Modules 1, 2, 3, 4, 6 + move evaluation | working, run end-to-end |
| `train_ga.py` | Module 5 (the GA) | working, full run completed |
| `play.py` | Terminal interface (learning & practice modes) | working, played through |
| `compare.py` | Tuned vs untuned experiment | working, 50-game run completed |
| `make_positions.py` | Lichess puzzles -> `positions.csv` | working, 20k positions built |
| `bench_stockfish.py` | Optional external benchmark | working (needs the Stockfish exe) |
| `best_weights.txt` | GA output, 7 numbers | generated by train_ga.py |
| `ga_convergence.csv` | best + avg fitness per generation | generated by train_ga.py |
| `results.txt` / `results.csv` | the experiment result | generated by compare.py |
| `positions.csv` | 20,000 FEN + best-move rows | built from the Lichess puzzle DB |

**Two notes on the implementation.**

*Feature scaling.* Each of the 7 features is pre-scaled inside Module 1
(mobility /10, king safety /4, king activity /3, centre control /4) so all
seven live on a comparable numeric range. That is what makes one shared GA
weight range of 0-3 meaningful across all of them.

*Parallel evaluation.* `train_ga.py` and `compare.py` evaluate independent
individuals / games on several CPU cores using `concurrent.futures` from the
standard library. No new dependency, and it does not change the algorithm --
the engine is deterministic, so the results are identical either way. Set
`WORKERS = 1` in either file for the plain sequential loop.

---

## Commands

```
pip install chess flask    # dependencies (chess + local flask UI)

python engine.py           # self-check: fuzzy phases + a sample evaluation
python train_ga.py         # train -- writes best_weights.txt + ga_convergence.csv
python play.py             # play in terminal (learning & practice modes)
python app.py              # play in Chess.com-style local web UI (auto-opens browser)
python compare.py          # the experiment for the report

# optional extras
python make_positions.py lichess_db_puzzle.csv --limit 20000
python bench_stockfish.py --skill 0 --games 10
```

---

## Training data

Only needed for fitness mode `"positions"`. Mode `"games"` needs nothing.

- Source: **Lichess Open Database** — https://database.lichess.org/
- Best option: the **puzzle CSV** at https://database.lichess.org/#puzzles
  Each row already has a FEN and the correct move. No PGN parsing needed.
- **Do not download a full monthly game file — they are ~20 GB compressed.**
- 10,000–50,000 positions is plenty for a course project.

Format pipeline: `PGN` (text) → `FEN` (text, one position) →
`[3, 12, -1, 0, 5, 2, 0.5]` (list of numbers) → evaluation function.

**The data is fed in as plain numbers, not binary.** Binary board encoding
(8×8×12 planes) is only for neural-network approaches, which we are not
using. Bitboards are binary but live inside python-chess — we never touch
them.

The model is not memorising moves. It is learning the **weights** so that
its own move choices come to match those of strong players.

---

## The 5 SCI papers (for the report)

1. **David, van den Herik, Koppel & Netanyahu (2014)** — "Genetic
   Algorithms for Evolving Computer Chess Programs", *IEEE Trans. on
   Evolutionary Computation* 18(5):779–789. DOI 10.1109/TEVC.2013.2285111.
   Free PDF: https://arxiv.org/pdf/1711.08337
   → Our main reference. GA learns evaluation weights. Anchor for Module 5.

2. **Vázquez-Fernández, Coello Coello & Sagols Troncoso (2013)** — "An
   evolutionary algorithm with a history mechanism for tuning a chess
   evaluation function", *Applied Soft Computing* 13(7):3234–3247.
   DOI 10.1016/j.asoc.2013.02.015. (Paywalled.)
   → Source of the grandmaster-move-matching fitness function.

3. **Bošković, Brest, Zamuda, Greiner & Žumer (2011)** — "History mechanism
   supported differential evolution for chess evaluation function tuning",
   *Soft Computing* 15(4):667–683. DOI 10.1007/s00500-010-0593-z. (Paywalled.)
   → Reports ~155 Elo gain from tuning. Our benchmark target.

4. **Fogel, Hays, Hahn & Quon (2004)** — "A self-learning evolutionary
   chess program", *Proceedings of the IEEE* 92(12):1947–1954.
   DOI 10.1109/JPROC.2004.837633. (Paywalled.)
   → Blondie25. Classic evolutionary chess proof-of-concept.

5. **Silver et al. (2018)** — "A general reinforcement learning algorithm
   that masters chess, shogi, and Go through self-play", *Science*
   362(6419):1140–1144. DOI 10.1126/science.aar6404.
   Free preprint: https://arxiv.org/pdf/1712.01815
   → AlphaZero. Cited as state of the art / background only.

NOT SCI journals — supporting literature only, never cite as SCI:
DeepChess (ICANN conference), Maia (KDD conference),
Grandmaster-Level Chess Without Search (arXiv/NeurIPS).

---

## What's left to do

1. Install `chess`, run all four scripts, fix any runtime errors.
2. Confirm the GA fitness actually improves across generations — if the
   average fitness is flat, raise `GENERATIONS` or check the fitness function.
3. Run `compare.py` with `NUM_GAMES = 50` for a report-worthy result.
4. Save GA convergence data (best + average fitness per generation) so it
   can be plotted for the report.
5. Optional: build `positions.csv` from Lichess and switch fitness to
   `"positions"` mode.
6. Optional: add quiescence search (only search captures at the leaf) to
   stop the engine hanging pieces at the search boundary.
7. Optional: benchmark against Stockfish at a capped skill level.

---

## Known pitfalls

- **Horizon effect** — without quiescence search the engine hangs pieces
  right at the depth limit. Add it if play looks bad.
- **GA fitness too slow** — the fix is lower search depth during training
  (already at 2) or switch to `"positions"` fitness, not a bigger population.
- **Flat fitness across generations** — usually means the fitness function
  is too noisy. More games per individual, or switch fitness mode.
- **Tuned loses to untuned in compare.py** — train for more generations.
  The baseline weights are decent, so the GA needs enough time to beat them.
- **Do not delete the untuned baseline.** It is the control group; without
  it there is no result to report.
- Always alternate colours in comparison matches or the result is biased.

---

## Report structure (for reference)

1. Introduction — problem, why soft computing
2. Literature review — the 5 papers above
3. Proposed architecture — the 6 modules
4. Implementation — GA parameters, fuzzy rules, features
5. Data — Lichess, formats, pipeline
6. Results — tuned vs untuned table, GA convergence plot
7. Conclusion — what worked, what we would do next
