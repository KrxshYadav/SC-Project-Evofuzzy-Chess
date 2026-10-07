"""
train_ga.py -- ARCHITECTURE MODULE 5: the Genetic Algorithm (soft computing #1).

The GA learns the 7 base weights of the evaluation function (Module 4).
Everything is hand-written -- no DEAP, no numpy -- so every line can be
explained in a viva.

    Chromosome  : the 7 evaluation weights, real-valued, range 0..3
    Population  : 12
    Generations : 6
    Selection   : tournament, size 3
    Crossover   : BLX-alpha blend, rate 0.8
    Mutation    : Gaussian, rate 0.15, strength 0.25
    Elitism     : the best 2 individuals are carried over unchanged

Two fitness modes:
    "games"     -- play games against the untuned baseline.  Needs no data.
    "positions" -- count how many grandmaster/puzzle moves the individual
                   reproduces at 1 ply.  Faster.  Needs positions.csv.
                   (This is the fitness function from Paper 2,
                    Vazquez-Fernandez et al. 2013.)

Output:
    best_weights.txt      the tuned weights (used by play.py and compare.py)
    ga_convergence.csv    best + average fitness per generation, for the
                          convergence plot in the report

Run:  python train_ga.py
"""

import concurrent.futures
import csv
import os
import random
import time

import chess

import engine

# ---------------------------------------------------------------------------
# GA parameters -- tune these down if training is too slow on your laptop
# ---------------------------------------------------------------------------

POPULATION_SIZE = 12
GENERATIONS = 6
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.8
BLX_ALPHA = 0.5              # BLX-alpha blend crossover
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.25
ELITE_COUNT = 2

RANDOM_SEED = 42             # fixed seed => the whole run is reproducible

# --- fitness mode ----------------------------------------------------------
# "games"     : play against the baseline (no download needed)
# "positions" : match recorded strong moves (needs positions.csv)
FITNESS_MODE = "games"

# --- "games" mode settings -------------------------------------------------
TRAIN_DEPTH = 2              # low depth during training, for speed
MAX_PLIES = 120              # a game longer than this is adjudicated
ADJUDICATION_MARGIN = 1.5    # pawns of material needed to call an unfinished game

# --- "positions" mode settings ---------------------------------------------
POSITIONS_FILE = "positions.csv"
POSITIONS_PER_EVALUATION = 300   # sampled fresh each generation

CONVERGENCE_FILE = "ga_convergence.csv"

# Individuals in a generation are independent of each other, so they can be
# evaluated on several CPU cores at once.  This is pure speed -- it does not
# change the algorithm or the result.  Set WORKERS = 1 for the plain, easiest
# to follow sequential loop.
WORKERS = max(1, min(8, (os.cpu_count() or 1) - 2))


# ---------------------------------------------------------------------------
# Opening book.  Games start after one of these short lines so that different
# games are genuinely different positions -- the engine itself is deterministic,
# so without this every game would be an identical replay.
# ---------------------------------------------------------------------------

OPENING_LINES = [
    [],                                              # plain start position
    ["e4", "e5"],                                    # open game
    ["d4", "d5"],                                    # closed game
    ["e4", "c5"],                                    # Sicilian
    ["d4", "Nf6", "c4", "e6"],                       # Indian defence
    ["e4", "e6"],                                    # French
    ["c4", "e5"],                                    # English
    ["Nf3", "d5", "g3", "Nf6"],                      # Reti
    ["e4", "c6"],                                    # Caro-Kann
    ["d4", "f5"],                                    # Dutch
]

# How many opening lines each individual is tested on.  Each line is played
# twice (once with the individual as White, once as Black) so colour bias
# cannot creep into the fitness.
OPENINGS_PER_EVALUATION = 2


def opening_board(line):
    """Build a board from a list of SAN moves."""
    board = chess.Board()
    for san in line:
        board.push_san(san)
    return board


# ---------------------------------------------------------------------------
# Playing one training game
# ---------------------------------------------------------------------------

def play_training_game(white_weights, black_weights, opening_line,
                       depth=TRAIN_DEPTH, max_plies=MAX_PLIES):
    """Play one game between two weight vectors.

    Returns 1.0 if White won, 0.0 if Black won, 0.5 for a draw, together with
    the final material balance (positive = White ahead) which is used as a
    fine-grained tie-breaker so that fitness is not flat.
    """
    board = opening_board(opening_line)
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        weights = white_weights if board.turn == chess.WHITE else black_weights
        move = engine.choose_move(board, weights, depth=depth)
        if move is None:                       # no legal move: game is over
            break
        board.push(move)
        plies += 1

    material = engine._material_balance(board)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.5, material
        return (1.0 if outcome.winner == chess.WHITE else 0.0), material

    # Ran out of moves: adjudicate on material, the way a tournament arbiter
    # would call an unfinished game.
    if material >= ADJUDICATION_MARGIN:
        return 1.0, material
    if material <= -ADJUDICATION_MARGIN:
        return 0.0, material
    return 0.5, material


# ---------------------------------------------------------------------------
# FITNESS MODE 1 -- "games": play against the untuned baseline
# ---------------------------------------------------------------------------

def fitness_games(chromosome, opening_lines):
    """Score = (points won against the baseline) + small material bonus.

    Maximum is 1.0 per game.  The material term is deliberately small: it only
    breaks ties between individuals that scored the same number of points,
    which is what stops the fitness curve from going flat.
    """
    total = 0.0
    games = 0
    for line in opening_lines:
        # Individual plays White
        result, material = play_training_game(chromosome,
                                              engine.BASELINE_WEIGHTS, line)
        total += result + 0.02 * max(-5.0, min(5.0, material))
        games += 1

        # Same opening, individual plays Black
        result, material = play_training_game(engine.BASELINE_WEIGHTS,
                                              chromosome, line)
        total += (1.0 - result) + 0.02 * max(-5.0, min(5.0, -material))
        games += 1

    return total / games


# ---------------------------------------------------------------------------
# FITNESS MODE 2 -- "positions": reproduce recorded strong moves at 1 ply
# ---------------------------------------------------------------------------

def load_positions(path=POSITIONS_FILE):
    """Read positions.csv.  Expected header: fen,best_move (best_move in UCI).

    Build it with:  python make_positions.py
    """
    if not os.path.exists(path):
        raise SystemExit(
            "{} not found.\n"
            "Either run  python make_positions.py  to build it from the "
            "Lichess puzzle database,\n"
            "or set FITNESS_MODE = \"games\" at the top of train_ga.py."
            .format(path))

    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("fen") and row.get("best_move"):
                rows.append((row["fen"], row["best_move"]))
    if not rows:
        raise SystemExit("{} contains no usable rows.".format(path))
    return rows


def fitness_positions(chromosome, sample):
    """Fraction of positions where a 1-ply search picks the recorded move.

    1 ply means the evaluation function alone decides -- which is exactly what
    we are trying to tune.
    """
    matches = 0
    for fen, best_move_uci in sample:
        board = chess.Board(fen)
        move = engine.choose_move(board, chromosome, depth=1,
                                  use_quiescence=False)
        if move is not None and move.uci() == best_move_uci:
            matches += 1
    return matches / len(sample)


def evaluate_chromosome(job):
    """One fitness evaluation.  Written as a plain module-level function taking
    a single tuple so it can also be handed to a worker process."""
    chromosome, workload, fitness_mode = job
    if fitness_mode == "games":
        return fitness_games(chromosome, workload)
    return fitness_positions(chromosome, workload)


def evaluate_population_parallel(population, workload, fitness_mode):
    """Evaluate the whole generation across WORKERS CPU cores.  Results come
    back in population order, so the GA behaves exactly as it would in the
    sequential loop."""
    jobs = [(chromosome, workload, fitness_mode) for chromosome in population]
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(evaluate_chromosome, jobs))


# ---------------------------------------------------------------------------
# The genetic operators
# ---------------------------------------------------------------------------

def random_chromosome(rng):
    """A random point in the 7-dimensional weight space."""
    return [rng.uniform(engine.WEIGHT_MIN, engine.WEIGHT_MAX)
            for _ in range(engine.NUM_FEATURES)]


def clamp(value):
    return max(engine.WEIGHT_MIN, min(engine.WEIGHT_MAX, value))


def tournament_select(population, fitnesses, rng, size=TOURNAMENT_SIZE):
    """Pick `size` individuals at random, return a copy of the best of them."""
    contenders = rng.sample(range(len(population)), size)
    winner = max(contenders, key=lambda i: fitnesses[i])
    return list(population[winner])


def blx_crossover(parent_a, parent_b, rng, alpha=BLX_ALPHA):
    """BLX-alpha blend crossover.

    For each gene, sample uniformly from an interval that spans both parents
    and extends alpha*range beyond each of them, so children can explore
    slightly outside the parents' box.
    """
    child_a, child_b = [], []
    for gene_a, gene_b in zip(parent_a, parent_b):
        low, high = min(gene_a, gene_b), max(gene_a, gene_b)
        spread = high - low
        lower = low - alpha * spread
        upper = high + alpha * spread
        child_a.append(clamp(rng.uniform(lower, upper)))
        child_b.append(clamp(rng.uniform(lower, upper)))
    return child_a, child_b


def gaussian_mutate(chromosome, rng, rate=MUTATION_RATE,
                    strength=MUTATION_STRENGTH):
    """Each gene has `rate` probability of being nudged by Gaussian noise."""
    return [clamp(gene + rng.gauss(0.0, strength)) if rng.random() < rate
            else gene
            for gene in chromosome]


# ---------------------------------------------------------------------------
# The main GA loop
# ---------------------------------------------------------------------------

def format_weights(weights):
    return "[" + ", ".join("{:.3f}".format(w) for w in weights) + "]"


def run_ga(fitness_mode=FITNESS_MODE, population_size=POPULATION_SIZE,
           generations=GENERATIONS, seed=RANDOM_SEED, verbose=True):
    rng = random.Random(seed)

    positions = load_positions() if fitness_mode == "positions" else None

    # Start from random individuals, but seed one copy of the untuned baseline
    # into the population: the GA should never end up worse than its own
    # starting point.
    population = [random_chromosome(rng) for _ in range(population_size - 1)]
    population.append(list(engine.BASELINE_WEIGHTS))

    history = []
    best_ever = list(engine.BASELINE_WEIGHTS)
    best_ever_fitness = -1.0
    started = time.time()

    for generation in range(1, generations + 1):
        # Build the evaluation workload for this generation.  Every individual
        # in a generation faces exactly the same test, so the comparison inside
        # the generation is fair.
        if fitness_mode == "games":
            workload = rng.sample(
                OPENING_LINES, min(OPENINGS_PER_EVALUATION, len(OPENING_LINES)))
        else:
            workload = rng.sample(
                positions, min(POSITIONS_PER_EVALUATION, len(positions)))
        evaluate_one = lambda c: evaluate_chromosome((c, workload, fitness_mode))

        generation_started = time.time()
        if WORKERS > 1:
            fitnesses = evaluate_population_parallel(population, workload,
                                                     fitness_mode)
        else:
            fitnesses = [evaluate_one(chromosome) for chromosome in population]
        if verbose:
            for index, value in enumerate(fitnesses):
                print("  gen {:>2}  individual {:>2}/{:<2}  fitness {:.4f}"
                      .format(generation, index + 1, population_size, value),
                      flush=True)

        best_index = max(range(population_size), key=lambda i: fitnesses[i])
        best_fitness = fitnesses[best_index]
        average_fitness = sum(fitnesses) / population_size

        if best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever = list(population[best_index])

        history.append({
            "generation": generation,
            "best_fitness": best_fitness,
            "average_fitness": average_fitness,
            "best_weights": list(population[best_index]),
            "seconds": time.time() - generation_started,
        })

        if verbose:
            print("GENERATION {:>2}/{:<2}  best {:.4f}  average {:.4f}  "
                  "({:.1f}s)  {}".format(
                      generation, generations, best_fitness, average_fitness,
                      history[-1]["seconds"],
                      format_weights(population[best_index])), flush=True)

        if generation == generations:
            break

        # --- build the next generation ------------------------------------
        ranked = sorted(range(population_size), key=lambda i: fitnesses[i],
                        reverse=True)
        next_population = [list(population[i]) for i in ranked[:ELITE_COUNT]]

        while len(next_population) < population_size:
            parent_a = tournament_select(population, fitnesses, rng)
            parent_b = tournament_select(population, fitnesses, rng)
            if rng.random() < CROSSOVER_RATE:
                child_a, child_b = blx_crossover(parent_a, parent_b, rng)
            else:
                child_a, child_b = parent_a, parent_b
            next_population.append(gaussian_mutate(child_a, rng))
            if len(next_population) < population_size:
                next_population.append(gaussian_mutate(child_b, rng))

        population = next_population

    elapsed = time.time() - started
    return best_ever, best_ever_fitness, history, elapsed


def save_convergence(history, path=CONVERGENCE_FILE):
    """Save the convergence data so the report can plot it."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "best_fitness", "average_fitness",
                         "seconds"] + engine.FEATURE_NAMES)
        for row in history:
            writer.writerow([row["generation"],
                             "{:.6f}".format(row["best_fitness"]),
                             "{:.6f}".format(row["average_fitness"]),
                             "{:.1f}".format(row["seconds"])]
                            + ["{:.6f}".format(w) for w in row["best_weights"]])


def main():
    print("=" * 70)
    print("MODULE 5 -- GENETIC ALGORITHM: tuning the evaluation weights")
    print("=" * 70)
    print("fitness mode  : {}".format(FITNESS_MODE))
    print("population    : {}".format(POPULATION_SIZE))
    print("generations   : {}".format(GENERATIONS))
    print("selection     : tournament (size {})".format(TOURNAMENT_SIZE))
    print("crossover     : BLX-{} at rate {}".format(BLX_ALPHA, CROSSOVER_RATE))
    print("mutation      : Gaussian rate {} strength {}".format(
        MUTATION_RATE, MUTATION_STRENGTH))
    print("elitism       : {}".format(ELITE_COUNT))
    if FITNESS_MODE == "games":
        print("search depth  : {} (low, for training speed)".format(TRAIN_DEPTH))
        print("games per ind.: {} ({} openings x 2 colours vs the baseline)"
              .format(2 * OPENINGS_PER_EVALUATION, OPENINGS_PER_EVALUATION))
    print("cpu workers   : {}".format(WORKERS))
    print("baseline      : {}".format(format_weights(engine.BASELINE_WEIGHTS)))
    print("-" * 70, flush=True)

    best, best_fitness, history, elapsed = run_ga()

    print("-" * 70)
    print("TRAINING FINISHED in {:.1f}s ({:.1f} min)".format(elapsed, elapsed / 60))
    print("best fitness  : {:.4f}".format(best_fitness))
    print("tuned weights :")
    for name, value in zip(engine.FEATURE_NAMES, best):
        print("    {:<16} {:.4f}".format(name, value))

    engine.save_weights(best)
    save_convergence(history)
    print()
    print("wrote {}".format(engine.WEIGHTS_FILE))
    print("wrote {} (best + average fitness per generation, for the plot)"
          .format(CONVERGENCE_FILE))
    print()
    print("Convergence:")
    print("  gen   best      average")
    for row in history:
        print("  {:>3}   {:.4f}    {:.4f}".format(
            row["generation"], row["best_fitness"], row["average_fitness"]))
    print()
    print("Next step:  python compare.py   (tuned vs untuned -- the headline result)")


if __name__ == "__main__":
    main()
