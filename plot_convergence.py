import csv
import matplotlib.pyplot as plt

def plot_convergence():
    generations = []
    best_fitness = []
    avg_fitness = []

    try:
        with open("ga_convergence.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                generations.append(int(row["generation"]))
                best_fitness.append(float(row["best_fitness"]))
                avg_fitness.append(float(row["average_fitness"]))
    except FileNotFoundError:
        print("Error: ga_convergence.csv not found. Run train_ga.py first.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label="Best Fitness", marker='o', linewidth=2)
    plt.plot(generations, avg_fitness, label="Average Fitness", marker='s', linewidth=2, linestyle='--')
    
    plt.title("Genetic Algorithm Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("ga_convergence.png")
    print("Plot saved to ga_convergence.png")
    # plt.show() # Uncomment if running interactively to view it

if __name__ == "__main__":
    plot_convergence()
