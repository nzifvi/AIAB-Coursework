import matplotlib

matplotlib.use('TkAgg')  # Ensures a standalone interactive window
import matplotlib.pyplot as plt
from GenerationController import GenerationController

if __name__ == "__main__":
    runForNGenerations = 100
    popSize = 32

    generationController = GenerationController(
        popSize,
        checkpointControl=5
    )

    # 1. Data Storage
    averageFitnessOverTime = []
    bestFitnessOverTime = []
    worstFitnessOverTime = []
    startGen = generationController.generationNo

    # 2. Plot Setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # We keep the average line as a persistent object
    line_avg, = ax.plot([], [], 'b-', linewidth=2, label='Average Fitness')

    # This will hold our error bars so we can clear them each frame
    error_bars = None

    ax.set_title("Evolutionary Progress: Population Fitness Variance")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness Score")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

    # 3. Main Training Loop
    for gen in range(runForNGenerations):
        # Run simulation and unpack the 3 values
        avgFit, bestFit, worstFit = generationController.run(
            genotypeReverificationNo = 4
        )

        averageFitnessOverTime.append(avgFit)
        bestFitnessOverTime.append(bestFit)
        worstFitnessOverTime.append(worstFit)

        # Calculate the X-axis (handling potential checkpoint offsets)
        current_x = list(range(startGen, startGen + len(averageFitnessOverTime)))

        # Update the central average line
        line_avg.set_data(current_x, averageFitnessOverTime)

        # 4. Handle Error Bars (The Variance)
        # We calculate the distance from the average to the best/worst
        lower_err = [a - w for a, w in zip(averageFitnessOverTime, worstFitnessOverTime)]
        upper_err = [b - a for a, b in zip(averageFitnessOverTime, bestFitnessOverTime)]
        yerr = [lower_err, upper_err]

        # Remove previous error bars to prevent stacking/ghosting
        if error_bars:
            error_bars.remove()

        # Draw new error bars
        # fmt='none' ensures we don't draw extra markers, just the bars
        error_bars = ax.errorbar(
            current_x,
            averageFitnessOverTime,
            yerr=yerr,
            fmt='none',
            ecolor='skyblue',
            elinewidth=1,
            capsize=3,
            alpha=0.7
        )

        # 5. Refresh the GUI
        ax.relim()
        ax.autoscale_view()

        # Draw and pause to allow the window to process the update
        plt.draw()
        plt.pause(0.1)

        print(f"Generation {startGen + len(averageFitnessOverTime) - 1} | "
              f"Best: {bestFit:.4f} | Avg: {avgFit:.4f}")

    # Keep the window open after the loop finishes
    plt.ioff()
    plt.show()