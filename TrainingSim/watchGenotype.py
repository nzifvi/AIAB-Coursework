import Simulator
import os
import NeuralNetwork
import FitnessFunctions

def watch(generationNo, genotypeID, duration=10.0):
    genPath = f"../Data/Generations/Generation{generationNo}"
    if not os.path.exists(genPath):
        return
    else:
        sim = Simulator.Simulator(simDuration = duration, gui = True)

        try:
            telemetry = sim.runSimulation(
                nn = NeuralNetwork.NeuralNetwork(generationNo=generationNo, genotypeID=genotypeID),
            )
        except Exception as e:
            print(e)
        finally:
            print(
                FitnessFunctions.calculateBalanceFitness(
                    telemetryDF = telemetry,
                    timeStep = sim.timeStep
                )
            )
            sim.disconnect()

if __name__ == "__main__":
    targetGeneration = 210
    for i in range(0, 32):
        watch(targetGeneration, i)