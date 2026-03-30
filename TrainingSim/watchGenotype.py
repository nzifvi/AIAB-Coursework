import Simulator
import os
import NeuralNetwork

def watch(generationNo, genotypeID, duration=100.0):
    genPath = f"C:/Users/benja/Desktop/AIAB Coursework/Data/Generations/Generation{generationNo}"
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
            sim.disconnect()

if __name__ == "__main__":
    targetGeneration = 75
    for i in range(0, 32):
        watch(targetGeneration, i)