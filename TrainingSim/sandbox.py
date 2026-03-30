import pybullet
import time
import numpy as np
import torch
from Simulator import Simulator
from RobotController import RobotController
from NeuralNetwork import NeuralNetwork
import multiprocessing
import FitnessFunctions
import matplotlib.pyplot as plt

def runPDAutotune():
    sim = Simulator(
        simDuration = 10.0,
        gui         = True
    )

    mockNN = NeuralNetwork(
        generationNo = 0,
        genotypeID   = 0
    )

    kpRange = [50, 100, 150, 200]
    kdRange = [5, 10, 15, 20]

    bestParams = (0, 0)
    bestScore = float('inf')

    for candidateKP in kpRange:
        for candidateKD in kdRange:
            sim.reset()

            robot = RobotController(
                nn = mockNN,
                basePosition = [0, 0, 0.40],
                maxTorque = 15
            )

            jitter, height = robot.tunePDController(
                kp      = candidateKP,
                kd      = candidateKD,
                showGUI = True,
                duration = 2000
            )

            score = jitter if height > 0.3 else float('inf')

            if score < bestScore:
                bestScore = score
                bestParams = (candidateKP, candidateKD)

            print(f"(kp:{candidateKP}, kd:{candidateKD}) | jitter : {jitter:.4f} | height : {height:.3f}")
    print("\n")
    print(f"optimal kp: {bestParams[0]} | optimal kd: {bestParams[1]}")
    sim.disconnect()

def plotGenerationalFitness():
    generationCount = 215
    popCount = 32

    avgFitness = []

    sim = Simulator(simDuration=10.0, gui=False)
    i = 0
    while i <= generationCount:
        if i >= 210:
            popCount = 64
        sumFitness = 0.0
        for j in range(0, popCount):
            telemetry = sim.runSimulation(
                NeuralNetwork(
                    generationNo = i,
                    genotypeID = j
                )
            )
            sumFitness += FitnessFunctions.calculateBalanceFitness(
                telemetryDF = telemetry,
                timeStep = sim.timeStep
            )
        avgFitness.append(
            sumFitness / popCount
        )
        i += 5
    plt.plot(avgFitness)
    plt.show()




if __name__ == "__main__":
    plotGenerationalFitness()