import pybullet
import time
import numpy as np
import torch
from Simulator import Simulator
from RobotController import RobotController
from NeuralNetwork import NeuralNetwork

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

if __name__ == "__main__":
    runPDAutotune()