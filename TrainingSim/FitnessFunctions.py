import pandas
import numpy as np

SECONDS_SURVIVED_WEIGHT = 10.0
HEIGHT_STABILITY_WEIGHT = 10.0
POSTURE_QUALITY_WEIGHT  = 10.0
CALMNESS_WEIGHT         = 20.0
EFFORT_PENALTY_WEIGHT   = 0.005
Z_VELOCITY_WEIGHT       = 5.0
ROLL_JITTER_WEIGHT      = 10.0
PITCH_JITTER_WEIGHT     = 10.0

def calculateBalanceFitness(telemetryDF: pandas.DataFrame, timeStep: float) -> float:
    fallThreshold     = 0.35
    minStandingHeight = 0.28
    targetHeight      = 0.41

    isFallen = (telemetryDF["roll"].abs() > fallThreshold) | \
                (telemetryDF["pitch"].abs() > fallThreshold) | \
                (telemetryDF["baseZ"] < minStandingHeight)

    fall_indices = telemetryDF.index[isFallen].tolist()
    actualSteps  = fall_indices[0] if fall_indices else len(telemetryDF)

    if actualSteps < 15:
        return 0.0

    aliveDF         = telemetryDF.iloc[:actualSteps].copy()
    secondsSurvived = actualSteps * timeStep

    heightError     = (aliveDF["baseZ"] - targetHeight).abs().mean()
    heightStability = 1.0 / (1.0 + heightError)

    tiltError      = (aliveDF["roll"].abs() + aliveDF["pitch"].abs()).mean()
    postureQuality = 1.0 / (1.0 + tiltError)

    zVelocity = aliveDF["baseZ"].diff().abs().mean()

    rollJitter   = aliveDF["roll"].diff().abs().mean()
    pitchJitter  = aliveDF["pitch"].diff().abs().mean()
    totalJitter  = (zVelocity * Z_VELOCITY_WEIGHT) + (rollJitter * ROLL_JITTER_WEIGHT) + (pitchJitter * PITCH_JITTER_WEIGHT)

    calmnessBonus = 1.0 / (1.0 + totalJitter)

    timeScore      = secondsSurvived * SECONDS_SURVIVED_WEIGHT
    stabilityBonus = (heightStability * HEIGHT_STABILITY_WEIGHT) + (postureQuality * POSTURE_QUALITY_WEIGHT)

    scaledCalmBonus = calmnessBonus * CALMNESS_WEIGHT

    totalFitness = timeScore + stabilityBonus + scaledCalmBonus

    effortPenalty = aliveDF["totalEffort"].mean() * EFFORT_PENALTY_WEIGHT

    finalScore = totalFitness - effortPenalty

    return max(0.0, finalScore)