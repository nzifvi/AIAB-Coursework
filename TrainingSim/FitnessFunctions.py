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
    # --- 1. Thresholds & Constants ---
    fallThreshold = 0.35  # ~20 degrees
    minStandingHeight = 0.28  # Threshold for "cowering"
    targetHeight = 0.41  # Ideal COG height

    # --- 2. Survival Logic ---
    # Define death by tilt or collapse
    is_fallen = (telemetryDF["roll"].abs() > fallThreshold) | \
                (telemetryDF["pitch"].abs() > fallThreshold) | \
                (telemetryDF["baseZ"] < minStandingHeight)

    fall_indices = telemetryDF.index[is_fallen].tolist()
    actualSteps = fall_indices[0] if fall_indices else len(telemetryDF)

    # Instant fail if it can't even last a fraction of a second
    if actualSteps < 15:
        return 0.0

    # Slice to the "Alive" portion
    aliveDF = telemetryDF.iloc[:actualSteps].copy()
    secondsSurvived = actualSteps * timeStep

    # --- 3. Stability Metrics (The "Static" Quality) ---
    # Height Error: How close to 0.41m?
    heightError = (aliveDF["baseZ"] - targetHeight).abs().mean()
    heightStability = 1.0 / (1.0 + heightError)

    # Orientation Error: How flat is the back?
    tiltError = (aliveDF["roll"].abs() + aliveDF["pitch"].abs()).mean()
    postureQuality = 1.0 / (1.0 + tiltError)

    # --- 4. Calmness Metrics (The "Anti-Jitter" Quality) ---
    # We calculate the "Delta" (change) between every frame.
    # High change = Vibration/High-frequency noise.

    # Vertical Jitter (Z-axis bouncing)
    z_velocity = aliveDF["baseZ"].diff().abs().mean()

    # Angular Jitter (Shaking side-to-side or front-to-back)
    # Using .diff() catches the high-frequency oscillation that .mean() misses
    roll_jitter = aliveDF["roll"].diff().abs().mean()
    pitch_jitter = aliveDF["pitch"].diff().abs().mean()
    total_jitter = (z_velocity * Z_VELOCITY_WEIGHT) + (roll_jitter * ROLL_JITTER_WEIGHT) + (pitch_jitter * PITCH_JITTER_WEIGHT)

    # The Calmness Bonus rewards a "still" body
    calmnessBonus = 1.0 / (1.0 + total_jitter)

    # --- 5. Final Assembly ---
    # Survival is the base. Stability and Calmness act as multipliers.
    # We square the Calmness and Stability to heavily favor "Quiet" winners.

    timeScore = secondsSurvived * SECONDS_SURVIVED_WEIGHT
    stabilityBonus = (heightStability * HEIGHT_STABILITY_WEIGHT) + (postureQuality * POSTURE_QUALITY_WEIGHT)

    scaledCalmBonus = calmnessBonus * CALMNESS_WEIGHT

    totalFitness = timeScore + stabilityBonus + scaledCalmBonus

    # --- 6. Effort & Smoothness Penalties ---
    # Scaled up to make "expensive" vibration more punishing than a calm stand
    effortPenalty = aliveDF["totalEffort"].mean() * EFFORT_PENALTY_WEIGHT

    # Final result (ensure it's not negative)
    finalScore = totalFitness - effortPenalty

    return max(0.0, finalScore)