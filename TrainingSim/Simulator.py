import pandas
from RobotController import RobotController
from NeuralNetwork import NeuralNetwork
import time
import os
import sys
import random
import pybullet
import pybullet_data

class SupressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


class Simulator:
    def __init__(self, simDuration = 10.0, gui = False):
        self.simDuration = simDuration
        self.haveGUI = gui

        #with SupressOutput():
        if self.haveGUI:
            self.physicsClient = pybullet.connect(pybullet.GUI)
        else:
            self.physicsClient = pybullet.connect(pybullet.DIRECT)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.timeStep = 1.0 / 500

        self.plane = pybullet.loadURDF(
            "plane.urdf",
            basePosition = [0, 0, -0.1]
        )
        self._setupEnginePhysics()

    def _setupEnginePhysics(self):
        pybullet.setGravity(0, 0, -9.81)
        pybullet.setTimeStep(self.timeStep)

        pybullet.setPhysicsEngineParameter(
            fixedTimeStep=self.timeStep,
            numSolverIterations=50,
        )

    def reset(self):
        pybullet.resetSimulation()
        self.plane = pybullet.loadURDF(
            "plane.urdf",
            basePosition = [0, 0, 0]
        )
        self._setupEnginePhysics()

    def disconnect(self):
        pybullet.disconnect(self.physicsClient)

    def runSimulation(self, nn:NeuralNetwork):
        if not self.haveGUI:
            return self._runHeadless(nn)
        self.reset()

        nudgeRange = 0.035
        rollNudge = random.uniform(-nudgeRange, nudgeRange)
        pitchNudge = random.uniform(-nudgeRange, nudgeRange)

        nudgeOrientation = pybullet.getQuaternionFromEuler([rollNudge, pitchNudge, 0])

        robot = RobotController(
            basePosition = [0, 0, 0.40],
            baseOrientation = nudgeOrientation,
            nn = nn,
            maxTorque = 15
        )

        loopSteps = int(self.simDuration / self.timeStep)
        for step in range(loopSteps):
            robot.driveMotors(printTorques = self.haveGUI)
            pybullet.stepSimulation()
            robot.logTelemetry()

            if self.haveGUI:
                robotPos, _ = pybullet.getBasePositionAndOrientation(robot.robot)

                pybullet.resetDebugVisualizerCamera(
                    cameraDistance = 2.0,
                    cameraYaw = 50,
                    cameraPitch = -30,
                    cameraTargetPosition = robotPos
                )

                time.sleep(self.timeStep)

        return robot.getTelemetryData()

    def _runHeadless(self, nn: NeuralNetwork):
        # 1. Faster Reset: Reset simulation but keep the plane if possible
        # Or simply reset the robot's state if you keep the handle
        pybullet.resetSimulation()
        pybullet.loadURDF("plane.urdf", basePosition=[0, 0, -0.1], useFixedBase=True)
        self._setupEnginePhysics()

        nudge = 0.035
        nudgeOrientation = pybullet.getQuaternionFromEuler([
            random.uniform(-nudge, nudge),
            random.uniform(-nudge, nudge),
            0
        ])

        robot = RobotController(
            basePosition=[0, 0, 0.45],
            baseOrientation=nudgeOrientation,
            nn=nn
        )

        # 2. Local Variable Cache
        # Accessing 'pybullet.stepSimulation' through the module is slightly
        # slower than a local reference in a 5,000-iteration loop.
        step_sim = pybullet.stepSimulation
        drive = robot.driveMotors
        log = robot.logTelemetry

        loopSteps = int(self.simDuration / self.timeStep)

        # 3. The "Pure" Loop
        for _ in range(loopSteps):
            drive()
            step_sim()
            log()

        return robot.getTelemetryData()


