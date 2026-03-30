import pybullet
import pandas
import math
import torch
import NeuralNetwork
import tkinter as tk
from tkinter import ttk

LEARNING_BALANCE = True
LEARNING_WALK    = False

def degToRad(deg):
    return deg * math.pi / 180

class TorqueDisplay:
    def __init__(self, jointNames, maxTorque:float):
        self.maxTorque = maxTorque

        self.root = tk.Tk()
        self.root.title("Motor Torque Monitor")
        self.root.geometry("400x300")

        self.tree = ttk.Treeview(
            self.root,
            columns = ("Joint", "Torque"),
            show = "headings",
            height = 15
        )
        self.tree.heading("Joint", text = "Joint Name")
        self.tree.heading("Torque", text = "Torque (Nm)")
        self.tree.column("Joint", width = 250)
        self.tree.column("Torque", width = 150)
        self.tree.pack(fill = "both", expand = True)

        self.tree.tag_configure("normal", background = "white")
        self.tree.tag_configure("max", background = "red", foreground = "white")
        self.tree.tag_configure("warning", background = "orange")

        self.items = {}
        for name in jointNames:
            item = self.tree.insert("", "end", values=(name, "0.000"))
            self.items[name] = item

    def update(self, torqueDict):
        for name, torque in torqueDict.items():
            self.tree.item(self.items[name], values = (name, f"{torque:.3f}"))

            absTorque = abs(torque)
            if absTorque >= self.maxTorque:
                self.tree.item(self.items[name], tags=("max",))
            else:
                self.tree.item(self.items[name], tags=("normal",))

        self.root.update()

    def destroy(self):
        self.root.destroy()
class RobotController:
    def __init__(self, nn: NeuralNetwork.NeuralNetwork, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
                 maxTorque: float = 5.0, ):
        self.robot = pybullet.loadURDF(
            r"..\TrainingSim\LeggedRobotsForBullet\quadrupedal\urdf\quadrupedal.urdf",
            basePosition=basePosition,
            baseOrientation=baseOrientation,
            useFixedBase=False
        )

        self.timeStep = pybullet.getPhysicsEngineParameters()["fixedTimeStep"]
        self.maxTorque = maxTorque

        self.jointIDs = []
        self.jointNames = {}
        self.linkNames = {}  # FIX: Added for link-based sensor lookups

        for j in range(pybullet.getNumJoints(self.robot)):
            info = pybullet.getJointInfo(self.robot, j)
            idx = info[0]
            jName = info[1].decode("utf-8")
            lName = info[12].decode("utf-8")  # Index 12 is the Link Name

            self.jointIDs.append(idx)
            self.jointNames[jName] = idx
            self.linkNames[lName] = idx  # Map link name to the same index

        # Disable default motors for torque control
        for jointID in self.jointIDs:
            pybullet.setJointMotorControl2(
                bodyUniqueId=self.robot,
                jointIndex=jointID,
                controlMode=pybullet.VELOCITY_CONTROL,
                force=0
            )

        self.robotNN = nn
        self.telemetryLog = []
        self.printCounter = 0

        # Display setup
        jointNamesList = list(self.jointNames.keys())
        self.torqueDisplay = TorqueDisplay(jointNamesList, self.maxTorque)

        self.lastTorques = {}
        self.ticks = 0
        self.simTime = 0.0

    def __del__(self):
        try:
            self.torqueDisplay.destroy()
        except:
            pass

    def checkFeetSensors(self) -> list:

        footLinks = [
            "rf_lower_link",
            "lf_lower_link",
            "rh_lower_link",
            "lh_lower_link"
        ]
        contacts = []

        for link in footLinks:
            contactPoints = pybullet.getContactPoints(
                bodyA = self.robot,
                linkIndexA = self.linkNames[link]
            )
            contacts.append(
                1.0 if len(contactPoints) > 0 else 0.0
            )
        return contacts

    def getRobotState(self) -> torch.Tensor:
        jointPositions  = []
        jointVelocities = []

        for jointID in self.jointIDs:
            state = pybullet.getJointState(
                self.robot,
                jointID
            )
            jointPositions.append(state[0])
            jointVelocities.append(state[1])

        pos, orient = pybullet.getBasePositionAndOrientation(self.robot)
        euler = pybullet.getEulerFromQuaternion(orient)
        linearVelocity, angularVelocity = pybullet.getBaseVelocity(self.robot)

        footContacts = self.checkFeetSensors()

        self.simTime = self.ticks * self.timeStep
        phaseSin = math.sin(2 * math.pi * self.simTime)
        phaseCos = math.cos(2 * math.pi * self.simTime)

        inputs = (
            jointPositions +
            jointVelocities +
            list(euler) +
            list(linearVelocity) +
            list(angularVelocity) +
            footContacts
        )

        while len(inputs) < 48:
            inputs.append(0.0)

        return torch.tensor(
            [inputs],
            dtype = torch.float32
        )

    def driveMotors(self, torqueScale=1.0, kp = 100.0, kd = 10.0, printTorques:bool = False) -> None:
        stateTensor = self.getRobotState()
        nnOutput = self.robotNN.inference(stateTensor).squeeze(0)

        numJoints = len(self.jointIDs)

        jointStates = pybullet.getJointStates(self.robot, self.jointIDs)
        currentPosition = torch.tensor([state[0] for state in jointStates])
        currentVelocity = torch.tensor([state[1] for state in jointStates])

        targetPositions = nnOutput[:numJoints]
        nnTorqueOffsets = nnOutput[numJoints:2*numJoints] if nnOutput.shape[0] >= 2*numJoints else torch.zeros(numJoints)

        maxAngle = 1.0
        targetPositions = torch.clamp(targetPositions, -maxAngle, maxAngle)

        pdTorques = kp * (targetPositions - currentPosition) + kd * (0.0 - currentVelocity)
        rawTorques = pdTorques + (nnTorqueOffsets * torqueScale)

        finalTorques = self.maxTorque * torch.tanh(rawTorques / self.maxTorque)


        torqueDict = {}
        for i in range(numJoints):
            pybullet.setJointMotorControl2(
                bodyUniqueId = self.robot,
                jointIndex = self.jointIDs[i],
                controlMode = pybullet.TORQUE_CONTROL,
                force = finalTorques[i].item(),
            )



            info = pybullet.getJointInfo(self.robot, self.jointIDs[i])
            jointName = info[1].decode("utf-8")
            torqueDict[jointName] = finalTorques[i].item()

        self.lastTorques = torqueDict
        self.ticks = self.ticks + 1

        if printTorques and self.printCounter % 10 == 0:
            self.torqueDisplay.update(torqueDict)
        if printTorques:
            self.printCounter += 1

    def logTelemetry(self) -> None:
        robotPos, robotOrientation = pybullet.getBasePositionAndOrientation(self.robot)
        linearVelocity, angularVelocity = pybullet.getBaseVelocity(self.robot)

        jointStates = pybullet.getJointStates(
            self.robot,
            self.jointIDs
        )

        contacts = self.checkFeetSensors()

        euler = pybullet.getEulerFromQuaternion(robotOrientation)

        self.telemetryLog.append(
            {
                "baseX": robotPos[0],
                "baseY": robotPos[1],
                "baseZ": robotPos[2],
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
                "linearVelocityX": linearVelocity[0],
                "linearVelocityY": linearVelocity[1],
                "linearVelocityZ": linearVelocity[2],
                "angularVelocityX": angularVelocity[0],
                "angularVelocityY": angularVelocity[1],
                "angularVelocityZ": angularVelocity[2],
                "contacts": contacts,
                "jointPositions": [state[0] for state in jointStates],
                "jointVelocities": [state[1] for state in jointStates],
                "totalEffort": sum(abs(t) for t in self.lastTorques.values()) if self.lastTorques else 0.0
            }
        )

    def getTelemetryData(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            self.telemetryLog
        )
