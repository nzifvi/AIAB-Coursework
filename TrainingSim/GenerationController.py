import Evolver
import Simulator

import os
import torch
torch.set_num_threads(1)
import pandas
import NeuralNetwork

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor
import functools

import FitnessFunctions

RETRIAL_AMOUNT = 4
WORKER_COUNT   = 10

def evaluateBatch(args):
    genotypeID, nn = args
    fitness = 0.0
    with Simulator.SupressOutput():
        sim = Simulator.Simulator(
            simDuration = 10.0,
            gui = False
        )
        try:
            for i in range(0, RETRIAL_AMOUNT):
                trialTelemetry = sim.runSimulation(nn = nn)
                fitness += FitnessFunctions.calculateBalanceFitness(
                    trialTelemetry,
                    sim.timeStep
                )
        except Exception as e:
            raise RuntimeError(f"! WorkerThread has caused a simulation crash at genotype {genotypeID}: {e}")
        finally:
            sim.disconnect()
    return genotypeID, fitness / RETRIAL_AMOUNT

class GenerationController:
    def __init__(self, popSize:int, checkpointControl:int):
        self.generationNo = self._readRecentCheckpoint()
        self.populationSize = popSize
        self.currentGeneration = []
        self.nextGeneration = []

        # each member of a generation represented as...
        # {
        #    "generationNo": parent1.get("generationNo", 0) + 1,
        #    "genotypeID": None,
        #    "genotypeNN": None,
        #    "genotype": child,
        #    "fitness": None
        # }

        self.checkpointControl = checkpointControl

        self.layer1Shape = (48, 64, 3072, 64)
        self.layer2Shape = (64, 32, 2048, 32)
        self.layer3Shape = (32, 48, 1536, 48)
        self.layer4Shape = (48, 24, 1152, 24)

        self.evolver = Evolver.Evolver(
            tournamentSize = 5,
            mutationRate = 0.1,
            sigma = 0.1
        )
        # gen 40: mutation rate boosted from 0.15 to 0.2. sigma boosted from 0.05 to 0.25
        # gen 75: mutation rate reduced to 0.1 from 0.2. sigma reduced to 0.1 from 0.25

        if self.generationNo == 0: # create progenitor generation
            self._initProgenitorGeneration()
        else: # load descendant generation
            self._initDescendantGeneration()

    def _initProgenitorGeneration(self):
        self._createGenerationDirectory()

        for i in range(0, self.populationSize):
            weights, biases = self._createProgenitorNeuralNetwork()
            nn = NeuralNetwork.NeuralNetwork(weights = weights, biases = biases)
            self.currentGeneration.append(
                {
                    "generationNo" : 0,
                    "genotypeID"   : i,
                    "genotypeNN"   : nn,
                    "genotype"     : self._flattenNeuralNetwork(nn),
                    "fitness"      : None
                }
            )

    def _initDescendantGeneration(self):
        self.populationSize = self._readPopulationSize()

        self.currentGeneration = self._loadGeneration()

    def _readRecentCheckpoint(self) -> int:
        try:
            with open("C:/Users/benja/Desktop/AIAB Coursework/Data/Generations/GenerationCount.txt", "r") as f:
                return int(f.readline())
        except Exception as e:
            raise ValueError("Cannot load GenerationCount.txt") from e

    def _writeRecentCheckpoint(self, newCheckpoint ) -> None:
        try:
            with open(r"C:\Users\benja\Desktop\AIAB Coursework\Data\Generations\GenerationCount.txt", "w") as f:
                f.write(str(newCheckpoint))
        except Exception as e:
            raise ValueError("Cannot write GenerationCount.txt") from e

    def _readPopulationSize(self) -> int:
        filePath = f"C:/Users/benja/Desktop/AIAB Coursework/Data/Generations/Generation{self.generationNo}/PopulationCount.txt"
        try:
            with open(filePath, "r") as f:
                return int(f.read())
        except Exception as e:
            raise ValueError(f"Cannot read PopulationCount from {filePath}") from e

    def _createProgenitorNeuralNetwork(self) -> tuple:
        totalParams = 7976
        genotype = torch.randn(totalParams)
        weights, biases = self._unflattenNeuralNetwork(genotype)
        return weights, biases

    def _saveGeneration(self) -> None:
        self._createGenerationDirectory()
        allWeights = []
        allBiases = []

        for indiv in self.currentGeneration:
            weights, biases = self._unflattenNeuralNetwork(indiv["genotype"])
            allWeights.append(weights)
            allBiases.append(biases)

        self._saveNeuralNetworks(allWeights, allBiases)

    def _saveNeuralNetworks(self, weights, biases) -> None:
        generationDirectoryPath = "C:/Users/benja/Desktop/AIAB Coursework/Data/Generations/Generation" + str(
            self.generationNo)
        try:
            for i in range(self.populationSize):
                weightsPath = os.path.join(generationDirectoryPath, f"NeuralNetworks/Genotype{i}/Weights/weights.pt")
                biasesPath = os.path.join(generationDirectoryPath, f"NeuralNetworks/Genotype{i}/Biases/biases.pt")

                torch.save(weights[i], weightsPath)
                torch.save(biases[i], biasesPath)
        except Exception as e:
            raise ValueError(f"! Error saving binary data at generation {self.generationNo}: {e}")

    def _loadGeneration(self) -> list:
        genPopulation = []
        for genotypeID in range(0, self.populationSize):
            nn = NeuralNetwork.NeuralNetwork(self.generationNo, genotypeID)
            genPopulation.append({
                "generationNo": self.generationNo,
                "genotypeID": genotypeID,
                "genotypeNN": nn,
                "genotype": self._flattenNeuralNetwork(nn),
                "fitness": None
            })
        return genPopulation

    def _flattenNeuralNetwork(self, nn:NeuralNetwork.NeuralNetwork) -> torch.Tensor:
        parameters = []
        for i in range(len(nn.weights)):
            parameters.append(nn.weights[i].flatten())
            parameters.append(nn.biases[i].flatten())
        return torch.cat(parameters)

    def _unflattenNeuralNetwork(self, genotype:torch.Tensor) -> tuple:
        unflattenedWeights = []
        unflattenedBiases = []
        j = 0

        for shape in [self.layer1Shape, self.layer2Shape, self.layer3Shape, self.layer4Shape]:
            inDim, outDim, weightsArea, biasArea = shape

            unflattenedWeights.append(
                genotype[j : j + weightsArea].reshape(inDim, outDim)
            )
            j = j + weightsArea
            unflattenedBiases.append(
                genotype[j : j + biasArea].reshape(1, outDim)
            )
            j = j + biasArea

        return unflattenedWeights, unflattenedBiases

    def _createGenerationDirectory(self) -> None:
        try:
            newGenerationDirectoryPath = "C:/Users/benja/Desktop/AIAB Coursework/Data/Generations/Generation" + str(self.generationNo)
            os.makedirs(newGenerationDirectoryPath, exist_ok = True)
            os.makedirs(newGenerationDirectoryPath + "/TelemetryData", exist_ok = True)
            os.makedirs(newGenerationDirectoryPath + "/NeuralNetworks", exist_ok = True)

            for i in range(self.populationSize):
                os.makedirs(newGenerationDirectoryPath + "/TelemetryData/" + "Genotype" +str(i), exist_ok = True)

            for i in range(self.populationSize):
                os.makedirs(newGenerationDirectoryPath + "/NeuralNetworks/" + "Genotype" + str(i), exist_ok = True)
                os.makedirs(newGenerationDirectoryPath + "/NeuralNetworks/" + "Genotype" + str(i) + "/Weights", exist_ok = True)
                os.makedirs(newGenerationDirectoryPath + "/NeuralNetworks/" + "Genotype" + str(i) + "/Biases", exist_ok = True)

            file = open(newGenerationDirectoryPath + "/PopulationCount.txt", "w")
            file.write(str(self.populationSize))
            file.close()

            file = open(
                newGenerationDirectoryPath + "/NeuralNetworks/HyperParameters.txt", "w"
            )
            file.close()

            for i in range(self.populationSize):
                file = open(
                    newGenerationDirectoryPath + "/TelemetryData/Genotype" + str(i) + "/LogCount.txt", "w"
                )
                file.write(str(0))
                file.close()

            file.close()
        except Exception as e:
            print(e)

    def _runSimulator(self):
        tasks = [
            (
                indiv["genotypeID"],
                indiv["genotypeNN"]
            )
            for indiv in self.currentGeneration
        ]

        with ProcessPoolExecutor(max_workers = WORKER_COUNT) as executor:
            results = list(
                tqdm(
                    executor.map(evaluateBatch, tasks, chunksize = 1), total = self.populationSize, leave=False
                )
            )

        for genotypeID, fitness in results:
            self.currentGeneration[genotypeID]["fitness"] = fitness

    def run(self, genotypeReverificationNo:int):
        self._runSimulator()

        scores = [indiv["fitness"] for indiv in self.currentGeneration if indiv["fitness"] is not None]
        avgFitness = sum(scores) / len(scores)
        bestFitness = max(scores)
        worstFitness = min(scores)

        if self.generationNo == 0 or self.generationNo % self.checkpointControl == 0:
            self._saveGeneration()
            self._writeRecentCheckpoint(self.generationNo)

        self.nextGeneration = self.evolver.produceNextGeneration(
            self.currentGeneration
        )

        for i in range(len(self.nextGeneration)):
            self.nextGeneration[i]["genotypeID"] = i
            weights, biases = self._unflattenNeuralNetwork(self.nextGeneration[i]["genotype"])
            self.nextGeneration[i]["genotypeNN"] = NeuralNetwork.NeuralNetwork(
                weights = weights,
                biases  = biases
            )

        self.currentGeneration = self.nextGeneration
        self.generationNo = self.generationNo + 1
        return avgFitness, bestFitness, worstFitness

    def _getFitness(self, i):
        return self.currentGeneration[i]["fitness"]

    def _reevaluateElites(self, verifyNum, extraTrialCount):
        sortedGeneration = sorted(self.currentGeneration, key=lambda x:x["fitness"], reverse=True)
        topGenotypes = sortedGeneration[:verifyNum]

        tasks = [(e["genotypeID"], e["genotypeNN"]) for e in topGenotypes for _ in range(extraTrialCount)]

        with ProcessPoolExecutor(max_workers = WORKER_COUNT) as executor:
            results = list(executor.map(evaluateBatch, tasks))

        values = {}
        for elite in topGenotypes:
            values[elite["genotypeID"]] = [elite["fitness"]]

        for result in results:
            genotypeID, fitness = result
            values[genotypeID].append(fitness)

        for indiv in self.currentGeneration:
            genotypeID = indiv["genotypeID"]
            if genotypeID in values:
                fitnessValues = values[genotypeID]

                fitnessValueSum = 0.0
                for fitness in fitnessValues:
                    fitnessValueSum = fitnessValueSum + fitness

                avgFitness = fitnessValueSum / len(fitnessValues)

                indiv["fitness"] = avgFitness






def calculateAverageFitness(generation:list) -> float:
    sum = 0.0
    for indiv in generation:
        if indiv["fitness"] is None:
            raise RuntimeError("! Attempted to calculate average fitness of a generation before the generation was simulated")
        else:
            sum += indiv["fitness"]
    return sum / len(generation)
