import copy
import random
import torch

class Evolver:
    def __init__(self, tournamentSize:int = 3, mutationRate:float = 0.15, sigma:float = 0.1):
        self.tournamentSize = tournamentSize
        self.mutationRate = mutationRate

        self.baseSigma = sigma
        self.currentSigma = sigma

        self.previousBestFitness = -float("inf")
        self.stagnationCounter = 0

    def produceNextGeneration(self, currentGeneration:list, kFittest = 2) -> list:
        nextGen = []
        populationSize = len(currentGeneration)

        sortedPop = sorted(currentGeneration, key = lambda x:x["fitness"], reverse = True)

        if self._checkStagnation(sortedPop[0]["fitness"]):
            print(f"    ! Stagnation detected. Sigma temporarily boosted to {self.currentSigma}")

        for i in range(0, kFittest):
            eliteGenotype = copy.deepcopy(sortedPop[i])
            eliteGenotype["generationNo"] += 1
            eliteGenotype["fitness"] = None
            nextGen.append(eliteGenotype)

        parentPoolSize = int(populationSize * 0.7)
        parentPool = sortedPop[:parentPoolSize]

        while len(nextGen) < populationSize:
            parent1 = self._tournamentSelection(parentPool)
            parent2 = self._tournamentSelection([ind for ind in parentPool if ind != parent1])

            child = self._mutate(
                self._reproduce(
                    parent1["genotype"],
                    parent2["genotype"]
                )
            )

            nextGen.append(
                {
                    "generationNo" : parent1.get("generationNo", 0) + 1,
                    "genotypeID"   : None,
                    "genotypeNN"   : None,
                    "genotype"     : child,
                    "fitness"      : None
                }
            )
        return nextGen

    def _tournamentSelection(self, pop:list):
        potentialParents = random.sample(pop, self.tournamentSize)
        return max(potentialParents, key = lambda individual: individual["fitness"])

    def _mutate(self, genotype:torch.Tensor) -> torch.Tensor:
        # gaussian mutation with per-gene masking.
        noise = torch.randn(genotype.shape) * self.currentSigma

        mutationMask = (torch.rand(genotype.shape) < self.mutationRate).float()

        return genotype + (noise * mutationMask)


    def _reproduce(self, p1:torch.Tensor, p2:torch.Tensor) -> torch.Tensor:
        # uses blend crossover
        # alpha = random.uniform(0.4, 0.6)
        # return alpha * p1 + (1 - alpha) * p2

        # uniform crossover
        mask = (torch.rand(p1.shape) > 0.5).float()
        return (mask * p1) + ((1 - mask) * p2)

    def _checkStagnation(self, currentBest:float, tolerance = 0.001):
        if currentBest > self.previousBestFitness + tolerance:
            self.previousBestFitness = currentBest
            self.stagnationCounter = 0
            self.currentSigma = self.baseSigma
            return False

        self.stagnationCounter += 1

        if self.stagnationCounter >= 10:
            self.currentSigma = min(self.currentSigma * 2.0, 0.5)
            return True

        return False