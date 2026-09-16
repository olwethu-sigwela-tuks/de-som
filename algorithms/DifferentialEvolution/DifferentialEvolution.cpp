#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <unistd.h>
#include <gsl/gsl_rng.h>
#include "DataFileReader.h"
#include "DataPattern.h"
#include "DifferentialEvolution.h"

std::vector<std::vector<double>> generateInitPopulation(int NP, int neuronSize, int numNeurons, std::vector<std::vector<double>>& dataset){
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 42);

    std::vector<double> minVal(neuronSize, DBL_MAX);
    std::vector<double> maxVal(neuronSize, -DBL_MAX);

    for (const auto& sample : dataset){
        for (int k = 0; k < neuronSize; ++k){
            if (sample[k] < minVal[k]){
                minVal[k] = sample[k];
            }

            if (sample[k] > maxVal[k]){
                maxVal[k] = sample[k];
            }
        }
    }

    int totalNumWeights = neuronSize * numNeurons;
    std::vector<std::vector<double>> population(NP, std::vector<double>(totalNumWeights));
    
    for (int n = 0; n < NP; ++n){
        for (int i = 0; i < totalNumWeights; ++i){
            int k = i % neuronSize;
            population[n][i] = minVal[k] + (maxVal[k] - minVal[k]) * gsl_rng_uniform(rng);
        }
    }

    gsl_rng_free(rng);
    return population;
}

std::vector<int> getRandomIndices(gsl_rng* rng, int currIndex, int NP){
    std::vector<int> indices;
    indices.reserve(3);

    while(indices.size() < 3){
        int randomIndex = gsl_rng_uniform_int(rng, NP);
        if ((randomIndex != currIndex) && (std::find(indices.begin(), indices.end(), randomIndex) == indices.end())){
            indices.push_back(randomIndex);
        }
    }

    return indices;
}

double euclideanDistance(const std::vector<double>& vect1, const std::vector<double>& vect2){
    if (vect1.size() != vect2.size()){
        throw std::invalid_argument("length of vect1 does not equal length of vect2");
    }

    double sum = 0;
    for (int i = 0; i < vect1.size(); i++){
        double dx = vect2 [i]- vect1[i];
        sum += dx * dx;
    }

    return std::sqrt(sum);
}

double quantizationError(const std::vector<double>& weights, const std::vector<std::vector<double>>& dataset){
    int neuronSize = dataset[0].size();

    double totalDistance = 0;
    for (int d = 0; d < dataset.size(); d++){
        double minimumDistance = DBL_MAX;

        for (int i = 0; i < weights.size(); i += neuronSize){

            std::vector<double> neuron(weights.data() + i, weights.data() + i + neuronSize);

            double distance = euclideanDistance(neuron, dataset[d]);

            if (distance < minimumDistance)
            {
                minimumDistance = distance;
            }
        }

        totalDistance += minimumDistance;

    }

    return totalDistance / dataset.size();
}

std::string vectorToString(const std::vector<double>& vect){

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << "[";

    for (size_t i = 0; i < vect.size(); ++i){
        if (i > 0){
            oss << ", ";
        }
        oss << vect[i];
    }

    oss << "]";

    return oss.str();
}

std::string vector2dToString(const std::vector<std::vector<double>>& vect){
    std::string output;

    for (int i = 0; i < vect.size(); ++i){
        if (i == 0){
            output = output + "[" + vectorToString(vect[i]);
        }
        else {
            output = output + ", " + vectorToString(vect[i]);
        }
    }

    output = output + "]";

    return output;
}

std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP,  int nGenerations, double CR, double F){
    auto population = generateInitPopulation(NP, neuronSize, numNeurons, dataset);

    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 42);
    int totalWeights = neuronSize * numNeurons;
    
    double lowestQuantizationError = DBL_MAX;
    std::vector<double> bestWeights;

    for (int n = 0; n < nGenerations; n++){
        std::vector<std::vector<double>> newPopulation;

        for (int i = 0; i < population.size(); i++){
            auto randomIndices = getRandomIndices(rng, i, NP);
            int r1 = randomIndices[0];
            int r2 = randomIndices[1];
            int r3 = randomIndices[2];

            std::vector<double> v;

            for (int x = 0; x < population[0].size(); x++)
            {
                double vComponent = population[r1][x] + F * (population[r2][x] - population[r3][x]);
                v.push_back(vComponent);
            }

            std::vector<double> u;
            int randIdx = gsl_rng_uniform_int(rng, totalWeights);

            for(int j = 0; j < population[i].size(); j++)
            {
                double randJ = gsl_rng_uniform(rng);

                if ((randJ <= CR) || (j == randIdx)){
                    u.push_back(v[j]);
                }
                else if ((randJ > CR) && (j != randIdx)){
                    u.push_back(population[i][j]);
                }
            }
            
            double parentError = quantizationError(population[i], dataset);
            double childError = quantizationError(u, dataset);

            if (childError < parentError)
            {
                newPopulation.push_back(u);
            }
            else{
                newPopulation.push_back(population[i]);
            }

            if (childError < lowestQuantizationError){
                lowestQuantizationError = childError;
                bestWeights = u;
            }
            if (parentError < lowestQuantizationError){
                lowestQuantizationError = parentError;
                bestWeights = population[i];
            }
        }

        population = newPopulation;
    }

    gsl_rng_free(rng);
    return bestWeights;
}

std::vector<std::vector<double>> loadDataSet(char* dataFileName, char* namesFileName){

    DataFileReader reader(dataFileName, namesFileName);

    unsigned int numRecords = reader.getRecordCount();
    unsigned int numFeatures = reader.getAttributeCount();

    std::vector<std::vector<double>> dataset;
    dataset.reserve(numRecords);

    for (unsigned int i = 0; i < numRecords; ++i){
        DataPattern record = reader.getRecord(i);
        std::vector<double> recordVector(numFeatures);

        for (unsigned int j = 0; j < numFeatures; ++j){
            recordVector[j] = record.getComponent(j);
        }

        dataset.push_back(recordVector);
    }

    return dataset;



}

std::vector<double> loadMapFile(const std::string& mapFileName){
    std::ifstream file(mapFileName);
    std::string line;
    std::vector<double> weights;

    std::getline(file, line);

    while(std::getline(file, line)){
        if (line.empty()){
            continue;
        }
        std::istringstream iss(line);
        double val;
        while(iss >> val){
            weights.push_back(val);
        }
    }

    return weights;
}

std::vector<double> runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR){
    int neuronSize = trainingDataset[0].size();
    auto bestWeights = differentialEvolution(trainingDataset, neuronSize, numNeurons, NP, numGenerations, CR, F);

    double trainingError = quantizationError(bestWeights, trainingDataset);
    double testError = quantizationError(bestWeights, testDataset);

    std::cout << "Best training error: " << trainingError << std::endl;
    std::cout << "Best test error: " << testError << std::endl;

    std::vector<double> trainTestError(2);
    trainTestError[0] = trainingError;
    trainTestError[1] = testError;

    return trainTestError;
}

#ifndef NO_MAIN
int main(int argc, char** argv){

    if (argc < 9){
        std::cerr << "Usage: " << argv[0] << " <datasetFileName> <namesFileName> <NP> <numGenerations> <numNeurons> <CR> <F> <baselineWeightsFileName>" << std::endl;
        return 1;
    }

    auto dataset = loadDataSet(argv[1], argv[2]);
    int neuronSize = dataset[0].size();
    int NP = std::stoi(argv[3]);
    int numGenerations = std::stoi(argv[4]);
    int numNeurons = std::stoi(argv[5]);
    double CR = std::stod(argv[6]);
    double F = std::stod(argv[7]);
    std::string baselineWeightsFileName = argv[8];

    auto bestWeights = differentialEvolution(dataset, neuronSize, numNeurons, NP, numGenerations, CR, F);

    std::cout << std::endl << std::endl;

    std::cout << "Best Weights:" << std::endl << std::endl << vectorToString(bestWeights) <<std::endl << std::endl;
    std::cout << "Best error: " << quantizationError(bestWeights, dataset) << std::endl;

    auto baselineWeights = loadMapFile(baselineWeightsFileName);

    std::cout << std::endl << std::endl;

    std::cout << "Baseline Weights:" << std::endl << std::endl << vectorToString(baselineWeights) <<std::endl << std::endl;
    std::cout << "Baseline error: " << quantizationError(baselineWeights, dataset) << std::endl;

    return 0;

}
#endif
