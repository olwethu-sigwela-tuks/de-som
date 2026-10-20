#include <random>
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
#include <numeric>
#include "DataFileReader.h"
#include "DataPattern.h"
#include "DifferentialEvolution.h"

std::vector<std::vector<double>> generateInitPopulation(int NP, int neuronSize, int numNeurons, std::vector<std::vector<double>>& dataset, std::mt19937& generator){
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
    
    std::vector<std::uniform_real_distribution<double>> distributions;
    for (int k = 0; k < neuronSize; ++k){
        double low = minVal[k];
        double high = maxVal[k];
        if (high - low < 1e-12){
            low -= 1e-6;
            high += 1e-6;
        }
        distributions.emplace_back(low, high);
    }

    int totalNumWeights = neuronSize * numNeurons;
    std::vector<std::vector<double>> population(NP, std::vector<double>(totalNumWeights));
    
    for (int n = 0; n < NP; ++n){
        for (int i = 0; i < totalNumWeights; ++i){
            population[n][i] = distributions[i % neuronSize](generator);
        }
    }

    return population;
}

std::vector<int> getRandomIndices(std::mt19937& generator, int currIndex, int NP){
    std::vector<int> indices;
    indices.reserve(3);
    std::uniform_int_distribution<int> distrib(0, NP - 1);

    while(indices.size() < 3){
        int randomIndex = distrib(generator);
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

    double sum = 0.0;
    for (size_t i = 0; i < vect1.size(); i++){
        double dx = vect2[i] - vect1[i];
        sum += dx * dx;
    }

    return std::sqrt(sum);
}

double quantizationError(const std::vector<double>& weights, const std::vector<std::vector<double>>& dataset){
    size_t neuronSize = dataset[0].size();
    size_t numNeurons = weights.size() / neuronSize;
    double totalDistance = 0.0;

    for (size_t d = 0; d < dataset.size(); ++d){
        const double* x = dataset[d].data();
        double minDistSq = DBL_MAX;

        for (size_t i = 0; i < numNeurons; ++i){
            const double* w = weights.data() + i * neuronSize;
            double distSq = 0.0;
            for (size_t k = 0; k < neuronSize; ++k){
                double diff = x[k] - w[k];
                distSq += diff * diff;
            }
            if (distSq < minDistSq){
                minDistSq = distSq;
            }
        }
        totalDistance += std::sqrt(minDistSq);
    }
    return totalDistance / dataset.size();
}

double topographicError(const std::vector<double>& weights, int width, int height, bool isHexa, const std::vector<std::vector<double>>& dataset){
    size_t neuronSize = dataset[0].size();
    size_t numNeurons = width * height;
    int errorCount = 0;

    for (size_t d = 0; d < dataset.size(); ++d){
        const double* x = dataset[d].data();
        int bmu1 = -1;
        int bmu2 = -1;
        double minDistSq1 = DBL_MAX;
        double minDistSq2 = DBL_MAX;

        for (size_t i = 0; i < numNeurons; ++i){
            const double* w = weights.data() + i * neuronSize;
            double distSq = 0.0;
            for (size_t k = 0; k < neuronSize; ++k){
                double diff = x[k] - w[k];
                distSq += diff * diff;
            }
            if (distSq < minDistSq1){
                minDistSq2 = minDistSq1;
                bmu2 = bmu1;
                minDistSq1 = distSq;
                bmu1 = i;
            } else if (distSq < minDistSq2){
                minDistSq2 = distSq;
                bmu2 = i;
            }
        }

        int y1 = bmu1 / width;
        int x1 = bmu1 % width;
        int y2 = bmu2 / width;
        int x2 = bmu2 % width;
        double dx = x1 - x2;
        double dy = y1 - y2;
        if (isHexa && (y1 % 2 != y2 % 2)){
            if (y1 % 2 == 1) dx += 0.5;
            else dx -= 0.5;
        }
        double gridDist = std::sqrt(dx * dx + dy * dy * (isHexa ? 0.75 : 1.0));

        if (gridDist > 1.05){
            errorCount++;
        }
    }

    return (double)errorCount / dataset.size();
}

void minMaxScaleTrainTest(std::vector<std::vector<double>>& trainSet, std::vector<std::vector<double>>& testSet){
    if (trainSet.empty()){
        return;
    }
    size_t numFeatures = trainSet[0].size();
    size_t nTrain = trainSet.size();
    std::vector<double> minVal(numFeatures, DBL_MAX);
    std::vector<double> maxVal(numFeatures, -DBL_MAX);
    for (size_t i = 0; i < nTrain; ++i){
        for (size_t j = 0; j < numFeatures; ++j){
            if (trainSet[i][j] < minVal[j]) minVal[j] = trainSet[i][j];
            if (trainSet[i][j] > maxVal[j]) maxVal[j] = trainSet[i][j];
        }
    }
    for (size_t i = 0; i < nTrain; ++i){
        for (size_t j = 0; j < numFeatures; ++j){
            double range = maxVal[j] - minVal[j];
            if (range > 1e-12){
                trainSet[i][j] = (trainSet[i][j] - minVal[j]) / range;
            } else {
                trainSet[i][j] = 0.0;
            }
        }
    }
    for (size_t i = 0; i < testSet.size(); ++i){
        for (size_t j = 0; j < numFeatures; ++j){
            double range = maxVal[j] - minVal[j];
            if (range > 1e-12){
                testSet[i][j] = (testSet[i][j] - minVal[j]) / range;
            } else {
                testSet[i][j] = 0.0;
            }
        }
    }
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

std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP, int nGenerations, double CR, double F, std::mt19937& gen){
    auto population = generateInitPopulation(NP, neuronSize, numNeurons, dataset, gen);
    std::uniform_int_distribution<int> distribVector(0, (neuronSize * numNeurons) - 1);
    std::uniform_real_distribution<double> distribDouble(0.0, 1.0);
    
    std::vector<double> popErrors(NP);
    double lowestQuantizationError = DBL_MAX;
    std::vector<double> bestWeights;

    for (int i = 0; i < NP; ++i) {
        popErrors[i] = quantizationError(population[i], dataset);
        if (popErrors[i] < lowestQuantizationError) {
            lowestQuantizationError = popErrors[i];
            bestWeights = population[i];
        }
    }

    for (int n = 0; n < nGenerations; n++){
        std::vector<std::vector<double>> newPopulation;
        newPopulation.reserve(NP);

        for (size_t i = 0; i < population.size(); i++){
            auto randomIndices = getRandomIndices(gen, i, NP);
            int r1 = randomIndices[0];
            int r2 = randomIndices[1];
            int r3 = randomIndices[2];

            std::vector<double> v;
            v.reserve(population[0].size());
            for (size_t x = 0; x < population[0].size(); x++){
                double vComponent = population[r1][x] + F * (population[r2][x] - population[r3][x]);
                v.push_back(vComponent);
            }

            std::vector<double> u;
            u.reserve(population[i].size());
            int randIdx = distribVector(gen);

            for(size_t j = 0; j < population[i].size(); j++){
                double randJ = distribDouble(gen);
                if ((randJ <= CR) || (j == (size_t)randIdx)){
                    u.push_back(v[j]);
                } else {
                    u.push_back(population[i][j]);
                }
            }
            
            double childError = quantizationError(u, dataset);

            if (childError < popErrors[i]){
                newPopulation.push_back(u);
                popErrors[i] = childError;
                if (childError < lowestQuantizationError){
                    lowestQuantizationError = childError;
                    bestWeights = u;
                }
            } else {
                newPopulation.push_back(population[i]);
            }
        }

        population = newPopulation;
    }

    return bestWeights;
}

TrainTestSplit trainTestSplit(std::vector<std::vector<double>>& dataset, double testRatio, std::mt19937& g){
    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    TrainTestSplit split;
    size_t trainDatasetSize = std::round((1.0 - testRatio) * dataset.size());

    for (size_t i = 0; i < indices.size(); ++i){
        if (i < trainDatasetSize){
            split.train.push_back(dataset[indices[i]]);
        } else {
            split.test.push_back(dataset[indices[i]]);
        }
    }
    
    return split; 
}

std::vector<std::vector<std::vector<double>>> kFoldSplit(std::vector<std::vector<double>>& dataset, int numFolds, std::mt19937& g){
    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<std::vector<double>>> folds(numFolds);
    size_t foldSize = dataset.size() / numFolds;
    size_t foldIdx = 0;

    for (size_t i = 0; i < indices.size(); ++i){
        folds[foldIdx].push_back(dataset[indices[i]]);
        if ((foldIdx < (size_t)(numFolds - 1)) && ((i + 1) % foldSize == 0)){
            ++foldIdx;
        }
    }
    
    return folds;
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

void loadDataSetWithLabels(char* dataFileName, char* namesFileName, std::vector<std::vector<double>>& dataset, std::vector<std::string>& labels){
    DataFileReader reader(dataFileName, namesFileName);
    unsigned int numRecords = reader.getRecordCount();
    unsigned int numFeatures = reader.getAttributeCount();

    dataset.clear();
    labels.clear();
    dataset.reserve(numRecords);
    labels.reserve(numRecords);

    for (unsigned int i = 0; i < numRecords; ++i){
        DataPattern record = reader.getRecord(i);
        std::vector<double> recordVector(numFeatures);
        for (unsigned int j = 0; j < numFeatures; ++j){
            recordVector[j] = record.getComponent(j);
        }
        dataset.push_back(recordVector);
        labels.push_back(record.getLabel());
    }
}

std::vector<double> loadMapFile(const std::string& mapFileName){
    std::ifstream file(mapFileName);
    std::string line;
    std::vector<double> weights;

    std::getline(file, line);

    while(std::getline(file, line)){
        if (line.empty() || line[0] == '#'){
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

TrainTestError runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator){
    int neuronSize = trainingDataset[0].size();
    auto bestWeights = differentialEvolution(trainingDataset, neuronSize, numNeurons, NP, numGenerations, CR, F, generator);

    double trainingError = quantizationError(bestWeights, trainingDataset);
    double testError = quantizationError(bestWeights, testDataset);

    TrainTestError trainTestError;
    trainTestError.trainingError = trainingError;
    trainTestError.testError = testError;

    return trainTestError;
}

CrossValError crossValidation(std::vector<std::vector<double>>& dataset, int numFolds, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator, const std::string& datasetName){
    auto folds = kFoldSplit(dataset, numFolds, generator);

    std::vector<double> trainingErrors;
    std::vector<double> testErrors;

    for (int i = 0; i < numFolds; ++i){
        auto testFold = folds[i];
        std::vector<std::vector<double>> trainingFold;

        for (int j = 0; j < numFolds; ++j){
            if (j != i){
                trainingFold.insert(trainingFold.end(), folds[j].begin(), folds[j].end());
            }
        }

        minMaxScaleTrainTest(trainingFold, testFold);

        TrainTestError trainTestError = runDE(trainingFold, testFold, numGenerations, numNeurons, NP, F, CR, generator);
        trainingErrors.push_back(trainTestError.trainingError);
        testErrors.push_back(trainTestError.testError);
    }

    double sumTrain = std::accumulate(trainingErrors.begin(), trainingErrors.end(), 0.0);
    double meanTrain = sumTrain / trainingErrors.size();

    double sq_sumTrain = std::inner_product(trainingErrors.begin(), trainingErrors.end(), trainingErrors.begin(), 0.0);
    double stdevTrain = std::sqrt(std::max(0.0, sq_sumTrain / trainingErrors.size() - meanTrain * meanTrain));

    double sumTest = std::accumulate(testErrors.begin(), testErrors.end(), 0.0);
    double meanTest = sumTest / testErrors.size();

    double sq_sumTest = std::inner_product(testErrors.begin(), testErrors.end(), testErrors.begin(), 0.0);
    double stdevTest = std::sqrt(std::max(0.0, sq_sumTest / testErrors.size() - meanTest * meanTest));

    CrossValError cve;
    cve.meanTrainingError = meanTrain;
    cve.stdTrainingError = stdevTrain;
    cve.meanTestError = meanTest;
    cve.stdTestError = stdevTest;

    return cve;
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

    int seed = 42;
    std::mt19937 gen(seed);

    auto bestWeights = differentialEvolution(dataset, neuronSize, numNeurons, NP, numGenerations, CR, F, gen);

    std::cout << std::endl << std::endl;
    std::cout << "Best Weights:" << std::endl << std::endl << vectorToString(bestWeights) << std::endl << std::endl;
    std::cout << "Best error: " << quantizationError(bestWeights, dataset) << std::endl;

    auto baselineWeights = loadMapFile(baselineWeightsFileName);
    std::cout << std::endl << std::endl;
    std::cout << "Baseline Weights:" << std::endl << std::endl << vectorToString(baselineWeights) << std::endl << std::endl;
    std::cout << "Baseline error: " << quantizationError(baselineWeights, dataset) << std::endl;

    return 0;
}
#endif
