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

std::vector<std::vector<double>> generateInitPopulation(int NP, int neuronSize, int numNeurons, std::vector<std::vector<double>>& dataset, gsl_rng* generator){
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
            double diff = maxVal[k] - minVal[k];
            population[n][i] = minVal[k] + diff * gsl_rng_uniform(generator);
        }
    }

    return population;
}

std::vector<std::vector<double>> generateInitPopulation(int NP, int neuronSize, int numNeurons, std::vector<std::vector<double>>& dataset, std::mt19937& generator){
    gsl_rng* g = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(g, 42);
    auto res = generateInitPopulation(NP, neuronSize, numNeurons, dataset, g);
    gsl_rng_free(g);
    return res;
}

std::vector<int> getRandomIndices(gsl_rng* generator, int currIndex, int NP){
    std::vector<int> indices;
    indices.reserve(3);

    while(indices.size() < 3){
        int randomIndex = gsl_rng_uniform_int(generator, NP);
        if ((randomIndex != currIndex) && (std::find(indices.begin(), indices.end(), randomIndex) == indices.end())){
            indices.push_back(randomIndex);
        }
    }

    return indices;
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
            if (trainSet[i][j] < minVal[j]){
                minVal[j] = trainSet[i][j];
            }
            if (trainSet[i][j] > maxVal[j]){
                maxVal[j] = trainSet[i][j];
            }
        }
    }

    for (size_t i = 0; i < nTrain; ++i){
        for (size_t j = 0; j < numFeatures; ++j){
            if (minVal[j] == maxVal[j]){
                trainSet[i][j] = 0.0;
            } else {
                trainSet[i][j] = (trainSet[i][j] - minVal[j]) / (maxVal[j] - minVal[j]);
            }
        }
    }

    for (size_t i = 0; i < testSet.size(); ++i){
        for (size_t j = 0; j < numFeatures; ++j){
            if (minVal[j] == maxVal[j]){
                testSet[i][j] = 0.0;
            } else {
                testSet[i][j] = (testSet[i][j] - minVal[j]) / (maxVal[j] - minVal[j]);
            }
        }
    }
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

void saveMapFile(const std::string& mapFileName, const std::vector<double>& weights, int width, int height, bool isHexa, int neuronSize){
    std::ofstream file(mapFileName);
    file << neuronSize << (isHexa ? " hexa " : " rect ") << width << " " << height << " gaussian\n";
    for (size_t i = 0; i < weights.size(); i += neuronSize){
        for (size_t k = 0; k < neuronSize; ++k){
            if (k > 0) file << " ";
            file << std::setprecision(15) << weights[i + k];
        }
        file << "\n";
    }
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

std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP,  int nGenerations, double CR, double F, gsl_rng* gen){

    auto population = generateInitPopulation(NP, neuronSize, numNeurons, dataset, gen);

    int totalNumWeights = neuronSize * numNeurons;
    double lowestQuantizationError = DBL_MAX;
    std::vector<double> bestWeights;

    for (int n = 0; n < nGenerations; n++){

        std::vector<std::vector<double>> newPopulation;

        for (int i = 0; i < population.size(); i++){
            auto randomIndices = getRandomIndices(gen, i, NP);
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
            int randIdx = gsl_rng_uniform_int(gen, totalNumWeights);

            for(int j = 0; j < population[i].size(); j++)
            {
                double randJ = gsl_rng_uniform(gen);

                if ((randJ <= CR) || (j == randIdx)){
                    u.push_back(v[j]);
                }
                else {
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

    return bestWeights;

}

std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP, int nGenerations, double CR, double F, std::mt19937& /*gen*/){
    gsl_rng* g = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(g, 42);
    auto res = differentialEvolution(dataset, neuronSize, numNeurons, NP, nGenerations, CR, F, g);
    gsl_rng_free(g);
    return res;
}

TrainTestSplit trainTestSplit(std::vector<std::vector<double>>& dataset, double testRatio, std::mt19937& g){


    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    TrainTestSplit split;


    int trainDatasetSize = std::round((1 - testRatio) * dataset.size());

    for (int i = 0; i < indices.size(); ++i){

        if (i < trainDatasetSize){
            split.train.push_back(dataset[indices[i]]);
        }
        else{
            split.test.push_back(dataset[indices[i]]);
        }
    }
    
    return split; 

}

std::vector<std::vector<std::vector<double>>> kFoldSplit(std::vector<std::vector<double>>& dataset, int numFolds, std::mt19937& g){

    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);



    std::vector<std::vector<std::vector<double>>> folds;

    
    for (int n = 0; n < numFolds; ++n){
        std::vector<std::vector<double>> fold(0);
        folds.push_back(fold);
    }


    int foldSize = dataset.size() / numFolds;
    int foldIdx = 0;


    for (int i = 0; i < indices.size(); ++i){

        folds[foldIdx].push_back(dataset[indices[i]]);

        if ((foldIdx < numFolds - 1) && ((i + 1) % foldSize == 0)){
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

TrainTestError runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR, gsl_rng* generator){

    int neuronSize = trainingDataset[0].size();

    auto bestWeights = differentialEvolution(trainingDataset, neuronSize, numNeurons, NP, numGenerations, CR, F, generator);

    double trainingError = quantizationError(bestWeights, trainingDataset);
    double testError = quantizationError(bestWeights, testDataset);

    TrainTestError trainTestError;

    trainTestError.trainingError = trainingError;
    trainTestError.testError = testError;

    return trainTestError;

}

TrainTestError runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& /*generator*/){
    gsl_rng* g = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(g, 42);
    auto res = runDE(trainingDataset, testDataset, numGenerations, numNeurons, NP, F, CR, g);
    gsl_rng_free(g);
    return res;
}


CrossValError crossValidation(std::vector<std::vector<double>>& dataset,  int numFolds, int numGenerations, int numNeurons, int NP, double F, double CR, gsl_rng* generator){

    std::mt19937 g(42);
    auto folds = kFoldSplit(dataset, numFolds, g);

    std::vector<double> trainingErrors(0);
    std::vector<double> testErrors(0);

    for (int i = 0; i < numFolds; ++i){
        auto testFold = folds[i];

         std::vector<std::vector<double>> trainingFold(0);

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

CrossValError crossValidation(std::vector<std::vector<double>>& dataset,  int numFolds, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& /*generator*/){
    gsl_rng* g = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(g, 42);
    auto res = crossValidation(dataset, numFolds, numGenerations, numNeurons, NP, F, CR, g);
    gsl_rng_free(g);
    return res;
}

FoldPartition createFoldPartition(size_t datasetSize, int numFolds, unsigned int seed){
    FoldPartition part;
    part.numFolds = numFolds;
    part.foldIndices.resize(numFolds);

    std::vector<size_t> indices(datasetSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 g(seed);
    std::shuffle(indices.begin(), indices.end(), g);

    for (int f = 0; f < numFolds; ++f){
        size_t startIdx = (f * datasetSize) / numFolds;
        size_t endIdx = ((f + 1) * datasetSize) / numFolds;
        for (size_t i = startIdx; i < endIdx; ++i){
            part.foldIndices[f].push_back(indices[i]);
        }
    }
    return part;
}

void saveFoldPartition(const std::string& filename, const FoldPartition& partition){
    std::ofstream out(filename);
    if (!out.is_open()) return;
    out << partition.numFolds << "\n";
    for (int f = 0; f < partition.numFolds; ++f){
        for (size_t i = 0; i < partition.foldIndices[f].size(); ++i){
            if (i > 0) out << " ";
            out << partition.foldIndices[f][i];
        }
        out << "\n";
    }
    out.close();
}

FoldPartition loadOrCreateFoldPartition(const std::string& filename, size_t datasetSize, int numFolds, unsigned int seed){
    std::ifstream in(filename);
    if (in.is_open()){
        FoldPartition part;
        int folds = 0;
        if (in >> folds && folds == numFolds){
            std::string line;
            std::getline(in, line);
            part.numFolds = numFolds;
            part.foldIndices.resize(numFolds);
            bool valid = true;
            size_t totalCount = 0;
            for (int f = 0; f < numFolds; ++f){
                if (!std::getline(in, line)){
                    valid = false;
                    break;
                }
                std::istringstream iss(line);
                size_t idx;
                while (iss >> idx){
                    part.foldIndices[f].push_back(idx);
                    totalCount++;
                }
            }
            if (valid && totalCount == datasetSize){
                return part;
            }
        }
        in.close();
    }
    FoldPartition part = createFoldPartition(datasetSize, numFolds, seed);
    saveFoldPartition(filename, part);
    return part;
}

TrainTestSplit getTrainTestFromPartition(const std::vector<std::vector<double>>& dataset, const FoldPartition& partition, int testFoldIdx){
    TrainTestSplit split;
    if (testFoldIdx < 0 || testFoldIdx >= partition.numFolds){
        return split;
    }
    split.test.reserve(partition.foldIndices[testFoldIdx].size());
    for (size_t idx : partition.foldIndices[testFoldIdx]){
        split.test.push_back(dataset[idx]);
    }
    size_t trainCount = dataset.size() - split.test.size();
    split.train.reserve(trainCount);
    for (int f = 0; f < partition.numFolds; ++f){
        if (f == testFoldIdx) continue;
        for (size_t idx : partition.foldIndices[f]){
            split.train.push_back(dataset[idx]);
        }
    }
    return split;
}

CrossValError crossValidationWithPartition(const std::vector<std::vector<double>>& dataset, const FoldPartition& partition, int numGenerations, int numNeurons, int NP, double F, double CR, gsl_rng* generator){
    std::vector<double> trainingErrors;
    std::vector<double> testErrors;

    for (int i = 0; i < partition.numFolds; ++i){
        auto split = getTrainTestFromPartition(dataset, partition, i);
        minMaxScaleTrainTest(split.train, split.test);
        TrainTestError trainTestError = runDE(split.train, split.test, numGenerations, numNeurons, NP, F, CR, generator);

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

CrossValError crossValidationWithPartition(const std::vector<std::vector<double>>& dataset, const FoldPartition& partition, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& /*generator*/){
    gsl_rng* g = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(g, 42);
    auto res = crossValidationWithPartition(dataset, partition, numGenerations, numNeurons, NP, F, CR, g);
    gsl_rng_free(g);
    return res;
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
    gsl_rng* gen = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(gen, seed);

    auto bestWeights = differentialEvolution(dataset, neuronSize, numNeurons, NP, numGenerations, CR, F, gen);
    gsl_rng_free(gen);

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