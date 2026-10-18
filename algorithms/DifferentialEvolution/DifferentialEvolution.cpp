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
        distributions.emplace_back(minVal[k], maxVal[k]);
    }

    int totalNumWeights = neuronSize * numNeurons;
    std::vector<std::vector<double>> population(NP, std::vector<double>(totalNumWeights));
    
    for (int n = 0; n < NP; ++n){
        for (int i = 0; i < totalNumWeights; ++i){
            population[n][i] = distributions[i %  neuronSize](generator);
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

std::vector<double> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP,  int nGenerations, double CR, double F, std::mt19937& gen){

    auto population = generateInitPopulation(NP, neuronSize, numNeurons, dataset, gen);

    
    std::uniform_int_distribution<int> distribVector(0, (neuronSize * numNeurons) - 1);
    std::uniform_real_distribution<double> distribDouble(0.0, 1.0);
    
    double lowestQuantizationError = DBL_MAX;
    std::vector<double> bestWeights;

    for (int n = 0; n < nGenerations; n++){

        std::cout << "GENERATION " << (n+1) << std::endl;
        std::cout << "best error " << lowestQuantizationError << std::endl;
        // std::cout << "Population: " << vector2dToString(population) << std::endl << std::endl;

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
            int randIdx = distribVector(gen);

            for(int j = 0; j < population[i].size(); j++)
            {
                double randJ = distribDouble(gen);

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

    return bestWeights;

}

struct TrainTestSplit {
    std::vector<std::vector<double>> train;
    std::vector<std::vector<double>> test;
};

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


    int foldSize = dataset.size() / numFolds; //integer division, remainder goes to last fold
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

    //skip first (header) line
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

TrainTestError runDE(std::vector<std::vector<double>>& trainingDataset, std::vector<std::vector<double>>& testDataset, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator){

    auto bestWeights = differentialEvolution(trainingDataset, neuronSize, numNeurons, NP, numGenerations, CR, F, generator);

    double trainingError = quantizationError(bestWeights, trainingDataset);
    double testError = quantizationError(bestWeights, testDataset);

    std::cout << "Best training error: " << quantizationError(bestWeights, trainingDataset) << std::endl;
    std::cout << "Best test error: " << quantizationError(bestWeights, testDataset) << std::endl;

    TrainTestError trainTestError;

    trainTestError.trainingError = trainingError;
    trainTestError.testError = testError;

    return trainTestError;

}


CrossValError crossValidation(std::vector<std::vector<double>>& dataset,  int numFolds, int numGenerations, int numNeurons, int NP, double F, double CR, std::mt19937& generator){

    auto folds = kFoldSplit(dataset, numFolds, generator);

    std::vector<double> trainingErrors(0);
    std::vector<double> testErrors(0);

    for (int i = 0; i < numFolds; ++i){
        auto testFold = folds[i]

         std::vector<std::vector<double>> trainingFold(0)

         for (int j = 0; j < numFolds; ++j){
            if (j != i){
                trainingFold.insert(trainingFold.end(), folds[j].begin(), folds[j].end());
            }
         }

         TrainTestError trainTestError = runDE(trainFold, testFold, numGenerations, numNeurons, NP, F, CR, generator);

         trainingErrors.push_back(trainTestError.trainingError);
         testErrors.push_back(trainTestError.testError);
         
    }


    double sumTrain = std::accumulate(trainingErrors.begin(), trainingErrors.end(), 0.0);
    double meanTrain = sumTrain / trainingErrors.size();

    double sq_sumTrain = std::inner_product(trainingErrors.begin(), trainingErrors.end(), trainingErrors.begin(), 0.0);
    double stdevTrain = std::sqrt(sq_sumTrain / trainingErrors.size() - meanTrain * meanTrain);


    double sumTest = std::accumulate(testErrors.begin(), testErrors.end(), 0.0);
    double meanTest = sumTest / testErrors.size();

    double sq_sumTest = std::inner_product(testErrors.begin(), testErrors.end(), testErrors.begin(), 0.0);
    double stdevTest = std::sqrt(sq_sumTest / testErrors.size() - meanTest * meanTest);

    CrossValError cve;

    cve.meanTrainingError = meanTrain;
    cve.stdTrainingError = stddevTrain;
    cve.meanTestError = meanTest;
    cve.stdTestError = stdevTest;

    return cve;


}





int main(int argc, char** argv){

    if (argc < 9){
        std::cerr << "Usage: " << argv[0] << " <datasetFileName> <namesFileName> <NP> <numGenerations> <numNeurons> <CR> <F> <baselineWeightsFileName>" << std::endl;
        return 1;
    }

    //best hyperparameters found so far:
    //./DifferentialEvolution iris.dat iris.som.names 100 2000 48 0.01 0.5 iris_trained.map

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

    std::cout << "Best Weights:" << std::endl << std::endl << vectorToString(bestWeights) <<std::endl << std::endl;
    std::cout << "Best error: " << quantizationError(bestWeights, dataset) << std::endl;

    auto baselineWeights = loadMapFile(baselineWeightsFileName);

    std::cout << std::endl << std::endl;

    std::cout << "Baseline Weights:" << std::endl << std::endl << vectorToString(baselineWeights) <<std::endl << std::endl;
    std::cout << "Baseline error: " << quantizationError(baselineWeights, dataset) << std::endl;

    return 0;

}
