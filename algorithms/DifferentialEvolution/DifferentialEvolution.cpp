#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include "DataFileReader.h"
#include "DataPattern.h"

std::vector<std::vector<double>> generateInitPopulation(int NP, int neuronSize, int numNeurons){
    std::random_device rd; 
    
    std::mt19937 gen(42); 
    
    std::uniform_real_distribution<double> distrib(0.0, 1.0); 
     

    std::vector<std::vector<double>> population(NP, std::vector<double>(neuronSize * numNeurons));

    for (int i = 0; i < NP; i++){
        for (int j = 0; j < neuronSize * numNeurons; j++){
            population[i][j] = distrib(gen);
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
        throw std::invalid_argument("length of vect1 does not equal length of vect 2");
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

std::vector<std::vector<double>> differentialEvolution(std::vector<std::vector<double>>& dataset, int neuronSize, int numNeurons, int NP,  int nGenerations, double CR, double F){
    auto population = generateInitPopulation(NP, neuronSize, numNeurons);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> distribVector(0, (neuronSize * numNeurons) - 1);
    std::uniform_real_distribution<double> distribDouble(0.0, 1.0);
    
    

    for (int n = 0; n < nGenerations; n++){

        std::cout << "GENERATION " << (n+1) << std::endl;
        std::cout << "Population: " << vector2dToString(population) << std::endl << std::endl;

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

            if (quantizationError(u, dataset) < quantizationError(population[i], dataset))
            {
                newPopulation.push_back(u);
            }
            else{
                newPopulation.push_back(population[i]);
            }
        }

        population = newPopulation;
    }

    return population;

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

int main(int argc, char** argv){

    if (argc < 8){
        std::cerr << "Usage: " << argv[0] << " <datasetFileName> <namesFileName> <NP> <numGenerations> <numNeurons> <CR> <F>" << std::endl;
        return 1;
    }

    auto dataset = loadDataSet(argv[1], argv[2]);
    int neuronSize = dataset[0].size();
    int NP = std::stoi(argv[3]);
    int numGenerations = std::stoi(argv[4]);
    int numNeurons = std::stoi(argv[5]);
    double CR = std::stod(argv[6]);
    double F = std::stod(argv[7]);

    auto evolvedPopulation = differentialEvolution(dataset, neuronSize, numNeurons, NP, numGenerations, CR, F);


    return 0;

}
