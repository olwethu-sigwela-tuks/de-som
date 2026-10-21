#include "DifferentialEvolution.h"
#include <vector>
#include <gsl/gsl_rng.h>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <fstream>

struct Parameters{
    int NP;
    int numGenerations;
    int populationSize;
    int numNeurons;
    double CR;
    double F;
};

void selectParametersDE(std::vector<double>& NPs, std::vector<int>& numGenerations, std::vector<int>& populationSizes, std::vector<double>& CRs, std::vector<double> Fs, std::mt19937& generator){

    //best hyperparameters found so far:
    //./DifferentialEvolution iris.dat iris.som.names 100 2000 48 0.01 0.5 iris_trained.map

    auto dataset = loadDataSet("iris.dat", "iris.som.names");

    std::vector<Parameters> parameters;
    std::vector<double> crossValidationScores;

    int numFolds = 5;
    int numNeurons = 48;

    for (int i = 0; i < NPs.size(); ++i){
        for (int j = 0; j < numGenerations.size(); ++j){
            for (int k = 0; k < populationSizes.size(); ++k){
                for (int l = 0; l < CRs.size(); ++l){
                    for (int m = 0; m < Fs.size(); ++m){
                        double NP = NPs[i];
                        double numGeneration = numGenerations[j];
                        double populationSize = populationSizes[k];
                        double CR = CRs[l];
                        double F = Fs[m];

                        CrossValError cve = crossValidation(dataset, numFolds, numGeneration, numNeurons, NP, F, CR, generator);
                        
                        Parameters parameters;
                        parameters.NP = NP;
                        parameters.numGenerations = numGeneration;
                        parameters.populationSize = populationSize;
                        parameters.numNeurons = numNeurons;
                        parameters.CR = CR;
                        parameters.F = F;

                        std::cout << "{\nNP=" << parameters.NP 
                        << "\n, numGenerations=" << parameters.numGenerations
                        << "\n, populationSize=" << parameters.populationSize
                        << "\n, numNeurons=" << parameters.numNeurons
                        << "\n, CR=" << parameters.CR
                        << "\n, F=" << parameters.F << "\n}" << std::endl;
                        std::cout << numFolds << "-fold cross validation mean training quantisation error: " << cve.meanTrainingError << std::endl;
                        std::cout << numFolds << "-fold cross validation mean test quantisation error: " << cve.meanTestError << std::endl;

                    }
                }
            }
        }
    }

}

// double sampleUniform(gsl_rng* rng, double minVal, double maxVal){
//     return minVal + gsl_rng_uniform(rng) * (maxVal - minVal);
// }

// double sampleLogUniform(gsl_rng* rng, double MinVal, double maxVal){
//     double logMin = std::log10(minVal);
//     double logMax = std::log10(maxVal);

//     double logSample = logMin + gsl_rng_uniform(rng) * (logMax - logMin);
//     return std::pow(10.0, logSample);
// }

// int sampleChoice(gsl_rng* rng, const std::vector<int>& list){
//     size_t idx = gsl_rng_uniform_int(rng, options.size());
//     return options[idx];
// }
void randomSearchDE(std::vector<std::vector<double>>& dataset, std::mt19937& rng, int numTrials, std::vector<int> NPChoices, std::vector<int> generationChoices, std::vector<double> FChoices, std::vector<double> CRChoices){
    std::uniform_int_distribution<int> distNP(0, NPChoices.size() - 1);
    std::uniform_int_distribution<int> distGen(0, generationChoices.size() - 1);
    std::uniform_real_distribution<double> distF(FChoices[0], FChoices[FChoices.size() - 1]);
    std::uniform_real_distribution<double> distLogCR(std::log10(CRChoices[0]), std::log10(CRChoices[CRChoices.size() - 1]));

    double bestQE = DBL_MAX;
    int numFolds = 5;
    int numNeurons = 48;


    int bestNP = 0;
    int bestGen = 0;
    double bestF = 0.0;
    double bestCR = 0.0;
    std::ofstream logFile("random_trials_de.txt", std::ios_base::app);

    

    //not sure if this is the right place to put these or if they should be in some kind of for loop
    for (int n = 0; n < numTrials; ++n){
        int NP = NPChoices[distNP(rng)];
        // int genChoices[] = {2000, 3000, 4000};
        int numGen = generationChoices[distGen(rng)];

        double F = distF(rng);
        double CR = std::pow(10.0, distLogCR(rng));

        CrossValError cve = crossValidation(dataset, numFolds, numGen, numNeurons, NP, F, CR, rng);

        std::cout << "{\nNP=" << NP 
                        << "\n, numGen=" << numGen
                        << "\n, numNeurons=" << numNeurons
                        << "\n, CR=" << CR
                        << "\n, F=" << F << "\n}" << std::endl;
                        std::cout << numFolds << "-fold cross validation mean training quantisation error: " << cve.meanTrainingError << std::endl;
                        std::cout << numFolds << "-fold cross validation mean test quantisation error: " << cve.meanTestError << std::endl; 
        
        logFile << (n + 1) << "\t" << NP << "\t" << numGen << "\t" << F << "\t" << CR << "\t" << cve.meanTestError << std::endl;

        if (cve.meanTestError < bestQE) {
            bestQE = cve.meanTestError;
            bestNP = NP;
            bestGen = numGen;
            bestF = F;
            bestCR = CR;
            std::ofstream bestFile("random_params_de.txt");
            bestFile << "Best Grid: 8x6\n";
            bestFile << "Best Neurons: " << numNeurons << "\n";
            bestFile << "Best NP: " << bestNP << "\n";
            bestFile << "Best Gen: " << bestGen << "\n";
            bestFile << "Best F: " << bestF << "\n";
            bestFile << "Best CR: " << bestCR << "\n";
            bestFile << "Best 5-Fold Test QE: " << bestQE << "\n";
            bestFile.close();
            std::cout << "---> New best QE found and saved to file!" << std::endl;
        }
}


}


int main(int argc, char** argv){
    if (argc < 4){
        std::cerr << "Usage: " << argv[0] << " <dataFile> <namesFile> <numTrials>" << std::endl;
        return 1;
    }

    auto dataset = loadDataSet(argv[1], argv[2]);
    int numTrials = std::stoi(argv[3]);

    std::mt19937 rng(42);
    std::vector<int> NPChoices = {20, 30, 50, 100};
    std::vector<int> generationChoices = {1000, 2000, 3000, 4000};
    std::vector<double> FChoices = {0.1, 0.9};
    std::vector<double> CRChoices = {0.001, 0.9};

    randomSearchDE(dataset, rng, numTrials, NPChoices, generationChoices, FChoices, CRChoices);
    return 0;
}