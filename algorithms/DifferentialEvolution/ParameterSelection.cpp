#include "DifferentialEvolution.h"
#include <vector>
#include <gsl/gsl_rng.h>
#include <cmath>
#include <iostream>
#include <cfloat>

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
                        << "\n, F=" << parameters.F << "\n}";
                        std::cout << numFolds << "-fold cross validation mean training quantisation error: " << cve.meanTrainingError;
                        std::cout << numFolds << "-fold cross validation mean test quantisation error: " << cve.meanTestError;

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
                        << "\n, F=" << F << "\n}";
                        std::cout << numFolds << "-fold cross validation mean training quantisation error: " << cve.meanTrainingError;
                        std::cout << numFolds << "-fold cross validation mean test quantisation error: " << cve.meanTestError;

        if (cve.meanTestError < bestQE) {
             bestQE = cve.meanTestError;
            // store bestNP, bestGen, bestF, bestCR
            std::cout << "---> New best QE found!"
        }
}





    }
    
}
