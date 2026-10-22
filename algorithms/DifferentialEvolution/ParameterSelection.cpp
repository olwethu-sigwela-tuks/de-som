#include "DifferentialEvolution.h"
#include <vector>
#include <gsl/gsl_rng.h>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

struct Parameters{
    int NP;
    int numGenerations;
    int populationSize;
    int numNeurons;
    double CR;
    double F;
};

void selectParametersDE(std::vector<double>& NPs, std::vector<int>& numGenerations, std::vector<int>& populationSizes, std::vector<double>& CRs, std::vector<double> Fs, std::mt19937& generator){

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

void randomSearchDE(std::vector<std::vector<double>>& dataset, std::mt19937& rng, int numTrials, std::vector<int> NPChoices, std::vector<int> generationChoices, std::vector<double> FChoices, std::vector<double> CRChoices, const std::string& datasetName){
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

    mkdir("results", 0777);
    std::string outDir = "results/" + datasetName;
    mkdir(outDir.c_str(), 0777);

    std::ofstream logFile(outDir + "/random_trials_de.txt", std::ios_base::app);

    for (int n = 0; n < numTrials; ++n){
        int NP = NPChoices[distNP(rng)];
        int numGen = generationChoices[distGen(rng)];

        double F = distF(rng);
        double CR = std::pow(10.0, distLogCR(rng));

        std::cout << "Candidate " << std::setw(2) << (n + 1) << "/" << numTrials
                  << " | NP: " << std::setw(3) << NP
                  << " | Gen: " << std::setw(4) << numGen
                  << " | F: " << std::fixed << std::setprecision(4) << F
                  << " | CR: " << std::fixed << std::setprecision(6) << CR
                  << " | Evaluating 5 folds..." << std::flush;

        CrossValError cve = crossValidation(dataset, numFolds, numGen, numNeurons, NP, F, CR, rng);

        std::cout << " Done! Test QE: " << std::fixed << std::setprecision(6) << cve.meanTestError
                  << " (Train QE: " << std::fixed << std::setprecision(6) << cve.meanTrainingError << ")"
                  << std::endl;

        logFile << (n + 1) << "\t" << NP << "\t" << numGen << "\t" << F << "\t" << CR << "\t" << cve.meanTestError << std::endl;

        if (cve.meanTestError < bestQE) {
            bestQE = cve.meanTestError;
            bestNP = NP;
            bestGen = numGen;
            bestF = F;
            bestCR = CR;
            std::ofstream bestFile(outDir + "/random_params_de.txt");
            bestFile << "Best Grid: 8x6\n";
            bestFile << "Best Neurons: " << numNeurons << "\n";
            bestFile << "Best NP: " << bestNP << "\n";
            bestFile << "Best Gen: " << bestGen << "\n";
            bestFile << "Best F: " << bestF << "\n";
            bestFile << "Best CR: " << bestCR << "\n";
            bestFile << "Best 5-Fold Test QE: " << bestQE << "\n";
            bestFile.close();

            std::ofstream bestLocal("random_params_de.txt");
            bestLocal << "Best Grid: 8x6\n";
            bestLocal << "Best Neurons: " << numNeurons << "\n";
            bestLocal << "Best NP: " << bestNP << "\n";
            bestLocal << "Best Gen: " << bestGen << "\n";
            bestLocal << "Best F: " << bestF << "\n";
            bestLocal << "Best CR: " << bestCR << "\n";
            bestLocal << "Best 5-Fold Test QE: " << bestQE << "\n";
            bestLocal.close();

            std::cout << "  --> New Best DE-SOM Parameter Found!" << std::endl;
        }
    }

    std::cout << "\nDE-SOM Best Parameters for " << datasetName << ":" << std::endl;
    std::cout << "  Grid: 8x6 (48 neurons)" << std::endl;
    std::cout << "  NP: " << bestNP << std::endl;
    std::cout << "  Generations: " << bestGen << std::endl;
    std::cout << "  F: " << bestF << std::endl;
    std::cout << "  CR: " << bestCR << std::endl;
    std::cout << "  5-Fold Test QE: " << bestQE << std::endl;
    std::cout << "Saved optimal parameters to: " << outDir + "/random_params_de.txt" << std::endl;
}


int main(int argc, char** argv){
    if (argc < 4){
        std::cerr << "Usage: " << argv[0] << " <dataFile> <namesFile> <numTrials> [datasetName]" << std::endl;
        return 1;
    }

    auto dataset = loadDataSet(argv[1], argv[2]);
    int numTrials = std::stoi(argv[3]);

    std::string datasetName = argv[1];
    if (argc >= 5) {
        datasetName = argv[4];
    } else {
        size_t slash = datasetName.find_last_of("/\\");
        if (slash != std::string::npos) datasetName = datasetName.substr(slash + 1);
        size_t dot = datasetName.find_last_of(".");
        if (dot != std::string::npos) datasetName = datasetName.substr(0, dot);
    }

    std::mt19937 rng(42);
    std::vector<int> NPChoices = {20, 30, 50, 100};
    std::vector<int> generationChoices = {1000, 1500, 2000};
    std::vector<double> FChoices = {0.1, 0.9};
    std::vector<double> CRChoices = {0.001, 0.9};

    randomSearchDE(dataset, rng, numTrials, NPChoices, generationChoices, FChoices, CRChoices, datasetName);
    return 0;
}