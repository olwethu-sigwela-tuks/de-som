#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cfloat>
#include <cmath>
#include <random>
#include <string>
#include <ctime>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "DifferentialEvolution.h"

std::string getTimestampString() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H%M%S", &tm);
    return std::string(buf);
}

std::string resolveFile(const std::string& filename) {
    if (access(filename.c_str(), F_OK) == 0) return filename;
    std::string parent = "../../" + filename;
    if (access(parent.c_str(), F_OK) == 0) return parent;
    std::string inside = "somlib/" + filename;
    if (access(inside.c_str(), F_OK) == 0) return inside;
    return filename;
}

std::string resolveResultsDir(const std::string& datasetName) {
    if (access("../../iris.dat", F_OK) == 0) {
        mkdir("../../results", 0777);
        std::string dir = "../../results/" + datasetName;
        mkdir(dir.c_str(), 0777);
        return dir;
    }
    mkdir("results", 0777);
    std::string dir = "results/" + datasetName;
    mkdir(dir.c_str(), 0777);
    return dir;
}

int countCompletedTrials(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return 0;
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (line.rfind("Trial:", 0) == 0) {
            count++;
        }
    }
    return count;
}

int main(int argc, char** argv) {
    std::string datasetName = "iris";
    std::string dataFile = "iris.dat";
    std::string namesFile = "iris.som.names";
    int totalTrials = 30;
    unsigned long masterSeed = (unsigned long)std::time(nullptr);

    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "iris" || arg1 == "glass" || arg1 == "ionosphere") {
            datasetName = arg1;
            dataFile = arg1 + ".dat";
            namesFile = arg1 + ".som.names";
        } else {
            dataFile = arg1;
            datasetName = arg1;
            size_t slash = datasetName.find_last_of("/\\");
            if (slash != std::string::npos) datasetName = datasetName.substr(slash + 1);
            size_t dot = datasetName.find_last_of(".");
            if (dot != std::string::npos) datasetName = datasetName.substr(0, dot);
            if (argc >= 3) namesFile = argv[2];
            else namesFile = datasetName + ".som.names";
        }
    }

    if (argc >= 3) {
        try {
            int t = std::stoi(argv[2]);
            if (t > 0) totalTrials = t;
        } catch (...) {}
    }

    if (argc >= 4) {
        try {
            masterSeed = std::stoul(argv[3]);
        } catch (...) {}
    }

    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);
    auto dataset = loadDataSet((char*)resolvedData.c_str(), (char*)resolvedNames.c_str());

    std::string outDir = resolveResultsDir(datasetName);
    std::string activeFile = outDir + "/random_params_de.txt";
    std::string timestamp = getTimestampString();
    std::string timeFile = outDir + "/random_params_de_" + timestamp + ".txt";

    int completedTrials = countCompletedTrials(activeFile);
    if (completedTrials >= totalTrials) {
        std::cout << "Already evaluated " << completedTrials << " trials in " << activeFile << ". Exiting." << std::endl;
        return 0;
    }

    std::ofstream outActive(activeFile, std::ios::app);
    std::ofstream outTime(timeFile, std::ios::app);

    if (completedTrials == 0) {
        outActive << "# Random Search Hyperparameter Tuning for Differential Evolution (" << datasetName << ")\n";
        outTime << "# Random Search Hyperparameter Tuning for Differential Evolution (" << datasetName << ")\n";
    }

    std::mt19937 searchRng(masterSeed);
    std::uniform_int_distribution<int> distNP(0, 1);
    std::uniform_int_distribution<int> distGen(0, 2);
    std::uniform_real_distribution<double> distF(0.10, 0.40);
    std::uniform_real_distribution<double> distLogCR(std::log10(0.001), std::log10(0.050));

    int npChoices[] = {20, 50};
    int genChoices[] = {2000, 3000, 4000};
    int numNeurons = 48;

    double bestTestError = DBL_MAX;
    int bestNP = 20;
    int bestGen = 2000;
    double bestCR = 0.01;
    double bestF = 0.2;

    if (completedTrials > 0) {
        std::ifstream existing(activeFile);
        std::string line;
        while (std::getline(existing, line)) {
            if (line.rfind("Trial:", 0) == 0) {
                int trialNP = 20, trialGen = 2000;
                double trialCR = 0.0, trialF = 0.0, trialTestErr = DBL_MAX;
                std::stringstream ss(line);
                std::string part;
                while (std::getline(ss, part, '|')) {
                    if (part.find("NP:") != std::string::npos) trialNP = std::stoi(part.substr(part.find(":") + 1));
                    else if (part.find("Gen:") != std::string::npos) trialGen = std::stoi(part.substr(part.find(":") + 1));
                    else if (part.find("CR:") != std::string::npos) trialCR = std::stod(part.substr(part.find(":") + 1));
                    else if (part.find("F:") != std::string::npos && part.find("Fold") == std::string::npos) trialF = std::stod(part.substr(part.find(":") + 1));
                    else if (part.find("5-Fold Test Error:") != std::string::npos) trialTestErr = std::stod(part.substr(part.find(":") + 1));
                }
                if (trialTestErr < bestTestError) {
                    bestTestError = trialTestErr;
                    bestNP = trialNP;
                    bestGen = trialGen;
                    bestCR = trialCR;
                    bestF = trialF;
                }
            }
        }
    }

    for (int t = 0; t < completedTrials; ++t) {
        distNP(searchRng);
        distGen(searchRng);
        distF(searchRng);
        distLogCR(searchRng);
    }

    std::cout << "Starting Random Search DE for " << datasetName << " (Total Trials: " << totalTrials << ", Resuming from: " << (completedTrials + 1) << ")" << std::endl;

    for (int trial = completedTrials + 1; trial <= totalTrials; ++trial) {
        int NP = npChoices[distNP(searchRng)];
        int numGenerations = genChoices[distGen(searchRng)];
        double F = distF(searchRng);
        double CR = std::pow(10.0, distLogCR(searchRng));

        unsigned int evalSeed = 42;
        std::mt19937 cvRng(evalSeed);
        CrossValError cve = crossValidation(dataset, 5, numGenerations, numNeurons, NP, F, CR, cvRng, datasetName);

        std::cout << "Trial: " << std::setw(2) << trial 
                  << " | NP: " << NP 
                  << " | Gen: " << numGenerations 
                  << " | Neurons: " << numNeurons 
                  << " | CR: " << std::fixed << std::setprecision(5) << CR 
                  << " | F: " << std::fixed << std::setprecision(3) << F 
                  << " | Train Error: " << std::setprecision(6) << cve.meanTrainingError 
                  << " | 5-Fold Test Error: " << cve.meanTestError << std::endl;

        outActive << "Trial: " << std::setw(2) << trial 
                  << " | NP: " << NP 
                  << " | Gen: " << numGenerations 
                  << " | Neurons: " << numNeurons 
                  << " | CR: " << std::fixed << std::setprecision(5) << CR 
                  << " | F: " << std::fixed << std::setprecision(3) << F 
                  << " | Train Error: " << std::setprecision(6) << cve.meanTrainingError 
                  << " | 5-Fold Test Error: " << cve.meanTestError << std::endl;
        outActive.flush();

        outTime << "Trial: " << std::setw(2) << trial 
                << " | NP: " << NP 
                << " | Gen: " << numGenerations 
                << " | Neurons: " << numNeurons 
                << " | CR: " << std::fixed << std::setprecision(5) << CR 
                << " | F: " << std::fixed << std::setprecision(3) << F 
                << " | Train Error: " << std::setprecision(6) << cve.meanTrainingError 
                << " | 5-Fold Test Error: " << cve.meanTestError << std::endl;
        outTime.flush();

        if (cve.meanTestError < bestTestError) {
            bestTestError = cve.meanTestError;
            bestNP = NP;
            bestGen = numGenerations;
            bestCR = CR;
            bestF = F;
        }
    }

    std::cout << "\n=== Best Hyperparameters from Random Search ===" << std::endl;
    std::cout << "Best NP: " << bestNP << std::endl;
    std::cout << "Best Generations: " << bestGen << std::endl;
    std::cout << "Best Neurons: " << numNeurons << std::endl;
    std::cout << "Best CR: " << bestCR << std::endl;
    std::cout << "Best F: " << bestF << std::endl;
    std::cout << "Best 5-Fold Test Quantization Error: " << bestTestError << std::endl;

    outActive << "\n=== Best Hyperparameters from Random Search ===\n";
    outActive << "Best NP: " << bestNP << "\n";
    outActive << "Best Generations: " << bestGen << "\n";
    outActive << "Best Neurons: " << numNeurons << "\n";
    outActive << "Best CR: " << bestCR << "\n";
    outActive << "Best F: " << bestF << "\n";
    outActive << "Best 5-Fold Test Quantization Error: " << bestTestError << "\n";
    outActive.close();

    outTime << "\n=== Best Hyperparameters from Random Search ===\n";
    outTime << "Best NP: " << bestNP << "\n";
    outTime << "Best Generations: " << bestGen << "\n";
    outTime << "Best Neurons: " << numNeurons << "\n";
    outTime << "Best CR: " << bestCR << "\n";
    outTime << "Best F: " << bestF << "\n";
    outTime << "Best 5-Fold Test Quantization Error: " << bestTestError << "\n";
    outTime.close();

    return 0;
}
