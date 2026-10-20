#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <unistd.h>
#include <string>
#include <cfloat>
#include <sys/stat.h>
#include <sys/types.h>
#include <random>
#include "DifferentialEvolution.h"

struct Stats {
    double mean;
    double stddev;
    double min;
    double max;
};

Stats computeStats(const std::vector<double>& values) {
    Stats s = {0.0, 0.0, 0.0, 0.0};
    if (values.empty()) return s;

    double sum = 0.0;
    s.min = values[0];
    s.max = values[0];

    for (double v : values) {
        sum += v;
        if (v < s.min) s.min = v;
        if (v > s.max) s.max = v;
    }
    s.mean = sum / values.size();

    if (values.size() > 1) {
        double varSum = 0.0;
        for (double v : values) {
            double diff = v - s.mean;
            varSum += diff * diff;
        }
        s.stddev = std::sqrt(varSum / (values.size() - 1));
    }
    return s;
}

std::string getTimestampString() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H%M%S", &tm);
    return std::string(buf);
}

std::string resolveFile(const std::string& filename) {
    if (access(filename.c_str(), F_OK) == 0) return filename;
    std::string parentPath = "../../" + filename;
    if (access(parentPath.c_str(), F_OK) == 0) return parentPath;
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

std::string resolveSOMPAK(const std::string& binary) {
    if (access(("algorithms/SOM_PAK/" + binary).c_str(), X_OK) == 0) return "algorithms/SOM_PAK/" + binary;
    if (access(("../SOM_PAK/" + binary).c_str(), X_OK) == 0) return "../SOM_PAK/" + binary;
    if (access(("../../algorithms/SOM_PAK/" + binary).c_str(), X_OK) == 0) return "../../algorithms/SOM_PAK/" + binary;
    return binary;
}

void saveSOMDataFile(const std::string& filename, const std::vector<std::vector<double>>& data) {
    std::ofstream out(filename);
    out << data[0].size() << "\n";
    for (const auto& row : data) {
        for (size_t k = 0; k < row.size(); ++k) {
            if (k > 0) out << " ";
            out << std::setprecision(15) << row[k];
        }
        out << "\n";
    }
    out.close();
}

bool loadBestSOMParams(const std::string& filename, int& width, int& height, double& lr, double& lrDecay, double& radius, double& radiusDecay) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("Best Grid:") != std::string::npos) {
            std::string gridStr = line.substr(line.find(":") + 1);
            size_t xPos = gridStr.find("x");
            if (xPos != std::string::npos) {
                width = std::stoi(gridStr.substr(0, xPos));
                height = std::stoi(gridStr.substr(xPos + 1));
            }
        } else if (line.find("Best Initial LR:") != std::string::npos) {
            lr = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best LR Decay:") != std::string::npos) {
            lrDecay = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best Initial Radius:") != std::string::npos) {
            radius = std::stod(line.substr(line.find(":") + 1));
        } else if (line.find("Best Radius Decay:") != std::string::npos) {
            radiusDecay = std::stod(line.substr(line.find(":") + 1));
        }
    }
    return true;
}

struct ClassificationMetrics {
    double accuracy;
    double macroPrecision;
    double macroRecall;
    double macroF1;
};

ClassificationMetrics evaluateClassification(
    const std::vector<double>& weights,
    int neuronSize,
    int numNeurons,
    const std::vector<std::vector<double>>& trainData,
    const std::vector<std::string>& trainLabels,
    const std::vector<std::vector<double>>& testData,
    const std::vector<std::string>& testLabels) {

    std::map<std::string, int> globalLabelCounts;
    std::string majorityGlobalLabel = "";
    int maxGlobalCount = -1;
    for (const auto& l : trainLabels) {
        globalLabelCounts[l]++;
        if (globalLabelCounts[l] > maxGlobalCount) {
            maxGlobalCount = globalLabelCounts[l];
            majorityGlobalLabel = l;
        }
    }

    std::vector<std::map<std::string, int>> neuronHits(numNeurons);
    for (size_t d = 0; d < trainData.size(); ++d) {
        const double* x = trainData[d].data();
        double minDistSq = DBL_MAX;
        int bmu = 0;
        for (int i = 0; i < numNeurons; ++i) {
            const double* w = weights.data() + i * neuronSize;
            double distSq = 0.0;
            for (int k = 0; k < neuronSize; ++k) {
                double diff = x[k] - w[k];
                distSq += diff * diff;
            }
            if (distSq < minDistSq) {
                minDistSq = distSq;
                bmu = i;
            }
        }
        neuronHits[bmu][trainLabels[d]]++;
    }

    std::vector<std::string> neuronLabels(numNeurons, majorityGlobalLabel);
    for (int i = 0; i < numNeurons; ++i) {
        int maxHits = 0;
        std::string bestLabel = majorityGlobalLabel;
        for (const auto& pair : neuronHits[i]) {
            if (pair.second > maxHits) {
                maxHits = pair.second;
                bestLabel = pair.first;
            }
        }
        neuronLabels[i] = bestLabel;
    }

    std::vector<std::string> uniqueClasses;
    for (const auto& pair : globalLabelCounts) {
        uniqueClasses.push_back(pair.first);
    }
    std::sort(uniqueClasses.begin(), uniqueClasses.end());

    std::map<std::string, int> classIndex;
    for (size_t c = 0; c < uniqueClasses.size(); ++c) {
        classIndex[uniqueClasses[c]] = c;
    }

    int K = uniqueClasses.size();
    std::vector<std::vector<int>> confusion(K, std::vector<int>(K, 0));
    int correctCount = 0;

    for (size_t d = 0; d < testData.size(); ++d) {
        const double* x = testData[d].data();
        double minDistSq = DBL_MAX;
        int bmu = 0;
        for (int i = 0; i < numNeurons; ++i) {
            const double* w = weights.data() + i * neuronSize;
            double distSq = 0.0;
            for (int k = 0; k < neuronSize; ++k) {
                double diff = x[k] - w[k];
                distSq += diff * diff;
            }
            if (distSq < minDistSq) {
                minDistSq = distSq;
                bmu = i;
            }
        }
        std::string predicted = neuronLabels[bmu];
        std::string actual = testLabels[d];
        if (predicted == actual) {
            correctCount++;
        }
        if (classIndex.count(actual) && classIndex.count(predicted)) {
            confusion[classIndex[actual]][classIndex[predicted]]++;
        }
    }

    double accuracy = testData.empty() ? 0.0 : (double)correctCount / testData.size();
    double sumPrec = 0.0;
    double sumRec = 0.0;
    double sumF1 = 0.0;

    for (int c = 0; c < K; ++c) {
        int tp = confusion[c][c];
        int actualCount = 0;
        int predCount = 0;
        for (int j = 0; j < K; ++j) {
            actualCount += confusion[c][j];
            predCount += confusion[j][c];
        }
        double prec = (predCount > 0) ? (double)tp / predCount : 0.0;
        double rec = (actualCount > 0) ? (double)tp / actualCount : 0.0;
        double f1 = (prec + rec > 1e-12) ? (2.0 * prec * rec / (prec + rec)) : 0.0;
        sumPrec += prec;
        sumRec += rec;
        sumF1 += f1;
    }

    ClassificationMetrics m;
    m.accuracy = accuracy;
    m.macroPrecision = (K > 0) ? sumPrec / K : 0.0;
    m.macroRecall = (K > 0) ? sumRec / K : 0.0;
    m.macroF1 = (K > 0) ? sumF1 / K : 0.0;
    return m;
}

int main(int argc, char** argv) {
    std::string datasetName = "iris";
    std::string dataFile = "iris.dat";
    std::string namesFile = "iris.som.names";

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
            if (argc >= 3) {
                namesFile = argv[2];
            } else {
                namesFile = datasetName + ".som.names";
            }
        }
    }

    int width = 8;
    int height = 6;
    double lr = 5.0;
    double lrDecay = 750.0;
    double radius = 4.0;
    double radiusDecay = 50.0;

    std::string outDir = resolveResultsDir(datasetName);
    std::string randomFile = outDir + "/random_params_som.txt";

    if (loadBestSOMParams(randomFile, width, height, lr, lrDecay, radius, radiusDecay)) {
        std::cout << "Loaded optimal SOM_PAK parameters from: " << randomFile << std::endl;
    } else {
        std::cout << "Using default SOM_PAK parameters." << std::endl;
    }

    int numNeurons = width * height;
    bool isHexa = true;

    std::cout << "Running Conventional SOM_PAK 30 runs with:" << std::endl;
    std::cout << "  Dataset: " << datasetName << " (" << dataFile << ", " << namesFile << ")" << std::endl;
    std::cout << "  Grid: " << width << "x" << height << " (" << numNeurons << " neurons)" << std::endl;
    std::cout << "  LR: " << lr << " | LR Decay: " << lrDecay << " | Radius: " << radius << " | Radius Decay: " << radiusDecay << std::endl;

    std::string resolvedData = resolveFile(dataFile);
    std::string resolvedNames = resolveFile(namesFile);
    std::vector<std::vector<double>> dataset;
    std::vector<std::string> labels;
    loadDataSetWithLabels((char*)resolvedData.c_str(), (char*)resolvedNames.c_str(), dataset, labels);

    int neuronSize = dataset[0].size();

    std::vector<double> trainQEs;
    std::vector<double> testQEs;
    std::vector<double> trainTEs;
    std::vector<double> testTEs;
    std::vector<double> accuracies;
    std::vector<double> precisions;
    std::vector<double> recalls;
    std::vector<double> f1s;

    std::string timestamp = getTimestampString();
    std::string timestampedFile = outDir + "/runs_30_som_" + timestamp + ".txt";
    std::string activeFile = outDir + "/runs_30_som.txt";

    std::ofstream outTime(timestampedFile);
    std::ofstream outActive(activeFile);

    std::string randinitBin = resolveSOMPAK("randinit");
    std::string vsomBin = resolveSOMPAK("vsom");
    pid_t myPid = getpid();

    for (unsigned long run = 1; run <= 30; ++run) {
        sleep(1);
        unsigned long seed = (unsigned long)std::time(nullptr);

        std::vector<size_t> indices(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::mt19937 shuffleRng(seed);
        std::shuffle(indices.begin(), indices.end(), shuffleRng);

        size_t trainSize = dataset.size() * 4 / 5;
        std::vector<std::vector<double>> trainSet;
        std::vector<std::string> trainLabels;
        std::vector<std::vector<double>> testSet;
        std::vector<std::string> testLabels;

        trainSet.reserve(trainSize);
        trainLabels.reserve(trainSize);
        testSet.reserve(dataset.size() - trainSize);
        testLabels.reserve(dataset.size() - trainSize);

        for (size_t i = 0; i < dataset.size(); ++i) {
            if (i < trainSize) {
                trainSet.push_back(dataset[indices[i]]);
                trainLabels.push_back(labels[indices[i]]);
            } else {
                testSet.push_back(dataset[indices[i]]);
                testLabels.push_back(labels[indices[i]]);
            }
        }

        minMaxScaleTrainTest(trainSet, testSet);

        std::string tmpPrefix = "/tmp/sompak_run_" + std::to_string(myPid) + "_r" + std::to_string(run);
        std::string trainFile = tmpPrefix + "_train.som";
        std::string initMap = tmpPrefix + "_init.map";
        std::string outMap = tmpPrefix + "_out.map";

        saveSOMDataFile(trainFile, trainSet);

        std::string cmdInit = randinitBin + " -din " + trainFile + " -cout " + initMap + " -topol hexa -neigh gaussian -xdim " + std::to_string(width) + " -ydim " + std::to_string(height) + " -rand " + std::to_string(seed) + " > /dev/null 2>&1";
        std::system(cmdInit.c_str());

        std::string cmdTrain = vsomBin + " -din " + trainFile + " -vin " + trainFile + " -cin " + initMap + " -cout " + outMap + " -lrate " + std::to_string(lr) + " -lratedecay " + std::to_string(lrDecay) + " -radius " + std::to_string(radius) + " -radiusdecay " + std::to_string(radiusDecay) + " -rand " + std::to_string(seed) + " > /dev/null 2>&1";
        std::system(cmdTrain.c_str());

        auto weights = loadMapFile(outMap);

        std::remove(trainFile.c_str());
        std::remove(initMap.c_str());
        std::remove(outMap.c_str());

        double trainQE = quantizationError(weights, trainSet);
        double testQE = quantizationError(weights, testSet);
        double trainTE = topographicError(weights, width, height, isHexa, trainSet);
        double testTE = topographicError(weights, width, height, isHexa, testSet);

        auto cls = evaluateClassification(weights, neuronSize, numNeurons, trainSet, trainLabels, testSet, testLabels);

        trainQEs.push_back(trainQE);
        testQEs.push_back(testQE);
        trainTEs.push_back(trainTE);
        testTEs.push_back(testTE);
        accuracies.push_back(cls.accuracy);
        precisions.push_back(cls.macroPrecision);
        recalls.push_back(cls.macroRecall);
        f1s.push_back(cls.macroF1);

        std::cout << "Run: " << std::setw(2) << run 
                  << " | Seed: " << seed 
                  << " | Train QE: " << std::fixed << std::setprecision(6) << trainQE 
                  << " | Test QE: " << testQE 
                  << " | Train TE: " << trainTE 
                  << " | Test TE: " << testTE 
                  << " | Acc: " << std::setprecision(4) << cls.accuracy 
                  << " | Prec: " << cls.macroPrecision 
                  << " | Rec: " << cls.macroRecall 
                  << " | F1: " << cls.macroF1 << std::endl;

        std::string record = "Run: " + std::to_string(run) +
                             " | Seed: " + std::to_string(seed) +
                             " | Train QE: " + std::to_string(trainQE) +
                             " | Test QE: " + std::to_string(testQE) +
                             " | Train TE: " + std::to_string(trainTE) +
                             " | Test TE: " + std::to_string(testTE) +
                             " | Accuracy: " + std::to_string(cls.accuracy) +
                             " | Precision: " + std::to_string(cls.macroPrecision) +
                             " | Recall: " + std::to_string(cls.macroRecall) +
                             " | F1: " + std::to_string(cls.macroF1) + "\n";

        outActive << record;
        outActive.flush();
        outTime << record;
        outTime.flush();
    }

    auto sTrainQE = computeStats(trainQEs);
    auto sTestQE = computeStats(testQEs);
    auto sTrainTE = computeStats(trainTEs);
    auto sTestTE = computeStats(testTEs);
    auto sAcc = computeStats(accuracies);
    auto sPrec = computeStats(precisions);
    auto sRec = computeStats(recalls);
    auto sF1 = computeStats(f1s);

    std::stringstream summary;
    summary << "\n=============================================\n";
    summary << "SOM_PAK 30-Run Summary for: " << datasetName << "\n";
    summary << "Train QE  : Mean = " << sTrainQE.mean << " | StdDev = " << sTrainQE.stddev << " | Min = " << sTrainQE.min << " | Max = " << sTrainQE.max << "\n";
    summary << "Test QE   : Mean = " << sTestQE.mean << " | StdDev = " << sTestQE.stddev << " | Min = " << sTestQE.min << " | Max = " << sTestQE.max << "\n";
    summary << "Train TE  : Mean = " << sTrainTE.mean << " | StdDev = " << sTrainTE.stddev << " | Min = " << sTrainTE.min << " | Max = " << sTrainTE.max << "\n";
    summary << "Test TE   : Mean = " << sTestTE.mean << " | StdDev = " << sTestTE.stddev << " | Min = " << sTestTE.min << " | Max = " << sTestTE.max << "\n";
    summary << "Accuracy  : Mean = " << sAcc.mean << " | StdDev = " << sAcc.stddev << " | Min = " << sAcc.min << " | Max = " << sAcc.max << "\n";
    summary << "Precision : Mean = " << sPrec.mean << " | StdDev = " << sPrec.stddev << " | Min = " << sPrec.min << " | Max = " << sPrec.max << "\n";
    summary << "Recall    : Mean = " << sRec.mean << " | StdDev = " << sRec.stddev << " | Min = " << sRec.min << " | Max = " << sRec.max << "\n";
    summary << "F1-Score  : Mean = " << sF1.mean << " | StdDev = " << sF1.stddev << " | Min = " << sF1.min << " | Max = " << sF1.max << "\n";
    summary << "=============================================\n";

    std::cout << summary.str();
    outActive << summary.str();
    outTime << summary.str();

    outActive.close();
    outTime.close();

    return 0;
}
