#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <unistd.h>
#include <sys/stat.h>

struct RunData {
    int run;
    unsigned long seed;
    double trainQE;
    double testQE;
    double trainTE;
    double testTE;
    double accuracy;
    double precision;
    double recall;
    double f1;
};

std::string resolveResultsDir(const std::string& datasetName) {
    if (access("../../iris.dat", F_OK) == 0) {
        return "../../results/" + datasetName;
    }
    return "results/" + datasetName;
}

std::vector<RunData> parseRunFile(const std::string& filename) {
    std::vector<RunData> records;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open: " << filename << std::endl;
        return records;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("Run:", 0) == 0) {
            RunData rd = {0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            std::stringstream ss(line);
            std::string part;
            while (std::getline(ss, part, '|')) {
                if (part.find("Run:") != std::string::npos) {
                    rd.run = std::stoi(part.substr(part.find(":") + 1));
                } else if (part.find("Seed:") != std::string::npos) {
                    rd.seed = std::stoul(part.substr(part.find(":") + 1));
                } else if (part.find("Train QE:") != std::string::npos) {
                    rd.trainQE = std::stod(part.substr(part.find(":") + 1));
                } else if (part.find("Test QE:") != std::string::npos) {
                    rd.testQE = std::stod(part.substr(part.find(":") + 1));
                } else if (part.find("Train TE:") != std::string::npos) {
                    rd.trainTE = std::stod(part.substr(part.find(":") + 1));
                } else if (part.find("Test TE:") != std::string::npos) {
                    rd.testTE = std::stod(part.substr(part.find(":") + 1));
                } else if (part.find("Accuracy:") != std::string::npos) {
                    rd.accuracy = std::stod(part.substr(part.find(":") + 1));
                } else if (part.find("Precision:") != std::string::npos) {
                    rd.precision = std::stod(part.substr(part.find(":") + 1));
                } else if (part.find("Recall:") != std::string::npos) {
                    rd.recall = std::stod(part.substr(part.find(":") + 1));
                } else if (part.find("F1:") != std::string::npos) {
                    rd.f1 = std::stod(part.substr(part.find(":") + 1));
                }
            }
            records.push_back(rd);
        }
    }
    return records;
}

bool processDataset(const std::string& datasetName) {
    std::string dir = resolveResultsDir(datasetName);
    std::string deFile = dir + "/runs_30_de.txt";
    std::string somFile = dir + "/runs_30_som.txt";
    std::string csvFile = dir + "/runs_30_paired.csv";

    auto deRecords = parseRunFile(deFile);
    auto somRecords = parseRunFile(somFile);

    if (deRecords.empty() || somRecords.empty()) {
        std::cerr << "Missing or empty run records for " << datasetName << std::endl;
        return false;
    }

    if (deRecords.size() != somRecords.size()) {
        std::cerr << "Mismatched record counts for " << datasetName 
                  << ": DE=" << deRecords.size() << ", SOM=" << somRecords.size() << std::endl;
        return false;
    }

    std::ofstream csv(csvFile);
    if (!csv.is_open()) {
        std::cerr << "Could not create " << csvFile << std::endl;
        return false;
    }

    csv << "run,de_seed,som_seed,de_train_qe,de_test_qe,de_train_te,de_test_te,de_accuracy,de_precision,de_recall,de_f1,som_train_qe,som_test_qe,som_train_te,som_test_te,som_accuracy,som_precision,som_recall,som_f1\n";
    for (size_t i = 0; i < deRecords.size(); ++i) {
        const auto& d = deRecords[i];
        const auto& s = somRecords[i];
        csv << d.run << "," << d.seed << "," << s.seed << ","
            << std::fixed << std::setprecision(6)
            << d.trainQE << "," << d.testQE << "," << d.trainTE << "," << d.testTE << ","
            << d.accuracy << "," << d.precision << "," << d.recall << "," << d.f1 << ","
            << s.trainQE << "," << s.testQE << "," << s.trainTE << "," << s.testTE << ","
            << s.accuracy << "," << s.precision << "," << s.recall << "," << s.f1 << "\n";
    }
    csv.close();

    std::cout << "Successfully generated paired CSV for " << datasetName << " (" << deRecords.size() << " runs): " << csvFile << std::endl;
    return true;
}

int main(int argc, char** argv) {
    if (argc >= 2) {
        std::string arg = argv[1];
        if (arg == "all") {
            processDataset("iris");
            processDataset("glass");
            processDataset("ionosphere");
        } else {
            processDataset(arg);
        }
    } else {
        processDataset("iris");
        processDataset("glass");
        processDataset("ionosphere");
    }
    return 0;
}
