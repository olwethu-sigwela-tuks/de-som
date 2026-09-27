#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

struct RunRecord {
    int run;
    unsigned long seed;
    double trainQE;
    double testQE;
    double trainTE;
    double testTE;
};

std::string getTimestampString() {
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H%M%S", &tm);
    return std::string(buf);
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

std::vector<RunRecord> parseResultFile(const std::string& filename) {
    std::vector<RunRecord> records;
    std::ifstream file(filename);
    if (!file.is_open()) return records;

    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind("Run:", 0) != 0) continue;

        RunRecord rec;
        std::stringstream ss(line);
        char bar;
        std::string temp;

        ss >> temp >> rec.run >> bar;
        ss >> temp >> rec.seed >> bar;
        ss >> temp >> temp >> rec.trainQE >> bar;
        ss >> temp >> temp >> rec.testQE >> bar;
        ss >> temp >> temp >> rec.trainTE >> bar;
        ss >> temp >> temp >> rec.testTE;

        records.push_back(rec);
    }
    return records;
}

void writeSingleCSV(const std::string& outFile, const std::vector<RunRecord>& records) {
    std::ofstream out(outFile);
    if (!out.is_open()) return;

    out << "run,seed,train_qe,test_qe,train_te,test_te\n";
    for (const auto& r : records) {
        out << r.run << "," << r.seed << "," 
            << r.trainQE << "," << r.testQE << "," 
            << r.trainTE << "," << r.testTE << "\n";
    }
    out.close();
}

void writePairedCSV(const std::string& outFile, const std::vector<RunRecord>& de, const std::vector<RunRecord>& som) {
    std::ofstream out(outFile);
    if (!out.is_open()) return;

    out << "run,seed,de_train_qe,de_test_qe,de_train_te,de_test_te,som_train_qe,som_test_qe,som_train_te,som_test_te\n";
    size_t n = std::min(de.size(), som.size());
    for (size_t i = 0; i < n; ++i) {
        out << de[i].run << "," << de[i].seed << ","
            << de[i].trainQE << "," << de[i].testQE << ","
            << de[i].trainTE << "," << de[i].testTE << ","
            << som[i].trainQE << "," << som[i].testQE << ","
            << som[i].trainTE << "," << som[i].testTE << "\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    std::string datasetName = "iris";
    std::string deFile = "";
    std::string somFile = "";

    if (argc == 1) {
        datasetName = "iris";
        std::string outDir = resolveResultsDir(datasetName);
        deFile = outDir + "/runs_30_de.txt";
        somFile = outDir + "/runs_30_som.txt";
    } else if (argc == 2) {
        datasetName = argv[1];
        std::string outDir = resolveResultsDir(datasetName);
        deFile = outDir + "/runs_30_de.txt";
        somFile = outDir + "/runs_30_som.txt";
    } else if (argc >= 3) {
        deFile = argv[1];
        somFile = argv[2];
        datasetName = "custom";
    }

    auto deRecords = parseResultFile(deFile);
    auto somRecords = parseResultFile(somFile);

    if (deRecords.empty() && somRecords.empty()) {
        std::cerr << "Warning: Could not read records from " << deFile << " or " << somFile << std::endl;
    }

    if (datasetName != "custom") {
        std::string outDir = resolveResultsDir(datasetName);
        std::string timestamp = getTimestampString();

        writeSingleCSV(outDir + "/runs_30_de.csv", deRecords);
        writeSingleCSV(outDir + "/runs_30_som.csv", somRecords);
        writePairedCSV(outDir + "/runs_30_paired.csv", deRecords, somRecords);

        writeSingleCSV(outDir + "/runs_30_de_" + timestamp + ".csv", deRecords);
        writeSingleCSV(outDir + "/runs_30_som_" + timestamp + ".csv", somRecords);
        writePairedCSV(outDir + "/runs_30_paired_" + timestamp + ".csv", deRecords, somRecords);

        std::cout << "Successfully exported CSV files for " << datasetName << ":" << std::endl;
        std::cout << "  - " << outDir << "/runs_30_de.csv (" << deRecords.size() << " records)" << std::endl;
        std::cout << "  - " << outDir << "/runs_30_som.csv (" << somRecords.size() << " records)" << std::endl;
        std::cout << "  - " << outDir << "/runs_30_paired.csv (" << std::min(deRecords.size(), somRecords.size()) << " paired records for Wilcoxon test)" << std::endl;
    } else {
        writeSingleCSV("results_30_runs_de.csv", deRecords);
        writeSingleCSV("results_30_runs_som.csv", somRecords);
        writePairedCSV("results_30_runs_paired.csv", deRecords, somRecords);

        std::cout << "Successfully exported CSV files:" << std::endl;
        std::cout << "  - results_30_runs_de.csv (" << deRecords.size() << " records)" << std::endl;
        std::cout << "  - results_30_runs_som.csv (" << somRecords.size() << " records)" << std::endl;
        std::cout << "  - results_30_runs_paired.csv (" << std::min(deRecords.size(), somRecords.size()) << " paired records for Wilcoxon test)" << std::endl;
    }

    return 0;
}
