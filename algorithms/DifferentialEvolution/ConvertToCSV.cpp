#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

struct RunRecord {
    int run;
    int seed;
    double trainQE;
    double testQE;
    double trainTE;
    double testTE;
};

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
    std::string deFile = "results_30_runs_de.txt";
    std::string somFile = "results_30_runs_som.txt";

    if (argc >= 2) deFile = argv[1];
    if (argc >= 3) somFile = argv[2];

    auto deRecords = parseResultFile(deFile);
    auto somRecords = parseResultFile(somFile);

    writeSingleCSV("results_30_runs_de.csv", deRecords);
    writeSingleCSV("results_30_runs_som.csv", somRecords);
    writePairedCSV("results_30_runs_paired.csv", deRecords, somRecords);

    std::cout << "Successfully exported CSV files:" << std::endl;
    std::cout << "  - results_30_runs_de.csv (" << deRecords.size() << " records)" << std::endl;
    std::cout << "  - results_30_runs_som.csv (" << somRecords.size() << " records)" << std::endl;
    std::cout << "  - results_30_runs_paired.csv (" << std::min(deRecords.size(), somRecords.size()) << " paired records for R Wilcoxon test)" << std::endl;

    return 0;
}
