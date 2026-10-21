#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <deRunsFile> <somRunsFile> <outCSV>" << std::endl;
        return 1;
    }

    std::ifstream deIn(argv[1]);
    std::ifstream somIn(argv[2]);
    std::ofstream csvOut(argv[3]);

    csvOut << "Run,Seed,Method,Train_QE,Test_QE,Train_TE,Test_TE,Time_s\n";

    std::string line;
    std::getline(deIn, line);
    while (std::getline(deIn, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int run;
        unsigned long seed;
        double trQE, teQE, trTE, teTE, timeS;
        if (iss >> run >> seed >> trQE >> teQE >> trTE >> teTE >> timeS) {
            csvOut << run << "," << seed << ",DE," << trQE << "," << teQE << "," << trTE << "," << teTE << "," << timeS << "\n";
        }
    }

    std::getline(somIn, line);
    while (std::getline(somIn, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int run;
        unsigned long seed;
        double trQE, teQE, trTE, teTE, timeS;
        if (iss >> run >> seed >> trQE >> teQE >> trTE >> teTE >> timeS) {
            csvOut << run << "," << seed << ",SOM," << trQE << "," << teQE << "," << trTE << "," << teTE << "," << timeS << "\n";
        }
    }

    return 0;
}