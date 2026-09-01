#include <random>
#include <vector>
#include <algorithm>
#include <iostream>

std::vector<std::vector<double>> generateInitPopulation(int NP, int vectorSize){
    std::random_device rd; 
    
    std::mt19937 gen(42); 
    
    std::uniform_real_distribution<double> distrib(0.0, 1.0); 
     

    std::vector<std::vector<double>> population(NP, std::vector<double>(vectorSize));

    for (int i = 0; i < NP; i++){
        for (int j = 0; j < vectorSize; j++){
            population[i][j] = distrib(gen);
        }
    }

    return population;
}

std::vector<int> getRandomIndices(std::mt19937& generator, int currIndex, std::vector<std::vector<double>> population){

    std::vector<int> indices;
    indices.reserve(3);
    std::uniform_int_distribution<int> distrib(0, NP - 1);
    

    for (int i = 0; i < 3; i++){
        int randomIndex = currIndex;

        while(randomIndex == currIndex){
            int choice = distrib(generator);
            if (std::ranges::contains(indices, choice)){
                continue;
            }
            randomIndex = choice;
            indices.push_back(randomIndex);
        }
    }

    return indices;
}


std::vector<std::vector<double>> differentialEvolution(int NP,  int nGenerations, int vectorSize, double CR, double F){
    auto population = generateInitPopulation(NP, vectorSize);

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> distribInt(0, NP - 1);
    std::uniform_real_distribution<double> distribDouble(0.0, 1.0);
    
    

    for (int n = 0; n < nGenerations; n++){

        std::vector<std::vector<double>> newPopulation;

        for (int i = 0; i < population.size(); i++){
            auto randomIndices = getRandomIndices(gen, n, population);
            int r1 = randomIndices[0];
            int r2 = randomIndices[1];
            int r3 = randomIndices[2];

            std::vector<double> v;

            for (int x = 0; x < population[0].size(); x++)
            {
                vComponent = population[r1][x] + F * (population[r2][x] - population[r3][x]);
                v.push_back(vComponent);

            }

            std::vector<double> u;
            int randIdx = distribInt(gen);

            for(int j = 0; j < population[i].size(); j++)
            {
                double randJ = distribDouble(gen);

                if ((rand_j <= CR) || (j == randIdx)){
                    u.push_back(v[j]);
                }
                else if ((rand_j > CR) && (j != randIdx)){
                    u.push_back(population[i][j]);
                }
            }

            if (error(u) < error(population[i]))
            {
                newPopulation.push_back(u);
            }
            else{
                newPopulation.push_back(population[i])
            }
        }

        newPopulation = population
    }

}
