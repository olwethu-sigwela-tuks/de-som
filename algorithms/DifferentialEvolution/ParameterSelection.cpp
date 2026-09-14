#include "DifferentialEvolution.h"


void selectParameters(){

    //best hyperparameters found so far:
    //./DifferentialEvolution iris.dat iris.som.names 100 2000 48 0.01 0.5 iris_trained.map

    auto dataset = loadDataSet("iris.dat", "iris.som.names");
    for (int NP = 5, NP <= 200; ++NP){
        for (int numGenerations = 1000; numGenerations <= 3000; ++numGenerations ){
            for (int numNeurons = 20; numNeurons <= 60; ++numNeurons){
                for (double CR = 0.01; CR <= 1; +-CR += 0.1){
                    for (double F = 0.1; F <= 1; F += 0.1){

                    }
                }
            }
        }
    }

}