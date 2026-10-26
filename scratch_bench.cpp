
#include "algorithms/DifferentialEvolution/DifferentialEvolution.h"
#include <chrono>
#include <iostream>

int main() {
    auto iris = loadDataSet("iris.dat", "iris.som.names");
    auto iono = loadDataSet("ionosphere.dat", "ionosphere.som.names");
    
    gsl_rng* rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, 42);
    
    // Test Iris 50 generations with NP=50
    auto t0 = std::chrono::high_resolution_clock::now();
    differentialEvolution(iris, 4, 48, 50, 50, 0.3, 0.5, rng);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_iris = std::chrono::duration<double>(t1 - t0).count();
    
    // Test Ionosphere 50 generations with NP=50
    t0 = std::chrono::high_resolution_clock::now();
    differentialEvolution(iono, 34, 48, 50, 50, 0.3, 0.5, rng);
    t1 = std::chrono::high_resolution_clock::now();
    double dt_iono = std::chrono::duration<double>(t1 - t0).count();
    
    std::cout << "Iris 50 gen: " << dt_iris << " s (" << dt_iris / 50.0 << " s/gen)" << std::endl;
    std::cout << "Ionosphere 50 gen: " << dt_iono << " s (" << dt_iono / 50.0 << " s/gen)" << std::endl;
    std::cout << "Ratio: " << dt_iono / dt_iris << "x" << std::endl;
    
    gsl_rng_free(rng);
    return 0;
}
