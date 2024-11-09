#include "kernel.cuh"
#include <cmath>
#include <iostream>
#include <numeric>
#include <iomanip>

int main() 
{
    Config config;
    config.func = function::Rastrigin;
    config.strat = improvment::Firstimprov;
    switch (config.func)
    {
        case function::Rastrigin: 
        {
            config.a = -5.12;
            config.b = 5.12;
            break;
        }
        case function::Michalewicz: 
        {
            config.a = 0;
            config.b = M_PI;
            break;
        }
        case function::Dejong: 
        {
            config.a = -5.12;
            config.b = 5.12;
            break;
        }
        case function::Schwefel:
        {
            config.a = -500;
            config.b = 500;
            break;
        }

        default:
            break;
    }
    config.it = 6;
    config.p = 5;
    config.d = 5;
    int segments = (config.b - config.a) * pow(10, config.p);
    config.bitsPerDim = static_cast<int>(std::ceil(log2(segments)));
    config.bits = config.bitsPerDim * config.d;
    config.threads = 64;
    config.blocks = (config.it + config.threads - 1) / config.threads;

    std::vector<double> result = launch(config);
    

    std::cout << "\n\n===================Final Results================\n\n\n" << std::flush;
    std::cout << "\n\n" << std::flush;;
       
    std::cout << std::fixed << std::setprecision(config.p);
    std::cout << "Best result: " << *std::min_element(result.begin(), result.end()) << '\n' << std::flush;
   // std::cout << "Best Runtime: " << *std::min_element(result.begin(), result.end()) << "\n\n" << std::flush;
    std::cout << "Average result: " << std::accumulate(result.begin(), result.end(), 0.0) / (double)result.size() << '\n' << std::flush;
   // std::cout << "Average Runtime: " << std::accumulate(sampleRuntimeDurations.begin(), sampleRuntimeDurations.end(), 0.0) / (double)sampleRuntimeDurations.size() << "\n\n" << std::flush;
    std::cout << "Worst result: " << *std::max_element(result.begin(), result.end()) << '\n' << std::flush;
   // std::cout << "Worst Runtime: " << *std::max_element(sampleRuntimeDurations.begin(), sampleRuntimeDurations.end()) << '\n' << std::flush;
    return 0;
}