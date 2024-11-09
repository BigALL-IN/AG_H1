#pragma once
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
enum class function { Rastrigin, Michalewicz, Dejong, Schwefel };
enum class improvment { Firstimprov, Bestimprov, Worstimprov, Annealing };

struct Config {

    double a;
    double b;
    int p;
    int d;
    int it;
    int bits;
    int bitsPerDim;
    int temp;
    int threads;
    int blocks;
    function func;
    improvment strat;
};

std::vector<double> launch(const Config& configs);
