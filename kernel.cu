
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand_kernel.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <numeric>
#include <iomanip>

#include "kernel.cuh"


__constant__ Config globalConfig;


__global__ void Init_states(curandState* states, long long seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    curand_init(seed, idx, 1000, &states[idx]);
}

__global__ void InitBitstring(curandState* states, bool* b) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    int startBit = threadIdx.x * globalConfig.bits;
    for (int i = startBit; i < startBit + globalConfig.bits; i++) 
    {
        b[i] = curand_uniform(&states[idx]) > 0.5f;
    }
}

__device__ void Convert(bool* bits, double* values) 
{
    for (int j = 0; j < globalConfig.d; j++) {
        unsigned long long dec = 0;
        for (int i = 0; i < globalConfig.bitsPerDim; i++)
        {
            dec = (dec << 1) | bits[j * globalConfig.bitsPerDim + i];
            
        }
        values[j] = globalConfig.a + dec * (globalConfig.b - globalConfig.a) / ((1ull << globalConfig.bitsPerDim) - 1);
     
        //printf("\n");
        if (abs(values[j]) > 5.12)
        {   
            printf("%llu %llu\n", dec, ((1ull << globalConfig.bitsPerDim) - 1));
            printf("%llu\n", dec);
        
            for (int i = 0; i < globalConfig.bitsPerDim; i++)
            {
                printf("%d", bits[j * globalConfig.bitsPerDim + i]);
            }
            printf("\n\n\n\n\n");
        }
    }
}
__global__ void GenRealValues(bool* bits, double* values) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    Convert(bits + idx * globalConfig.bits, values + idx * globalConfig.d);
}

__device__ double Rastrigin(double* v, int dimensions) {

    double res = 10 * dimensions;
    for (int i = 0; i < dimensions; i++) {
        res += v[i] * v[i] - 10 * cos(2 * M_PI * v[i]);
    }
    return res;
}

__device__ double Michalewicz(double* v, int dimensions) {
    double res = 0;
    for (int i = 0; i < dimensions; i++) {
        res += sin(v[i]) * pow(sin(((i + 1) * v[i] * v[i]) / M_PI), 20);
    }
    return -res;
}

//reminder to check if this si actually dejong
__device__ double Dejong(double* v, int dimensions) {
    double res = 0;
    for (int i = 0; i < dimensions; i++) {
        res += v[i] * v[i];
    }
    return res;
}

__device__ double Sumsq(double* v, int dimensions) {
    double res = 0;
    for (int i = 0; i < dimensions; i++) {
        res += (i + 1) * v[i] * v[i];
    }
    return res;
}

__device__ double Eval(double* values) 
{
    switch (globalConfig.func) 
    {
    case function::Rastrigin:
        return Rastrigin(values, globalConfig.d);
        break;
    case function::Michalewicz:
        return Michalewicz(values, globalConfig.d);
        break;
    
    case function::Schwefel:
        return Sumsq(values, globalConfig.d);
        break;
    
    case function::Dejong:
        return Dejong(values, globalConfig.d);
        break;
    }
}

__global__ void EvalFitness(double* values, double* candidates) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    candidates[idx] = Eval(idx * globalConfig.d + values);
}


__global__  void HillClimbFirstImpr(bool* bitstr, double* values, double* candidates) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    int startBit = threadIdx.x * globalConfig.bits;
    double bestValue = candidates[idx];
    double currentValue = bestValue;

    for (int i = 0; i < globalConfig.bits; i++) 
    {
        int bitflip = startBit + i;
        bitstr[bitflip] = !bitstr[bitflip];
        Convert(bitstr + startBit, values + idx * globalConfig.d);
        currentValue = Eval(values + idx * globalConfig.d);

        if (currentValue < bestValue) 
        {
            bestValue = currentValue;
            //if(idx==0) printf("%f\n", bestValue);
            i = 0;
        }
        else { bitstr[bitflip] = !bitstr[bitflip]; }
    }

    candidates[idx] = bestValue;
}

__global__  void HillClimbBestImpr(bool* bitstr, double* values, double* candidates) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    int startBit = threadIdx.x * globalConfig.bits;
}
__global__  void HillClimbWorstImpr(bool* bitstr, double* values, double* candidates) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    int startBit = threadIdx.x * globalConfig.bits;
}
__global__  void Annealing(bool* bitstr, double* values, double* candidates, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= globalConfig.it) return;
    int startBit = threadIdx.x * globalConfig.bits;
}


// CUDA-compatible Improve function
//__device__ bitstring Improve(bitstring* neigh, int neigh_size, bitstring vc, bool impr) {
//    double init = Eval(vc);
//    bitstring result = vc;
//
//    switch (impr) {
//    case 0:
//        for (int i = 0; i < neigh_size; i++) {
//            double candidate = Eval(neigh[i]);
//            if (init > candidate) {
//                result = neigh[i];
//                return result;
//            }
//        }
//        break;
//    case 1:
//        result = neigh[0];
//        for (int i = 1; i < neigh_size; i++) {
//            double candidate = Eval(neigh[i]);
//            if (init > candidate) {
//                init = candidate;
//                result = neigh[i];
//            }
//        }
//        break;
//    }
//
//    return result;
//}

// Kernel to perform optimization
//__global__ void local_vc(double* result, double* T, curandState* states) {
//    int idx = threadIdx.x + blockIdx.x * blockDim.x;
//    bool reachedLocal = false;
//    bitstring vc = Gen_num();  // Generate a random bitstring
//    double initcandidate = Eval(vc);
//    int counter = 0;
//    int changeCount = 0;
//    int maxAttempts = 100;
//    int neigh_size = 5;  // Example neighborhood size
//    bitstring neigh[5];  // Example neighborhood array
//
//    // Generate neighborhood
//    for (int i = 0; i < neigh_size; i++) {
//        neigh[i] = vc;  // Populate with current vc, but normally this should be varied
//        neigh[i] ^= (1 << (i % Nbit()));  // Example: flipping some bits to create a neighborhood
//    }
//
//    do {
//        counter++;
//        bitstring vn = Improve(neigh, neigh_size, vc, 1);
//        double candidateNeigh = Eval(vn);
//
//        if (candidateNeigh < initcandidate) {
//            vc = vn;
//            initcandidate = candidateNeigh;
//            changeCount++;
//        }
//        else {
//            // Simulated annealing-like condition with random number generation on the device
//            float random_vail = curand_uniform(&states[idx]);
//            if (random_val < exp(-fabs(candidateNeigh - initcandidate) / *T)) {
//                vc = vn;
//                changeCount++;
//                initcandidate = candidateNeigh;
//            }
//        }
//
//    } while (changeCount < maxAttempts && counter < 10 * maxAttempts);
//
//    *result = initcandidate;
//}

std::vector<double> launch(const Config& config) {
    
    bool* bitstr;
    double* candidates;
    double* realValues;
    curandState* states;
    std::vector<double> result(config.it);
   
    // Allocate device memory
    cudaMalloc(&bitstr, sizeof(bool) * config.bits);
    cudaMalloc(&candidates, sizeof(double) * config.it);
    cudaMalloc(&states, sizeof(curandState) * config.it);  
    cudaMalloc(&realValues, sizeof(double) * config.it * config.d);
    cudaMemcpyToSymbol(globalConfig, &config, sizeof(Config));
    

    // Launch kernel
    Init_states <<< config.blocks, config.threads >>> (states, std::random_device{}());
    InitBitstring <<< config.blocks, config.threads >>> (states, bitstr);
    GenRealValues <<< config.blocks, config.threads >>> (bitstr, realValues);
    EvalFitness <<< config.blocks, config.threads >>> (realValues, candidates);

  
    switch (globalConfig.strat)
    {
       case improvment::Firstimprov:
           HillClimbFirstImpr <<< config.blocks, config.threads >>> (bitstr, realValues, candidates);
          break;
       case improvment::Bestimprov:
           HillClimbBestImpr <<< config.blocks, config.threads >>> (bitstr, realValues, candidates);
           break;
       case improvment::Worstimprov:
           HillClimbWorstImpr <<< config.blocks, config.threads >>> (bitstr, realValues, candidates);
           break;
       case improvment::Annealing:
           Annealing <<< config.blocks, config.threads >>> (bitstr, realValues, candidates, states);
           break;
    default:
        break;
    }
 

    // Copy result back to host
    cudaMemcpy(result.data(), candidates, sizeof(double) * config.it, cudaMemcpyDeviceToHost);
   
 
    // Clean up device memory
    cudaFree(bitstr);
    cudaFree(candidates);
    cudaFree(states);
    cudaFree(realValues);

    return result;
}