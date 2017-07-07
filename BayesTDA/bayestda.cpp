//
//  bayestda.cpp
//
//  Created by Patrick Medina on 6/6/17.
//  Copyright © 2017 Patrick Medina. All rights reserved.
//

#include <iostream>
#include <vector>

#include "bayestda.h"
#include "linalg.h"

using namespace std;

// buffer value to prevent NAN in certain calculations 
double EPS = 1e-50;

// constructor for bayesRegressionDM class
btda::bayesRegressionDM::bayesRegressionDM(int numScales, int numWavelets, int numCovariates, int numSamples, vector< vector< float > > & waveletCoefficients, vector< vector< float > > & covariateMatrix, vector <int> & use, bool groupScaling)
{
    // extract needed parameters
    this -> numScales = numScales;
    this -> numWavelets = numWavelets;
    this -> numCovariates = numCovariates;
    this -> numSamples = numSamples;
    this -> groupScaling = groupScaling;
    numPi = groupScaling == false ? 4 : 4 * (numScales + 1);
    
    // Bayesian parameters
    priorK = 1.0;
    m = 1;
    
    // get data matrices
    this -> waveletCoefficients = waveletCoefficients;
    this -> covariateMatrix = covariateMatrix;
    this -> use = use;
    
    // allocate space for the inference storage variables
    piEstimates.resize(numPi, 0.0);
    posteriorProbabilities.resize(4, vector <double>(numCovariates, 0.0));
    lnLikelihoodRatios.resize(numCovariates, 0.0);
}

void btda::bayesRegressionDM::computePosteriorProbabilities(vector <int> & waveletCoordinates, int waveletArrayNumberCoordinates)
{
    // Declare constants
    double lnBF1, lnBF2, lnBF3; // store log Bayes factors
    double denom;               // stores weighted sum of Bayes Factors
//    double EPS = 1e-30;
    
    // Allocate memory for arrays
    vector<double> ssWavelets(3, 0.0);
    vector<double> ssCovariates(3, 0.0);
    vector<double> ssWaveletCovariate(4, 0.0);
    
    // move along the covariates.
    for (int cov = 0; cov < numCovariates; cov++)
    {
        
        // move along the scales
        for (int s = 0; s < waveletArrayNumberCoordinates - 1; s++)
        {
            // move along the wavelet coefficients at the given scale
            for (int wav = waveletCoordinates[s]; wav < waveletCoordinates[s + 1]; wav++)
            {
                // Compute Sum of Squares
                computeSumOfSquares(ssWavelets, ssCovariates, ssWaveletCovariate, cov, wav);
                
                // compute log Bayes Factors
                lnBF1 = computeLnBayesFactor12(ssWavelets, ssCovariates, ssWaveletCovariate, 0);
                
                lnBF2 = computeLnBayesFactor12(ssWavelets, ssCovariates, ssWaveletCovariate, 1);
                
                lnBF3 = computeLnBayesFactor3(ssWavelets, ssCovariates, ssWaveletCovariate);
                
                // compute interim posterior proabilities
                denom = piEstimates[s * 4] + piEstimates[s * 4 + 1] * exp(lnBF1) + piEstimates[s * 4 + 2] * exp(lnBF2) + piEstimates[s * 4 + 3] * exp(lnBF3);
                
                // TODO: Include appropriate code to handle when a probability is - or near - zero.
                posteriorProbabilities[0][cov] += log(piEstimates[s * 4] + EPS) - log(denom);
                posteriorProbabilities[1][cov] += log(piEstimates[s * 4 + 1] + piEstimates[s * 4 + 1] * exp(lnBF1) + EPS) - log(denom);
                posteriorProbabilities[2][cov] += log(piEstimates[s * 4 + 2] + piEstimates[s * 4 + 2] * exp(lnBF2) + EPS) - log(denom);
                
                // zero out arrays sum of square arrays
                fill(ssWavelets.begin(), ssWavelets.end(), 0.0);
                fill(ssCovariates.begin(), ssCovariates.end(), 0.0);
                fill(ssWaveletCovariate.begin(), ssWaveletCovariate.end(), 0.0);
            }
        }
        
        // compute final posterior probabilities
        posteriorProbabilities[0][cov] = exp(posteriorProbabilities[0][cov]);
        posteriorProbabilities[1][cov] = exp(posteriorProbabilities[1][cov]) - posteriorProbabilities[0][cov];
        posteriorProbabilities[2][cov] = exp(posteriorProbabilities[2][cov]) - posteriorProbabilities[0][cov];
        posteriorProbabilities[3][cov] = 1 - posteriorProbabilities[0][cov] - posteriorProbabilities[1][cov] - posteriorProbabilities[2][cov];
    }
}

void btda::bayesRegressionDM::performBayesianAnalysis()
{
    // initiate / allocate arrays
    vector <int> gammaAveragingTerms;
    vector <int> waveletCoordinates;
    
    // initialize the piEstimates and the waveletCoordinates arrays.
    int waveletArrayNumberCoordinates = initializeArrays(waveletCoordinates, gammaAveragingTerms);
    
    // construct Pi estimate matrix
    expectationMaximization(waveletCoordinates, gammaAveragingTerms, waveletArrayNumberCoordinates);
    
    // compute posterior probabilities
    computePosteriorProbabilities(waveletCoordinates, waveletArrayNumberCoordinates);
}

// Function: expectationMaximization
// Purpose: Perform EM maximization of the piEstimates of the group probabilities.
//
// Inputs:
// 1. waveletCoordinates
// 2. gammaAveragingTerms
// 3. waveletArrayNumberCoordinates
//
// Actions:
//
// General Notes:
// There are 4 * (numScales + 1) elements of the piEstimates array.  For example
// If numScales is 0, then no group scaling.  Thus piEstimates = [\pi_0, \pi_1, \pi_2, \pi_3].
// If numScales is 1, then there is group scaling and
// piEstimates =
// 0: [\pi_0, \pi_1, \pi_2, \pi_3]
// 1: [\pi_4, \pi_5, \pi_6, \pi_7]

void btda::bayesRegressionDM::expectationMaximization(vector <int> & waveletCoordinates, vector <int> & gammaAveragingTerms, int waveletArrayNumberCoordinates)
{
    /* Variable Initialization */
    // temporary vector to hold interim calculations of the pi estimates
    vector <double> tempPiEstimates(numPi, 0.0);
    
    // constants
    int NITERS = 1000;
    double thresh = 1e-20;
    
    /* Perform Expectation Maximization */
    double relativeLikelihood, lnLikelihoodOld = 0.0, lnLikelihood = -1.0;
    int iters = 0;
    
    // perform EM
    do
    {
        // update the log Likelihood variables
        lnLikelihoodOld = lnLikelihood;
        lnLikelihood = 0.0;
        
        // move along the covariates.
        for (int cov = 0; cov < numCovariates; cov++)
        {
            
            // move along the scales
            for (int s = 0; s < waveletArrayNumberCoordinates - 1; s++)
            {
                // move along the wavelet coefficients at the given scale
                for (int wav = waveletCoordinates[s]; wav < waveletCoordinates[s+1]; wav++)
                {
                    
                    // compute the logLikelihood for the given coefficient
                    if (use[wav] == 1) {
                        lnLikelihood += computeSingleLogLikelihood(tempPiEstimates, cov, wav, s);
                    }
                }
            }
            
            // update the max lnLikelihood for the covariate
            if (lnLikelihood > lnLikelihoodRatios[cov])
            {
                lnLikelihoodRatios[cov] = lnLikelihood;
            }
                
        }
        
        // Update piEstimates
        for (int i = 0; i < numPi; i++)
        {
            piEstimates[i] = tempPiEstimates[i] / (double) gammaAveragingTerms[i];
            tempPiEstimates[i] = 0.0;
        }
        
        // compute relative likelihood
        relativeLikelihood = abs(lnLikelihoodOld - lnLikelihood) / abs(lnLikelihoodOld + EPS);
        iters += 1;
    } while (relativeLikelihood > thresh && iters < NITERS);
}

// Function:    computeSingleLogLikelihood
// Purpose:     Computes the log-likelihood for a specified wavelet coefficient at a specified covariate.
//
// Inputs:      1)
//              2)
//              3)
//              4)
//
// Outputs:     1)

double btda::bayesRegressionDM::computeSingleLogLikelihood(vector <double> & tempPiEstimates, int cov, int wav, int s)
{
    
    // Allocate constants
    double logLikelihood;
    double lnBF1 = 0, lnBF2 = 0, lnBF3 = 0;
    double gamma0 = 0, gamma1 = 0, gamma2 = 0, gamma3 = 0;
    
    // Allocate memory for arrays
    vector<double> ssWavelets(3, 0.0);
    vector<double> ssCovariates(3, 0.0);
    vector<double> ssWaveletCovariate(4, 0.0);
    
    // Compute Sum of Squares
    computeSumOfSquares(ssWavelets, ssCovariates, ssWaveletCovariate, cov, wav);
    
    
    // compute log Bayes Factors
    lnBF1 = computeLnBayesFactor12(ssWavelets, ssCovariates, ssWaveletCovariate, 0);
    
    lnBF2 = computeLnBayesFactor12(ssWavelets, ssCovariates, ssWaveletCovariate, 1);
    
    lnBF3 = computeLnBayesFactor3(ssWavelets, ssCovariates, ssWaveletCovariate);
    
    // update gamma values -- written to prevent exponential term from blowing up...
    double denom = (piEstimates[s * 4] * exp(-lnBF3) +
                    piEstimates[s * 4 + 1] * exp(lnBF1 - lnBF3) +
                    piEstimates[s * 4 + 2] * exp(lnBF2 - lnBF3) +
                    piEstimates[s * 4 + 3]);
    
    gamma0 = (piEstimates[s * 4] * exp(-lnBF3)) / denom;
    gamma1 = (piEstimates[s * 4 + 1] * exp(lnBF1 - lnBF3)) / denom;
    gamma2 = (piEstimates[s * 4 + 2] * exp(lnBF2 - lnBF3)) / denom;
    gamma3 = 1 - gamma0 - gamma1 - gamma2;
    
    
// ORIGINAL CODE -- KEEP IN CASE THE NEW METHOD IS INVALID OR FAILS
//    // update gamma values
//    double denom = (piEstimates[s * 4] +
//                    piEstimates[s * 4 + 1] * exp(lnBF1) +
//                    piEstimates[s * 4 + 2] * exp(lnBF2) +
//                    piEstimates[s * 4 + 3] * exp(lnBF3));
//    gamma0 = piEstimates[s * 4] / denom;
//    gamma1 = (piEstimates[s * 4 + 1] * exp(lnBF1)) / denom;
//    gamma2 = (piEstimates[s * 4 + 2] * exp(lnBF2)) / denom;
//    gamma3 = 1 - gamma0 - gamma1 - gamma2;
    
    // update the piEstimates arrray for the next step.
    tempPiEstimates[s * 4] += gamma0;
    tempPiEstimates[s * 4 + 1] += gamma1;
    tempPiEstimates[s * 4 + 2] += gamma2;
    tempPiEstimates[s * 4 + 3] += gamma3;
    
    // compute log-likelihood
    logLikelihood = gamma0 * log(piEstimates[s * 4] + EPS);
    logLikelihood += gamma1 * log(piEstimates[s * 4 + 1] + EPS) + gamma1 * lnBF1;
    logLikelihood += gamma2 * log(piEstimates[s * 4 + 2] + EPS) + gamma2 * lnBF2;
    logLikelihood += gamma3 * log(piEstimates[s * 4 + 3] + EPS) + gamma3 * lnBF3;
    
    return logLikelihood;
}

// Function: computeLnBayesFactor3()
// Purpose:  Computes the natural log of the Bayes' Factor for alternative hypothesis 3.
// Input:
// 1. ssWavelets          - address for the vector of sum of squares of the wavelets coefficients
// 2. ssCovariates        - address for the vector of sum of squares of the covariates
// 3. ssWaveletCovariate  - address for the vector of sum of squares of the wavelets / covariates interactions
// 4. priorK              - (double) prior \sigma_a^-2
// 5. numSamples          - (int) number of samples
// 6. numWavelets         - (int) number of wavelets per homology group
// 7. numCovariates       - (int) number of covariates in the model
// 8. m                   - (int) parameter in the matrix-T distribution
//
// Returns:
// 1. lnBF                - (double) The computed log Bayes Factor for the given covariate.

double btda::bayesRegressionDM::computeLnBayesFactor3(vector< double > & ssWavelets, vector< double > & ssCovariates, vector< double > & ssWaveletCovariate)
{
    // initialize variables
    double lnBF;
    
    // allocate arrays
    vector<double> sxx(4, 0.0);
    vector<double> sww(4, 0.0);
    vector<double> swx(4, 0.0);
    vector<double> sx0x0(4, 0.0);
    vector<double> sxx_inv(4, 0.0);
    vector<double> s_w_given_x(4, 0.0);
    
    // extract the arrays for computation
    // Extract sxx = (S_{XX} + K^{\to})
    sxx[0] = ssCovariates[0];
    sxx[1] = ssCovariates[1];
    sxx[2] = ssCovariates[1];
    sxx[3] = ssCovariates[2] + priorK;
    
    // Extract S_{WW}
    sww[0] = ssWavelets[0];
    sww[1] = ssWavelets[1];
    sww[2] = ssWavelets[1];
    sww[3] = ssWavelets[2];
    
    // Extract S_{WX}
    swx[0] = ssWaveletCovariate[0];
    swx[1] = ssWaveletCovariate[1];
    swx[2] = ssWaveletCovariate[2];
    swx[3] = ssWaveletCovariate[3];
    
    // compute sxx_inv = (S_{XX} + K^{\to})^{-1}
    invert2x2Matrix(sxx, sxx_inv);
    
    /*
     compute s_w_given_x = S_{WW} - S_{WX}(S_{XX} + K^{\to})^{-1}S_{WX}^T
     */
    
    // 1. Compute S_{WX}(S_XX + K^{\to})^{-1}
    s_w_given_x[0] = swx[0] * sxx_inv[0] + swx[1] * sxx_inv[2];
    s_w_given_x[1] = swx[0] * sxx_inv[1] + swx[1] * sxx_inv[3];
    s_w_given_x[2] = swx[2] * sxx_inv[0] + swx[3] * sxx_inv[2];
    s_w_given_x[3] = swx[2] * sxx_inv[1] + swx[3] * sxx_inv[3];
    
    // 2. Compute S_{WX}(S_XX + K^{\to})^{-1}S_{WX}^T
    // temporary variables to prevent values from being overwritten before use.
    double temp_0 = s_w_given_x[0], temp_1 = s_w_given_x[1];
    double temp_2 = s_w_given_x[2], temp_3 = s_w_given_x[3];
    
    s_w_given_x[0] = temp_0 * swx[0] + temp_1 * swx[1];
    s_w_given_x[1] = temp_0 * swx[2] + temp_1 * swx[3];
    s_w_given_x[2] = temp_2 * swx[0] + temp_3 * swx[1];
    s_w_given_x[3] = temp_2 * swx[2] + temp_3 * swx[3];
    
    // 3. Compute S_{WW} - S_{WX}(S_XX + K^{\to})^{-1}S_{WX}^T
    s_w_given_x[0] = sww[0] - s_w_given_x[0];
    s_w_given_x[1] = sww[1] - s_w_given_x[1];
    s_w_given_x[2] = sww[2] - s_w_given_x[2];
    s_w_given_x[3] = sww[3] - s_w_given_x[3];
    
    
    /*
     compute S_{WW} - n^{-1} S_{WX_0}S_{WX_0}^T
     */
    
    sx0x0[0] = sww[0] - swx[0] * swx[0] / (double) numSamples;
    sx0x0[1] = sww[1] - swx[0] * swx[2] / (double) numSamples;
    sx0x0[2] = sww[2] - swx[0] * swx[2] / (double) numSamples;
    sx0x0[3] = sww[3] - swx[2] * swx[2] / (double) numSamples;
    
    /*
     compute log(Bayes Factor 3)
     */
    
    lnBF = log(numSamples) - log(priorK) - log(computeDeterminant2x2(sxx));
    lnBF += (double) (m + numSamples) / (double) 2 * (log(computeDeterminant2x2(sx0x0) + EPS) - log(computeDeterminant2x2(s_w_given_x)) + EPS);
    
    return lnBF;
}

// Function: computeLnBayesFactor12()
// Purpose:  Computes the natural log of the Bayes' Factors for alternative hypothesis 1 and 2.
// Input:
// 1. ssWavelets          - address for the vector of sum of squares of the wavelets coefficients
// 2. ssCovariates        - address for the vector of sum of squares of the covariates
// 3. ssWaveletCovariate  - address for the vector of sum of squares of the wavelets / covariates interactions
// 4. priorK              - (double) prior \sigma_a^-2
// 5. numSamples          - (int) number of samples
// 6. numWavelets         - (int) number of wavelets per homology group
// 7. numCovariates       - (int) number of covariates in the model
// 8. m                   - (int) parameter in the matrix-T distribution
// 9. k                   - (int) binary value to indicate if we are compute BF^1 (k = 0) or BF^2 (k = 1)
// (k = 0 <=> w^0 is significant and w^1 is not).
// Returns:
// 1. lnBF                - (double) The computed log Bayes Factor for the given covariate.
// */

double btda::bayesRegressionDM::computeLnBayesFactor12(vector<double> & ssWavelets, vector<double> & ssCovariates, vector<double> & ssWaveletCovariate, int k)
{
    // initialize variables
    double sww;
    double lnBF;
    
    // allocate arrays
    vector<double> stxtx(4, 0.0);
    vector<double> stxtx_inv(4, 0.0);
    vector<double> sxx(9, 0.0);
    vector<double> sxx_inv(9, 0.0);
    vector<double> swx(3, 0.0);
    vector<double> swtx(2, 0.0);
    
    
    // extract the arrays for computation
    // if k == 0, then w^0 is sig, w^1 is not.
    if (k == 0)
    {
        // Extract stxtx = (S_{\tilde{X}\tilde{X})
        stxtx[0] = ssCovariates[0]; 
        stxtx[1] = ssWaveletCovariate[2];
        stxtx[2] = ssWaveletCovariate[2];
        stxtx[3] = ssWavelets[2];
        
        // Extract sxx = (S_{XX} + K_0^\to)
        sxx[0] = ssCovariates[0];
        sxx[1] = ssWaveletCovariate[2];
        sxx[2] = ssCovariates[1];
        sxx[3] = ssWaveletCovariate[2];
        sxx[4] = ssWavelets[2];
        sxx[5] = ssWaveletCovariate[3];
        sxx[6] = ssCovariates[1];
        sxx[7] = ssWaveletCovariate[3];
        sxx[8] = ssCovariates[2] + priorK;
        
        // Extract S_{WW}
        sww = ssWavelets[0];
        
        // Extract S_{W\tilde{X}}
        swtx[0] = ssWaveletCovariate[0];
        swtx[1] = ssWavelets[1];
        
        // Extract S_{WX}}
        swx[0] = ssWaveletCovariate[0];
        swx[1] = ssWavelets[1];
        swx[2] = ssWaveletCovariate[1];
        
    }
    else
    {
        // Extract stxtx = (S_{\tilde{X}\tilde{X})
        stxtx[0] = ssCovariates[0];
        stxtx[1] = ssWaveletCovariate[0];
        stxtx[2] = ssWaveletCovariate[0];
        stxtx[3] = ssWavelets[0];
        
        // Extract sxx = (S_{XX} + K_0^\to)
        sxx[0] = ssCovariates[0];
        sxx[1] = ssWaveletCovariate[0];
        sxx[2] = ssCovariates[1];
        sxx[3] = ssWaveletCovariate[0];
        sxx[4] = ssWavelets[0];
        sxx[5] = ssWaveletCovariate[1];
        sxx[6] = ssCovariates[1];
        sxx[7] = ssWaveletCovariate[1];
        sxx[8] = ssCovariates[2] + priorK;
        
        // Extract S_{WW}
        sww = ssWavelets[2];
        
        // Extract S_{W\tilde{X}}
        swtx[0] = ssWaveletCovariate[2];
        swtx[1] = ssWavelets[1];
        
        // Extract S_{WX}}
        swx[0] = ssWaveletCovariate[2];
        swx[1] = ssWavelets[1];
        swx[2] = ssWaveletCovariate[3];
    }
    
    // compute sxx_inv = S_{\tilde{X}\tilde{X}}^{-1}
    invert2x2Matrix(stxtx, stxtx_inv);
    
    // compute sxx_inv = (S_{XX} + K^{\to})^{-1}
    invert3x3Matrix(sxx, sxx_inv);
    
    /* Compute log Bayes Factor */
    lnBF = log(computeDeterminant2x2(stxtx) + EPS) - log(computeDeterminant3x3(sxx) + EPS) - log(priorK);
    
    // Compute ln of S_{W|X} terms
    double temp_den, temp_num;
    
    // denominator
    temp_den = swx[0] * (swx[0] * sxx_inv[0] + swx[1] * sxx_inv[3] + swx[2] * sxx_inv[6]);
    temp_den += swx[1] * (swx[0] * sxx_inv[1] + swx[1] * sxx_inv[4] + swx[2] * sxx_inv[7]);
    temp_den += swx[2] * (swx[0] * sxx_inv[2] + swx[1] * sxx_inv[5] + swx[2] * sxx_inv[8]);
    temp_den = log(sww - temp_den);
    
    // numerator
    temp_num = swtx[0] * (swtx[0] * stxtx_inv[0] + swtx[1] * stxtx_inv[2]);
    temp_num += swtx[1] * (swtx[0] * stxtx_inv[1] + swtx[1] * stxtx_inv[3]);
    temp_num = log(sww - temp_num);
    
    lnBF += ((double) (m + numSamples) / (double) 2) * (temp_num - temp_den);
    
    return lnBF;
}

// Function: computeSumOfSquares()
//
// Purpose:  Computes the sum of squares used in the Bayes' Factor calculations
// for all wavelet coefficients and all covariates.
//
// Input:
// 1. ssWavelets          - address for matrix of sum of squares of wavelets matrix
// 2. ssCovariates        - address for matrix of sum of squares of covariates matrix
// 3. ssWaveletCovariate  - address for matrix of sum of squares of wavelets / covariates matrix
// 4. cov                 - (int) index of the covariate matrix we are accessing
// 5. wav                 - (int) index of the wavelet matrix we are accessing
//
// Action:
// 1. ssWavelets          - Fills the matrix of the sum of squares for the wavelets
// 2. ssWaveletCovariate  - Fills the matrix of the sum of squares for the wavelets and covariates
// 3. ssCovariates        - Fills the matrix of the sum of squares for the covariates

void btda::bayesRegressionDM::computeSumOfSquares(vector<double> & ssWavelets, vector<double> & ssCovariates, vector<double> & ssWaveletCovariate, int cov, int wav)
{
    ssCovariates[0] = (double) numSamples;
    
    for (int row = 0; row < numSamples; row++)
    {
        double x = covariateMatrix[row][cov];
        double w0 = waveletCoefficients[row][wav];
        double w1 = waveletCoefficients[row][wav + numWavelets];
        
        // Compute the values for the wavelet sum of squares (S_{WW} = W^T \cdot W) array
        ssWavelets[0] += w0 * w0;
        ssWavelets[1] += w0 * w1;
        ssWavelets[2] += w1 * w1;
        
        // Compute the values for S_{XX} = X^T \cdot X
        // -! K will be added later
        ssCovariates[1] += x;
        ssCovariates[2] += x * x;
        
        // Compute the values for S_{WX} = W^T \cdot X
        ssWaveletCovariate[0] += w0;
        ssWaveletCovariate[1] += w0 * x;
        ssWaveletCovariate[2] += w1;
        ssWaveletCovariate[3] += w1 * x;
    }
}

//  Function: initializeArrays
//  Purpose:  initialize the waveletArrayNumberCoordinates and piEstimates arrays
//  Input:    1) waveletCoordinates - (address) array that stores the column indices associated with the waveletCoefficient levels.
//            2) piEstimates - (address) initiates the estimates of the pi terms.
//            3) use - (address) array indicating which wavelet coefficients to use in each array.
//
// Action:    1)
//
// Returns:   1) waveletArrayNumberCoordinates - (int) number of elements in the waveletCoordinates array.

int btda::bayesRegressionDM::initializeArrays(vector<int> & waveletCoordinates, vector<int> & gammaAveragingTerms)
{
    // allocate variables
    int waveletArrayNumberCoordinates;
    gammaAveragingTerms.resize(numPi, 0);
    
    // initialize pi estimates
    for (int i = 0; i < numPi; i++)
    {
        piEstimates[i] = 1 / (double) numPi;
    }
    
    // get the wavelet coordinates for each scale.
    if (groupScaling == false || numWavelets == 4)
    {
        waveletCoordinates.resize(2);
        waveletCoordinates[0] = 0;
        waveletCoordinates[1] = numWavelets;
        waveletArrayNumberCoordinates = 2;
    }
    else
    {
        waveletArrayNumberCoordinates = numScales + 2;
        waveletCoordinates.resize(waveletArrayNumberCoordinates);
        for (int i = 1; i < waveletArrayNumberCoordinates; i++)
        {
            waveletCoordinates[i] = pow(4, i - 1);
        }
    }
    
    // get the divisor for each gamma term
    for (int i = 0; i < waveletArrayNumberCoordinates - 1; i++)
    {
        int tempCounts = 0;
        int lb = waveletCoordinates[i];
        int ub = waveletCoordinates[i+1];
        
        for (int j = lb; j < ub; j++)
        {
            if (use[j] == 1)
                tempCounts += 1;
        }
        
        // TODO: Return an error if tempCounts is zero and kill the program.
        gammaAveragingTerms[i * 4] = tempCounts * numCovariates;
        gammaAveragingTerms[i * 4 + 1] = tempCounts * numCovariates;
        gammaAveragingTerms[i * 4 + 2] = tempCounts * numCovariates;
        gammaAveragingTerms[i * 4 + 3] = tempCounts * numCovariates;
    }
    
    return waveletArrayNumberCoordinates;
}
