//
//  bayestda.h
//
//  Created by Patrick Medina on 6/5/17.
//  Copyright © 2017 Patrick Medina. All rights reserved.
//

#ifndef bayestda_h
#define bayestda_h

#include <iostream>
#include <vector>

using namespace std;

// namespace for the Bayes TDA classes / routines
namespace btda {
    
    // Class:   bayesRegressionDM
    // Perform Bayesian regression as outlined in Chapter 4 of [1] using the dependence model.
    //
    // Notes: 1) Template allows generalization of the input.  This allows the Python user to not worry about type of the main input variables -- Python will handle it directly.
    
    class bayesRegressionDM {
        
        private:
            // Wavelet / Inference parameters
            int numScales, numWavelets, numCovariates, numSamples;
            bool groupScaling;
            int numPi;
        
            // Bayesian parameters
            double priorK;
            int m;
        
            // user input data
            vector< vector< float > > waveletCoefficients;
            vector< vector< float > > covariateMatrix;
            vector <int> use;
        
            // Objects we are storing our inference in.
            vector <double> piEstimates;
            vector <double> lnLikelihoodRatios;
            vector < vector<double> > posteriorProbabilities;
        
        public:
            // Constructor / Destructor
            bayesRegressionDM(int, int, int, int, vector< vector< float > > &, vector< vector< float > > &, vector<int> &, bool);
            ~bayesRegressionDM() { };
        
            // Bayesian Analysis Routines
            void performBayesianAnalysis();
            void expectationMaximization(vector<int> &, vector<int> &, int);
            double computeSingleLogLikelihood(vector<double> & tempPiEstimates, int cov, int wav, int s);
            void computePosteriorProbabilities(vector<int> &, int);
            double computeLnBayesFactor3(vector<double> &, vector<double> &, vector<double> &);
            double computeLnBayesFactor12(vector<double> &, vector<double> &, vector<double> &, int);

            // Support Routines
            int initializeArrays(vector<int> &, vector<int> &);
            void computeSumOfSquares(vector<double> &, vector<double> &, vector<double> &, int, int);
            vector< vector<double> > * getPosteriorProbabilities() {return & posteriorProbabilities;};
            vector<double> * getPiEstimates() {return & piEstimates;};
    };
}


#endif /* bayestda_h */
