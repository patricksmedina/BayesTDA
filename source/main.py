import numpy as np
import math

def computeDeterminant2x2(matrix):
    return(matrix[0] * matrix[3] - matrix[1] * matrix[2])


def computeDeterminant3x3(matrix):
    det = 0.0

    det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
    det -= matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
    det += matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])

    return(det)

def invert2x2Matrix(matrix):
    inverse = np.zeros(4, float)

    det_mat = computeDeterminant2x2(matrix)

    inverse[0] = matrix[3] / det_mat
    inverse[1] = -1.0 * matrix[1] / det_mat
    inverse[2] = -1.0 * matrix[2] / det_mat
    inverse[3] = matrix[0] / det_mat

    return(inverse)

def invert3x3Matrix(matrix):
    inverse = np.zeros(9, float)

    inverse[0] = matrix[4] * matrix[8] - matrix[5] * matrix[7]
    inverse[1] = -1.0 * (matrix[1] * matrix[8] - matrix[2] * matrix[7])
    inverse[2] = matrix[1] * matrix[5] - matrix[2] * matrix[4]
    inverse[3] = -1.0 * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
    inverse[4] = matrix[0] * matrix[8] - matrix[2] * matrix[6]
    inverse[5] = -1.0 * (matrix[0] * matrix[5] - matrix[2] * matrix[3])
    inverse[6] = matrix[3] * matrix[7] - matrix[4] * matrix[6]
    inverse[7] = -1.0 * (matrix[0] * matrix[7] - matrix[1] * matrix[6])
    inverse[8] = matrix[0] * matrix[4] - matrix[1] * matrix[3]

    den_mat = matrix[0] * inverse[0] + matrix[1] * inverse[3] + matrix[2] * inverse[6]

    # scale by the determinant
    for i in range(9):
        inverse[i] /= den_mat

    return(inverse)

# generate samples
def generateSamples(numSamples=25, numCovariates=2):
    randomSamples = np.random.randint(256, size=(2, 2, numSamples * 2))
    waveletCoefficients = []
    for i in range(numSamples * 2):
        waveletCoefficients.append(wv.construct2DWaveletVector(
            randomSamples[:, :, i], basis="haar"))

    waveletCoefficients = np.array(waveletCoefficients)
    waveletCoefficients = np.hstack((waveletCoefficients[:numSamples, :], waveletCoefficients[numSamples:, :]))

    covariateMatrix = np.random.randint(0, 2, size=(numSamples, numCovariates))

    return(waveletCoefficients, covariateMatrix)

# compute Sum of Squares
def computeSumOfSquares(waveletCoefficients,
                        covariateMatrix,
                        numSamples,
                        numWavelets,
                        cov,
                        wav):

    ssWavelets = np.zeros(3, float)
    ssCovariates = np.zeros(3, float)
    ssWaveletCovariate = np.zeros(4, float)

    ssCovariates[0] = numSamples

    for row in range(numSamples):
        x = covariateMatrix[row][cov]
        w0 = waveletCoefficients[row][wav]
        w1 = waveletCoefficients[row][wav + numWavelets]

        # Compute the values for the wavelet sum of squares (S_{WW} = W^T \cdot W) array
        ssWavelets[0] += w0 * w0
        ssWavelets[1] += w0 * w1
        ssWavelets[2] += w1 * w1

        # Compute the values for S_{XX} = X^T \cdot X
        # -! K will be added later
        ssCovariates[1] += x
        ssCovariates[2] += x * x

        # Compute the values for S_{WX} = W^T \cdot X
        ssWaveletCovariate[0] += w0
        ssWaveletCovariate[1] += w0 * x
        ssWaveletCovariate[2] += w1
        ssWaveletCovariate[3] += w1 * x

    return(ssWavelets, ssCovariates, ssWaveletCovariate)

# compute logBayesFactor
# def computeLnBayesFactor12NP( ssWavelets,
#                               ssCovariates,
#                               ssWaveletCovariate,
#                               waveletCoefficients,
#                               covariateMatrix,
#                               wav,cov,
#                               numSamples,
#                               numWavelets,
#                               numCovariates,
#                               m=1,
#                               priorK=0.05,
#                               k=0):
#
#     if k == 0:
#         tx = np.hstack((np.ones(shape=(numSamples,1)),
#                           waveletCoefficients[:,wav + numWavelets].reshape(-1,1)))
#         stxtx2 = np.dot(tx.T,tx)
#         x = np.hstack((np.ones(shape=(numSamples,1)),
#                        waveletCoefficients[:,wav + numWavelets].reshape(-1,1),
#                        covariateMatrix[:,cov].reshape(-1,1)))
#         sxx2 = np.dot(x.T,x) + np.diag((0,0,priorK))
#         sww2 = np.dot(waveletCoefficients[:, wav].T, waveletCoefficients[:, wav])
#         swtx2 = np.dot(waveletCoefficients[:, wav].T,tx)
#         swx2 = np.dot(waveletCoefficients[:, wav].T,x)
#     else:
#         tx = np.hstack((np.ones(shape=(numSamples,1)),
#                           waveletCoefficients[:,wav].reshape(-1,1)))
#         stxtx2 = np.dot(tx.T,tx)
#         x = np.hstack((np.ones(shape=(numSamples,1)),
#                        waveletCoefficients[:,wav].reshape(-1,1),
#                        covariateMatrix[:,cov].reshape(-1,1)))
#         sxx2 = np.dot(x.T,x) + np.diag((0,0,priorK))
#         sww2 = np.dot(waveletCoefficients[:, wav + numWavelets].T, waveletCoefficients[:, wav + numWavelets])
#         swtx2 = np.dot(waveletCoefficients[:, wav].T,tx)
#         swx2 = np.dot(waveletCoefficients[:, wav].T,x)
#
#     ########
#
#     num = np.log(sww2 - np.dot(np.dot(swtx2, np.linalg.inv(stxtx2)).reshape(1,-1), swtx2))
#     den = np.log(sww2 - np.dot(np.dot(swx2, np.linalg.inv(sxx2)).reshape(1,-1), swx2))
#
#     logBF = math.log(1 / priorK) + np.log(np.linalg.det(stxtx2)) - np.log(np.linalg.det(sxx2))
#     logBF += float(m + numSamples) / 2.0 * (num - den)
#     print(logBF)
#
# def computeLnBayesFactor12( ssWavelets,
#                             ssCovariates,
#                             ssWaveletCovariate,
#                             numSamples,
#                             numWavelets,
#                             numCovariates,
#                             m=1,
#                             priorK=0.05,
#                             k=0):
#
#         # initialize variables
#         sww = 0.0
#         lnBF = 0.0
#
#         # allocate arrays
#         stxtx = np.zeros(4, float)
#         stxtx_inv = np.zeros(4, float)
#         sxx = np.zeros(9, float)
#         sxx_inv = np.zeros(9, float)
#         swx = np.zeros(3, float)
#         swtx = np.zeros(2, float)
#
#         # extract the arrays for computation
#         # if k == 0, then w^0 is sig, w^1 is not.
#         if k == 0:
#             # // Extract stxtx = (S_{\tilde{X}\tilde{X})
#             stxtx[0] = ssCovariates[0]
#             stxtx[1] = ssWaveletCovariate[2]
#             stxtx[2] = ssWaveletCovariate[2]
#             stxtx[3] = ssWavelets[2]
#
#             # // Extract sxx = (S_{XX} + K_0^\to)
#             sxx[0] = ssCovariates[0]
#             sxx[1] = ssWaveletCovariate[2]
#             sxx[2] = ssCovariates[1]
#             sxx[3] = ssWaveletCovariate[2]
#             sxx[4] = ssWavelets[2]
#             sxx[5] = ssWaveletCovariate[3]
#             sxx[6] = ssCovariates[1]
#             sxx[7] = ssWaveletCovariate[3]
#             sxx[8] = ssCovariates[2] + priorK
#
#             # // Extract S_{WW}
#             sww = ssWavelets[0]
#
#             # // Extract S_{W\tilde{X}}
#             swtx[0] = ssWaveletCovariate[0]
#             swtx[1] = ssWavelets[1]
#
#             # // Extract S_{WX}}
#             swx[0] = ssWaveletCovariate[0]
#             swx[1] = ssWavelets[1]
#             swx[2] = ssWaveletCovariate[1]
#         else:
#             #// Extract stxtx = (S_{\tilde{X}\tilde{X})
#             stxtx[0] = ssCovariates[0]
#             stxtx[1] = ssWaveletCovariate[0]
#             stxtx[2] = ssWaveletCovariate[0]
#             stxtx[3] = ssWavelets[0]
#
#             # // Extract sxx = (S_{XX} + K_0^\to)
#             sxx[0] = ssCovariates[0]
#             sxx[1] = ssWaveletCovariate[0]
#             sxx[2] = ssCovariates[1]
#             sxx[3] = ssWaveletCovariate[0]
#             sxx[4] = ssWavelets[0]
#             sxx[5] = ssWaveletCovariate[1]
#             sxx[6] = ssCovariates[1]
#             sxx[7] = ssWaveletCovariate[1]
#             sxx[8] = ssCovariates[2] + priorK
#
#             #// Extract S_{WW}
#             sww = ssWavelets[2]
#
#             #// Extract S_{W\tilde{X}}
#             swtx[0] = ssWaveletCovariate[2]
#             swtx[1] = ssWavelets[1]
#
#             #// Extract S_{WX}}
#             swx[0] = ssWaveletCovariate[2]
#             swx[1] = ssWavelets[1]
#             swx[2] = ssWaveletCovariate[3]
#
#         #// compute sxx_inv = S_{\tilde{X}\tilde{X}}^{-1}
#         stxtx_inv = invert2x2Matrix(stxtx)
#
#         #// compute sxx_inv = (S_{XX} + K^{\to})^{-1}
#         sxx_inv = invert3x3Matrix(sxx)
#
#         #/* Compute log Bayes Factor */
#         lnBF = math.log(1 / priorK) + math.log(computeDeterminant2x2(stxtx)) - math.log(computeDeterminant3x3(sxx))
#
#         #// Compute ln of S_{W|X} terms
#         temp_den, temp_num = 0.0, 0.0
#
#         #// denominator
#         temp_den = swx[0] * (swx[0] * sxx_inv[0] + swx[1] * sxx_inv[3] + swx[2] * sxx_inv[6])
#         temp_den += swx[1] * (swx[0] * sxx_inv[1] + swx[1] * sxx_inv[4] + swx[2] * sxx_inv[7])
#         temp_den += swx[2] * (swx[0] * sxx_inv[2] + swx[1] * sxx_inv[5] + swx[2] * sxx_inv[8])
#         temp_den = math.log(sww - temp_den)
#
#         #// numerator
#         temp_num = swtx[0] * (swtx[0] * stxtx_inv[0] + swtx[1] * stxtx_inv[2])
#         temp_num += swtx[1] * (swtx[0] * stxtx_inv[1] + swtx[1] * stxtx_inv[3])
#         temp_num = math.log(sww - temp_num)
#
#         lnBF += float(m + numSamples) / float(2) * (temp_num - temp_den)
#         return(lnBF)

def computeLnBayesFactor3NP(waveletCoefficients,
                            covariateMatrix,
                            numSamples,
                            numWavelets,
                            numCovariates,
                            cov,
                            wav,
                            priorK=0.05,
                            m=1):
    X0 = np.ones(shape=(numSamples,1))
    sX0X0 = np.sum(X0)

    X = np.hstack((X0, covariateMatrix[:,cov].reshape(-1,1)))
    sXX = np.dot(X.T,X) + np.diag((0,priorK))
    sWW = np.dot(waveletCoefficients[:, (wav, wav + numWavelets)].T,
                 waveletCoefficients[:, (wav, wav + numWavelets)])

    sWX0 = np.dot(waveletCoefficients[:, (wav, wav + numWavelets)].T,X0)
    sWX = np.dot(waveletCoefficients[:, (wav, wav + numWavelets)].T,X)

    ########
    sXX_inv = np.linalg.inv(sXX)
    # print(sWW - np.dot(np.dot(sWX.T,sXX_inv),sWX))
    # print(np.linalg.det(sWW - np.dot(np.dot(sWX.T,sXX_inv),sWX)))
    num = np.log(np.linalg.det(sWW - 1 / float(numSamples) * np.outer(sWX0, sWX0)))
    den = np.log(np.linalg.det())
    #print(den)
    #
    logBF = math.log(priorK) + np.log(numSamples) - np.log(np.linalg.det(sXX))
    logBF += float(m + numSamples) / 2.0 * (num - den)
    print(logBF)

def computeLnBayesFactor3(ssWavelets,
                          ssCovariates,
                          ssWaveletCovariate,
                          numSamples,
                          numWavelets,
                          numCovariates,
                          priorK=0.05,
                          m=1):

    # // allocate arrays
    sxx = np.zeros(4,float)
    sww = np.zeros(4,float)
    swx = np.zeros(4,float)
    sx0x0 = np.zeros(4,float)
    s_w_given_x = np.zeros(4,float)

    # // extract the arrays for computation
    # // Extract sxx = (S_{XX} + K^{\to})
    sxx[0] = ssCovariates[0]
    sxx[1] = ssCovariates[1]
    sxx[2] = ssCovariates[1]
    sxx[3] = ssCovariates[2] + priorK

    # // Extract S_{WW}
    sww[0] = ssWavelets[0]
    sww[1] = ssWavelets[1]
    sww[2] = ssWavelets[1]
    sww[3] = ssWavelets[2]

    # // Extract S_{WX}
    swx[0] = ssWaveletCovariate[0]
    swx[1] = ssWaveletCovariate[1]
    swx[2] = ssWaveletCovariate[2]
    swx[3] = ssWaveletCovariate[3]

    # // compute sxx_inv = (S_{XX} + K^{\to})^{-1}
    sxx_inv = invert2x2Matrix(sxx)

    # /*
    #  compute s_w_given_x = S_{WW} - S_{WX}(S_{XX} + K^{\to})^{-1}S_{WX}^T
    #  */

    # // 1. Compute S_{WX}(S_XX + K^{\to})^{-1}
    s_w_given_x[0] = swx[0] * sxx_inv[0] + swx[1] * sxx_inv[2]
    s_w_given_x[1] = swx[0] * sxx_inv[1] + swx[1] * sxx_inv[3]
    s_w_given_x[2] = swx[2] * sxx_inv[0] + swx[3] * sxx_inv[2]
    s_w_given_x[3] = swx[2] * sxx_inv[1] + swx[3] * sxx_inv[3]

    # // 2. Compute S_{WX}(S_XX + K^{\to})^{-1}S_{WX}^T
    # // temporary variables to prevent values from being overwritten before use.
    temp_0 = s_w_given_x[0]
    temp_1 = s_w_given_x[1]
    temp_2 = s_w_given_x[2]
    temp_3 = s_w_given_x[3]

    s_w_given_x[0] = temp_0 * swx[0] + temp_1 * swx[1]
    s_w_given_x[1] = temp_0 * swx[2] + temp_1 * swx[3]
    s_w_given_x[2] = temp_2 * swx[0] + temp_3 * swx[1]
    s_w_given_x[3] = temp_2 * swx[2] + temp_3 * swx[3]

    # // 3. Compute S_{WW} - S_{WX}(S_XX + K^{\to})^{-1}S_{WX}^T
    s_w_given_x[0] = sww[0] - s_w_given_x[0]
    s_w_given_x[1] = sww[1] - s_w_given_x[1]
    s_w_given_x[2] = sww[2] - s_w_given_x[2]
    s_w_given_x[3] = sww[3] - s_w_given_x[3]

    # /*
    #  compute S_{WW} - n^{-1} S_{WX_0}S_{WX_0}^T
    #  */

    sx0x0[0] = sww[0] - swx[0] * swx[0] / float(numSamples)
    sx0x0[1] = sww[1] - swx[0] * swx[2] / float(numSamples)
    sx0x0[2] = sww[2] - swx[0] * swx[2] / float(numSamples)
    sx0x0[3] = sww[3] - swx[2] * swx[2] / float(numSamples)

    # /*
    #  compute log(Bayes Factor 3)
    #  */
    print(np.array([[s_w_given_x[0], s_w_given_x[1]],[s_w_given_x[2],s_w_given_x[3]]]))
    print(math.log(computeDeterminant2x2(s_w_given_x)))

    lnBF = math.log(numSamples) + math.log(priorK) - math.log(computeDeterminant2x2(sxx))
    lnBF += float(m + numSamples) / 2.0 * (math.log(computeDeterminant2x2(sx0x0)) - math.log(computeDeterminant2x2(s_w_given_x)))

    return(lnBF)

if __name__ == "__main__":
    # generate the samples
numSamples = 1000
numWavelets = 4
numCovariates = 1
groupScaling = True
numScales = np.log(int(numWavelets) / 4)

mu0, mu1 = 10.0, -10.0
wc_b0, wc_b1 = 10.0, -10.0
b0, b1 = 5.0, -5.0

covariateMatrix = [1] * int(np.ceil(numSamples / 2.0)) + \
                  [0] * int(np.floor(numSamples / 2.0))
covariateMatrix = np.array(covariateMatrix).reshape(-1, 1)

wc0 = np.random.normal(
        mu0,
        scale=1,
        size=(numSamples, numWavelets)
)
wc0 += b0 * np.tile(covariateMatrix, numWavelets)

wc1 = np.random.normal(
        mu1,
        scale=1,
        size=(numSamples, numWavelets)
)
wc1 += b1 * np.tile(covariateMatrix, numWavelets)

# wc0 += b0 * np.tile(covariateMatrix, numWavelets)
# wc0 += np.random.normal(0, scale=1, size=(numSamples, numWavelets))

waveletCoefficients = np.hstack((wc0, wc1))
np.savetxt("waveletCoefficients.csv", waveletCoefficients, delimiter=",")
np.savetxt("covariateMatrix.csv", covariateMatrix, delimiter="}")

    # waveletCoefficients = np.array([[149.5,-21.5,-106.5,52.5,224.5,-18.5,-49.5,43.5],[245.5,153.5,49.5,17.5,348.0,80.0,-19.0,-1.0],[231.0,-1.0,34.0,56.0,314.5,-76.5,-48.5,-123.5],[162.5,-8.5,-20.5,-35.5,261.5,39.5,73.5,119.5],[261.5,-41.5,41.5,66.5,380.0,71.0,-118.0,61.0]])
    # covariateMatrix = np.array([[1,0],[0,1],[1,1],[0,1],[1,1]])


    # for cov in range(numCovariates):
    #     for wav in range(numWavelets):
    #         ssWavelets, ssCovariates, ssWaveletCovariate = computeSumOfSquares(waveletCoefficients,
    #                                                                            covariateMatrix,
    #                                                                            numSamples,
    #                                                                            numWavelets,
    #                                                                            cov,
    #                                                                            wav)
    #
    #         print "[INFO] logBF3 (C++) is {}".format( computeLnBayesFactor3(ssWavelets,
    #                                     ssCovariates,
    #                                     ssWaveletCovariate,
    #                                     numSamples,
    #                                     numWavelets,
    #                                     numCovariates))
    #
    #         computeLnBayesFactor3NP(waveletCoefficients,
    #                                 covariateMatrix,
    #                                 numSamples,
    #                                 numWavelets,
    #                                 numCovariates,
    #                                 cov,
    #                                 wav)
