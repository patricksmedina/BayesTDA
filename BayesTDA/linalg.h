//
//  misc.h
//
//
//  Created by Patrick Medina on 6/5/17.
//  Copyright © 2017 Patrick Medina. All rights reserved.
//

#ifndef linalg_h
#define linalg_h

#include <vector>
#include <cmath>

using namespace std;

/*
Function: computeDeterminant2x2
Purpose: Compute the determinant of a 2x2 array.
*/

double computeDeterminant2x2(vector<double> & matrix)
{
    return matrix[0] * matrix[3] - matrix[1] * matrix[2];
}

/*
Function: computeDeterminant3x3
Purpose: Compute the determinant of a 3x3 array.
*/

double computeDeterminant3x3(vector<double> & matrix)
{
    double det;

    det = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7]);
    det -= matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6]);
    det += matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);

    return det;
}

/*
Function: invert2x2Matrix
Purpose: Compute the inverse of a given 2x2 matrix.
*/

void invert2x2Matrix(vector<double> & matrix, vector<double> & inverse)
{
    double det_mat = computeDeterminant2x2(matrix);

    inverse[0] = matrix[3] / det_mat;
    inverse[1] = -1.0 * matrix[1] / det_mat;
    inverse[2] = -1.0 * matrix[2] / det_mat;
    inverse[3] = matrix[0] / det_mat;
}

/*
Function: invert3x3Matrix
Purpose: Compute the inverse of a given 3x3 matrix.
*/

void invert3x3Matrix(vector<double> & matrix, vector<double> & inverse)
{
    inverse[0] = matrix[4] * matrix[8] - matrix[5] * matrix[7];
    inverse[1] = -1.0 * (matrix[1] * matrix[8] - matrix[2] * matrix[7]);
    inverse[2] = matrix[1] * matrix[5] - matrix[2] * matrix[4];
    inverse[3] = -1.0 * (matrix[3] * matrix[8] - matrix[5] * matrix[6]);
    inverse[4] = matrix[0] * matrix[8] - matrix[2] * matrix[6];
    inverse[5] = -1.0 * (matrix[0] * matrix[5] - matrix[2] * matrix[3]);
    inverse[6] = matrix[3] * matrix[7] - matrix[4] * matrix[6];
    inverse[7] = -1.0 * (matrix[0] * matrix[7] - matrix[1] * matrix[6]);
    inverse[8] = matrix[0] * matrix[4] - matrix[1] * matrix[3];

    double den_mat = matrix[0] * inverse[0] + matrix[1] * inverse[3] + matrix[2] * inverse[6];

    // scale by the determinant
    for (int i = 0; i < 9; i++)
    {
        inverse[i] /= den_mat;
    }
}


#endif /* linalg_h */
