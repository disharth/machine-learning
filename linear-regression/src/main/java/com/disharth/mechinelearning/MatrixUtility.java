package com.disharth.mechinelearning;

import org.apache.commons.math3.linear.RealMatrix;

/**
 * Some Util methods.
 */
public class MatrixUtility {

    public static RealMatrix elementwiseMultiply(RealMatrix sourceMatrix , RealMatrix targetMatrix) {

        int numRows = sourceMatrix.getRowDimension();
        int numCols = sourceMatrix.getColumnDimension();

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                sourceMatrix.multiplyEntry(i,j , targetMatrix.getEntry(i, j));
            }
        }
        return sourceMatrix;


    }
    public static double sum(RealMatrix m) {
        int numRows = m.getRowDimension();
        int numCols = m.getColumnDimension();
        double sum = 0;
        // loop through the rows and compute the sum
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                sum += m.getEntry(i, j);
            }
        }
        return sum;
    }

}
