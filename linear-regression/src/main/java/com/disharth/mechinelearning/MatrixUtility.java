package com.disharth.mechinelearning;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;


/**
 * Some Util methods.
 */
public class MatrixUtility {

    public static RealMatrix elementwiseMultiply(RealMatrix sourceMatrix , RealMatrix targetMatrix) {

        RealMatrix resultMatrix = targetMatrix.copy();

        int numRows = sourceMatrix.getRowDimension();
        int targetCols = targetMatrix.getColumnDimension();

        for (int i = 0; i < numRows; i++) {
            double multiplier = sourceMatrix.getEntry(i,0);
            for (int j = 0; j < targetCols; j++) {
                resultMatrix.multiplyEntry(i,j , multiplier);
            }
        }
        return resultMatrix;


    }
    public static double[] columwiseSum(RealMatrix m) {
        int numRows = m.getRowDimension();
        int numCols = m.getColumnDimension();
        double[] sums = new double[numCols];
        double sum = 0;
        // loop through the rows and compute the sum
        for (int j = 0; j < numCols; j++) {
            sum = 0;
            for (int i = 0; i < numRows; i++) {
                sum += m.getEntry(i, j);
            }
            sums[j] = sum;
        }
        return sums;
    }

    /**
     * Scales elements of a matrix between -1 to 1.
     * @param sourceMatrix
     * @param cols - which columns to scale
     * @return Scaled matrix
     */

    public static RealMatrix scaleFeatures(RealMatrix sourceMatrix , int[] cols){

        RealMatrix realMatrix = sourceMatrix.copy();

        for(int i=0;i<cols.length;i++) {
            int column = cols[i];
            realMatrix.setColumn(column ,StatUtils.normalize(realMatrix.getColumn(column)));

        }

        return realMatrix;

    }



}
