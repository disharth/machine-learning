package com.disharth.mechinelearning;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import static com.disharth.mechinelearning.MatrixUtility.elementwiseMultiply;
import static com.disharth.mechinelearning.MatrixUtility.sum;

/**
 * Implements Gradient Descent.
 */
public class GradientDescent {

    public double[][] optimizeUsingGradientDescent(RealMatrix X , RealMatrix y, double lambda , double[][] theta , int ITERATIONS){

        int dataSize = X.getRowDimension();
        int maxRowIndex = dataSize - 1;
        int featureSize = X.getColumnDimension();
        for(int i=0;i<ITERATIONS;i++) {

            RealMatrix hx = Hypothesis.getLinearHypothesis(X , theta);
            RealMatrix diff = hx.subtract(y);
            double sumOfDiff = sum(diff);
            double[][] newTheta = new double[1][featureSize];
            newTheta[0][0] = theta[0][0] - (lambda / dataSize) * sumOfDiff;
            for(int f=1;f<featureSize;f++) {
                int[] colInd = {f};
                RealMatrix x = X.getSubMatrix(0, maxRowIndex, f,f);
                sumOfDiff = sum(elementwiseMultiply(diff,x));
                newTheta[0][f] = theta[0][f] - (lambda / dataSize) * sumOfDiff;
            }

            theta = newTheta;

        }
        System.out.println("Final Theta ["+theta[0][0]+" "+theta[0][1]+"]");

        return theta;
    }



}
