package com.disharth.mechinelearning;


import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import static com.disharth.mechinelearning.MatrixUtility.elementwiseMultiply;
import static com.disharth.mechinelearning.MatrixUtility.columwiseSum;

/**
 * Implements Gradient Descent.
 */
public class GradientDescent {

    public double[][] optimizeUsingGradientDescent(RealMatrix X , RealMatrix y, double lambda , double[][] theta , int ITERATIONS){

        int dataSize = X.getRowDimension();
        int featureSize = X.getColumnDimension();
        for(int i=0;i<ITERATIONS;i++) {

            RealMatrix hx = Hypothesis.getLinearHypothesis(X , theta);
            RealMatrix diff = hx.subtract(y);
            double[] sumOfDiff = columwiseSum(elementwiseMultiply(diff,X));
            double[][] newTheta = new double[1][featureSize];
            for(int f=0;f<featureSize;f++) {
                newTheta[0][f] = theta[0][f] - (lambda / dataSize) * sumOfDiff[f];
            }

            theta = newTheta;

        }

        return theta;
    }



}
