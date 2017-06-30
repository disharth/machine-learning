package com.disharth.mechinelearning;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * Implements Hypothesis
 */
public class Hypothesis {

    /**
     * Hypothesis for Linear regression is hx = theta(0) + theta(1)*X(1) ....
     */
    public static RealMatrix getLinearHypothesis(RealMatrix X , double[][] theta){

        RealMatrix thetaMatrix = MatrixUtils.createRealMatrix(theta);
        RealMatrix hypothesis = X.multiply(thetaMatrix.transpose());
        return hypothesis;
    }
}
