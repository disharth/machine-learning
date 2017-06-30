package com.disharth.mechinelearning;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

/**
 * Linar regression Predictions
 *
 */
public class LinearRegression
{
    public static void main(String[] args) {
        double[][] inputData = new double[97][2] ;
        double[][] results = new double[97][1] ;
        try {
            //Get file from resources folder
            ClassLoader classLoader = LinearRegression.class.getClassLoader();
            File file = new File(classLoader.getResource("ex1data1.txt").getFile());
            BufferedReader in = new BufferedReader(new FileReader(file));
            String str;
            int i=0;
            while ((str = in.readLine()) != null) {
                String[] ar=str.split(",");
                double x=Double.parseDouble(ar[0]);
                double y=Double.parseDouble(ar[1]);
                inputData[i][0] = 1;
                inputData[i][1] = x;
                results[i][0] = y;
                i++;
            }
            in.close();
        } catch (IOException e) {
            System.out.println("File Read Error");
        }
        RealMatrix X = MatrixUtils.createRealMatrix(inputData);
        RealMatrix y =  MatrixUtils.createRealMatrix(results);
        double[][] theta = {{0,0}};
        new GradientDescent().optimizeUsingGradientDescent(X,y,0.01,theta,1500);

    }






}
