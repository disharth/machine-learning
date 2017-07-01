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


        RealMatrix[] dataMatrix = readDataFile("ex1data1.txt" ,97 , 1);

        double[][] theta = {{0,0}};
        double[][] optimizedTheta = new GradientDescent().optimizeUsingGradientDescent(dataMatrix[0],dataMatrix[1],0.01,theta,1400);
        print(optimizedTheta);
        //now Lets use multiple features and apply normalization to it.

        int[] colsToNormalize = {1}; // index starts 0.
        RealMatrix[] dataMatrix2 = readDataFile("ex1data2.txt" ,47 , 2);

        RealMatrix X2 = MatrixUtility.scaleFeatures(dataMatrix2[0],colsToNormalize);
        RealMatrix y2 =  dataMatrix2[1];
        double[][] theta2 = {{0,0,0}};
        double[][] optimizedTheta2 = new GradientDescent().optimizeUsingGradientDescent(X2,y2,0.1,theta2,15000);
        print(optimizedTheta2);

    }

    private static void print(double[][] data){

        for (int i=0;i<data.length;i++){
            System.out.print("[");
            for(int j=0;j<data[i].length;j++){
                System.out.println(" "+data[i][j]);
            }
            System.out.print(" ]");
        }
    }

    private static RealMatrix[] readDataFile(String fileName , int numOfRecords , int numOfFeatures){
        RealMatrix[] resultMatrix = new RealMatrix[2]; // 0 features , 1 output

        double[][] inputData = new double[numOfRecords][numOfFeatures+1] ;
        double[][] results = new double[numOfRecords][1] ;
        try {
            //Get file from resources folder
            ClassLoader classLoader = LinearRegression.class.getClassLoader();
            File file = new File(classLoader.getResource(fileName).getFile());
            BufferedReader in = new BufferedReader(new FileReader(file));
            String str;
            int i=0;
            while ((str = in.readLine()) != null) {
                String[] ar=str.split(",");
                inputData[i][0] = 1;
                for (int feature=0;feature<numOfFeatures;feature++) {
                    double x = Double.parseDouble(ar[feature]);
                    inputData[i][feature + 1] = x;
                }

                double y = Double.parseDouble(ar[numOfFeatures]); // last item
                results[i][0] = y;
                i++;
            }
            in.close();
        } catch (IOException e) {
            System.out.println("File Read Error");
        }
        resultMatrix[0] = MatrixUtils.createRealMatrix(inputData);
        resultMatrix[1] =  MatrixUtils.createRealMatrix(results);
        return resultMatrix;

    }






}
