import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

/**
 * Created by Erik on 11/13/2015.
 */
public class DriverTraining {

    public static void main(String[] args) {

        // open data
        DataSet trainingSet = new DataSet(28, 6);
        List<String> inputlist = new ArrayList<String>();
        List<String> outputlist = new ArrayList<String>();
        File input = new File("C:\\Users\\Erik\\IdeaProjects\\CI2015Car\\classes\\input.txt");
        File output = new File("C:\\Users\\Erik\\IdeaProjects\\CI2015Car\\classes\\output.txt");
        BufferedReader reader = null;

        try {
            reader = new BufferedReader(new FileReader(input));
            String text = null;

            while ((text = reader.readLine()) != null) {
                inputlist.add(text);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
            }
        }

        try {
            reader = new BufferedReader(new FileReader(output));
            String text = null;

            while ((text = reader.readLine()) != null) {
                outputlist.add(text);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
            }
        }

        int dataPoints = inputlist.size();

        for (int i = 0; i<dataPoints; i++){
            String[] dataFeatures = inputlist.get(i).split(",");
            double[] inputDoubles = new double[dataFeatures.length];
            for(int j = 0; j<dataFeatures.length; j++){
                inputDoubles[j] = Double.parseDouble(dataFeatures[j]);
            }
            dataFeatures = outputlist.get(i).split(",");
            double[] outputDoubles = new double[dataFeatures.length];
            for(int j = 0; j<dataFeatures.length; j++){
                outputDoubles[j] = Double.parseDouble(dataFeatures[j]);
            }
            trainingSet.addRow(inputDoubles, outputDoubles);
        }
        System.out.println(inputlist.size() +" "+outputlist.size());

// create multi layer perceptron
        MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.TANH, 28, 15, 6);
        System.out.println("training...");
// learn the training set
        myMlPerceptron.learn(trainingSet);
        System.out.println("training done");

// test perceptron
        System.out.println("Testing trained neural network");
        testNeuralNetwork(myMlPerceptron, trainingSet);

// save trained neural network
        myMlPerceptron.save("myMlPerceptron.nnet");

// load saved neural network
        NeuralNetwork loadedMlPerceptron = NeuralNetwork.createFromFile("myMlPerceptron.nnet");

// test loaded neural network
        System.out.println("Testing loaded neural network");
        testNeuralNetwork(loadedMlPerceptron, trainingSet);

    }

    public static void testNeuralNetwork(NeuralNetwork nnet, DataSet testSet) {

        for(DataSetRow dataRow : testSet.getRows()) {
            nnet.setInput(dataRow.getInput());
            nnet.calculate();
            double[ ] networkOutput = nnet.getOutput();
            System.out.print("Input: " + Arrays.toString(dataRow.getInput()) );
            System.out.println(" Output: " + Arrays.toString(networkOutput) );
        }

    }
}
