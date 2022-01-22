import Classes.NeuralNetwork;
import MatricesExceptions.DimensionError;

import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException, DimensionError {
        double[][] inputs = new double[][]{{0, 0}, {1, 1}, {0, 1}, {1, 0}};
        double[][] targets = new double[][]{{1}, {1}, {0}, {0}};

        NeuralNetwork nn = new NeuralNetwork(new int[]{2, 2, 1}, 0.6, 0.9, new File("Weights.txt"), new File("Biases.txt"));
        nn.setDataSet(inputs);
        nn.setTargetsSet(targets);

//        nn.trainFromDataInObjectWhileCostAboveMax(1e-4);
//        nn.trainFromDataInObject(1_000_000, 1_000_000);
        nn.batchTrainFromDataInObject(10);

        for (double[] input : inputs) {
            System.out.println(nn.guess(input));
            System.out.println();
        }
    }
}
