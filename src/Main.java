import classes.FC;
import classes.Matrice;
import matricesExceptions.DimensionError;

import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException, DimensionError {
//        double[][] inputs = new double[][]{{0, 0}, {1, 1}, {0, 1}, {1, 0}};
//        double[][] targets = new double[][]{{1}, {1}, {0}, {0}};
//
//        FC nn = new FC(new int[]{2, 2, 1}, 0.6, 0.9, new File("Weights.txt"), new File("Biases.txt"));
//        nn.setDataSet(inputs);
//        nn.setTargetsSet(targets);

//        nn.trainFromDataInObjectWhileCostAboveMax(1e-4);
//        nn.stochasticTrainFromDataInObject(1_000_000, 1_000_000);
//        nn.batchTrainFromDataInObject(10);

//        for (double[] input : inputs) {
//            System.out.println(nn.guess(input));
//            System.out.println();
//        }

        FC nn = new FC(new int[]{180 * 166, 30, 15, 15, 15, 2}, 0.5, 1, new File("Weights.txt"), new File("Biases.txt"));

        nn.stochasticTrainFromExternalData(Matrice.random(180 * 166, 1, - 5, 5), Matrice.random(2, 1, - 3, 3), 0, 400);

        long time = 0;
        for (int i = 0; i < 100; i++) {
            Matrice input = Matrice.random(180 * 166, 1, - 5, 5);
            Matrice target = Matrice.random(2, 1, - 3, 3);

            time += nn.stochasticTrainFromExternalData(input, target, i + 1, 400);
        }
        System.out.println(time / 100);
    }
}
