package classes;

import matricesExceptions.DimensionError;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class FC {
    private final ArrayList<Matrice> dataSet, targetsSet;
    private final ArrayList<Layer> layers;
    private final Database database;

    @SuppressWarnings("ResultOfMethodCallIgnored")
    public FC(int[] shape, double learningRate, double momentumRate) throws IOException {
        this.dataSet = new ArrayList<>();
        this.targetsSet = new ArrayList<>();

        File folder = new File("FC");
        folder.mkdir();

        File biasesFile = new File("FC\\biases.txt");
        File weightsFile = new File("FC\\weights.txt");

        this.database = new Database(weightsFile, biasesFile, shape);

        this.layers = new ArrayList<>();

        for (int i = 1; i < shape.length; i++) {
            if (this.database.isWeightsFileFull() && this.database.isBiasesFileFull()) {
                this.layers.add(new Layer(shape, i, this.database.getWeigths(i), this.database.getBiases(i), learningRate, momentumRate));
            } else {
                this.layers.add(new Layer(shape, i, learningRate, momentumRate));
            }
        }
        this.toFile();
    }

    public static float abs(float x) {
        return Math.abs(x);
    }

    public static int randint(int max) {  // nombre aléatoire entier dans l'intervalle [| 0, max |]
        Random rand = new Random();

        return rand.nextInt(max + 1);
    }

    public static String from_millisecondes(long milli) {  // a simplifier avec des boucles
        String end = "";

        int heure = (int) (milli / (3.6 * Math.pow(10, 6)));
        int minute = (int) ((milli % (3.6 * Math.pow(10, 6)) / 60_000));
        int seconde = (int) ((milli % 3_600_000 % 60_000) / 1000);
        int milliseconde = (int) (milli % 1000);

        if (heure != 0) {
            end += String.format("%sh", heure);
        }
        if (minute != 0) {
            end += String.format("%sm", minute);
        }
        if (seconde != 0) {
            end += String.format("%ss", seconde);
        }
        end += String.format("%sms", milliseconde);

        return end;
    }

    private static float toPourcentage(float x) {
        return x / 100;
    }

    public ArrayList<Layer> getLayers() {
        return this.layers;
    }

    public void toFile() throws IOException {
        ArrayList<Matrice> weightsList = new ArrayList<>();
        ArrayList<Matrice> biasesList = new ArrayList<>();

        for (Layer layer : this.layers) {
            weightsList.add(layer.getWeights());
            biasesList.add(layer.getBiases());
        }

        this.database.toFile(biasesList, weightsList);
    }

    private void backPropagation(Matrice inputs, Matrice targets) throws DimensionError {
        // calcul des deltas
        for (int i = this.layers.size() - 1; i >= 0; i--) {
            if (i == this.layers.size() - 1) {
                this.layers.get(i).setOutputDeltas(targets);
            } else {
                this.layers.get(i).setDeltas(this.layers.get(i + 1).getWeights(), this.layers.get(i + 1).getDeltas());
            }
        }

        // calcul des dCosts
        for (int i = this.layers.size() - 1; i >= 0; i--) {
            if (i == 0) {
                this.layers.get(i).setdCost_dWeights(inputs);
            } else {
                this.layers.get(i).setdCost_dWeights(this.layers.get(i - 1).getOutput());
            }
        }

        // tuning
        for (Layer layer : this.layers) {
            layer.tuning();
        }

    }

    private String evaluate_time(int iterations, int frequence) throws IOException, DimensionError {
        long temp1 = System.currentTimeMillis();
        this.toFile();
        long time_toFile = System.currentTimeMillis() - temp1;

        long temp2 = System.currentTimeMillis();
        for (int i = 0; i < 200; i++) {
            int choix = randint(this.dataSet.size() - 1);
            Matrice data = this.dataSet.get(choix);
            Matrice target = this.targetsSet.get(choix);
            this.feedForward(data);
            this.backPropagation(data, target);
        }

        long time_ff_bp = (System.currentTimeMillis() - temp2) / 200;

        System.out.println(from_millisecondes(time_ff_bp * iterations));

        long time_total = time_ff_bp * iterations + time_toFile * (iterations / frequence);

        return String.format("Temps total évalué : %s", from_millisecondes(time_total));
    }

    private Matrice feedForward(Matrice data) throws DimensionError {
        for (Layer layer : this.layers) {
            data = layer.feedForward(data);
        }
        return data;
    }

    public ArrayList<Double> guess(ArrayList<ArrayList<Double>> testData) throws DimensionError {
        Matrice dataMatrice = new Matrice(testData).transpose();

        Matrice guess = feedForward(dataMatrice);

        return guess.transpose().toArrayList().get(0);
    }

    public ArrayList<Double> guess(double[] testData) throws DimensionError {
        int choix = randint(testData.length - 1);

        double[][] data_ = new double[][]{testData};
        Matrice data_matrice = new Matrice(data_).transpose();

        Matrice guess = feedForward(data_matrice);

        return guess.transpose().toArrayList().get(0);

    }

    public Matrice guess(Matrice testData) throws DimensionError {
        Matrice dataMatrice;
        if (testData.getRows() == 1) {
            dataMatrice = testData.transpose();
        } else {
            dataMatrice = testData;
        }

        Matrice guess = feedForward(dataMatrice);

        return guess.transpose();
    }

    public void setDataSet(ArrayList<ArrayList<Double>> data) {
        for (ArrayList<Double> data_ : data) {
            ArrayList<ArrayList<Double>> temp = new ArrayList<>();
            temp.add(data_);
            this.dataSet.add(new Matrice(temp).transpose());
        }
    }

    public void setDataSet(double[][] data) {
        System.out.println(Arrays.deepToString(data));
        for (double[] data_ : data) {
            this.dataSet.add(new Matrice(new double[][]{data_}).transpose());
        }
    }

    public void setTargetsSet(ArrayList<ArrayList<Double>> targets) {
        for (ArrayList<Double> target : targets) {
            ArrayList<ArrayList<Double>> temp = new ArrayList<>();
            temp.add(target);
            this.targetsSet.add(new Matrice(temp).transpose());
        }
    }

    public void setTargetsSet(double[][] targets) {
        for (double[] target_ : targets) {
            this.targetsSet.add(new Matrice(new double[][]{target_}).transpose());
        }
    }

    public void trainFromDataInObject(int iterations, int frequence) throws IOException, DimensionError {
//        System.out.println(this.evaluate_time(iterations, frequence));
        long start = System.currentTimeMillis();

        FileWriter writer = new FileWriter("errors.txt");
        for (int i = 1; i <= iterations; i++) {
            long temp = System.currentTimeMillis();
            int choix = randint(this.dataSet.size() - 1);
            Matrice data = this.dataSet.get(choix);
            Matrice target = this.targetsSet.get(choix);

            this.feedForward(data);
            this.backPropagation(data, target);

            double outputCost = this.layers.get(this.layers.size() - 1).getOutputCosts(target).getItem(0, 0);
            writer.write(String.format("%s", outputCost));
            writer.write("\n");

            if (i % frequence == 0) {
                this.toFile();
            }

            long temps = System.currentTimeMillis() - temp;

            System.out.printf("%s: %s\n", i, from_millisecondes(temps));
        }
        writer.close();

        System.out.printf("Temps total: %s\n", from_millisecondes(System.currentTimeMillis() - start));
        System.out.println(LocalTime.now());

    }

    public void trainFromDataInObjectForTime(int secondes, int frequence) throws IOException, DimensionError {
        long start = System.currentTimeMillis();

        FileWriter writer = new FileWriter("errors.txt");
        int i = 0;
        while ((System.currentTimeMillis() - start) / 1_000 < secondes) {
            long temp = System.currentTimeMillis();
            int choix = randint(this.dataSet.size() - 1);
            Matrice data = this.dataSet.get(choix);
            Matrice target = this.targetsSet.get(choix);

            this.feedForward(data);
            this.backPropagation(data, target);

            double outputCost = this.layers.get(this.layers.size() - 1).getOutputCosts(target).getItem(0, 0);
            writer.write(String.format("%s", outputCost));
            writer.write("\n");

            if (i % frequence == 0) {
                this.toFile();
            }

            long temps = System.currentTimeMillis() - temp;
            System.out.printf("%s: %s\n", i, from_millisecondes(temps));

            i++;
        }
        writer.close();

        System.out.printf("Temps total: %s\n", from_millisecondes(System.currentTimeMillis() - start));
        System.out.println(LocalTime.now());
    }

    public void trainFromDataInObjectWhileCostAboveMax(double maxCost) throws IOException, DimensionError {
//        System.out.println(this.evaluate_time(iterations, frequence));
        long start = System.currentTimeMillis();

        FileWriter writer = new FileWriter("errors.txt");
        int nbIterations = 1;
        double cost = 1;

        while (cost > maxCost) {
            long temp = System.currentTimeMillis();
            int choix = randint(this.dataSet.size() - 1);
            Matrice data = this.dataSet.get(choix);
            Matrice target = this.targetsSet.get(choix);

            this.feedForward(data);
            this.backPropagation(data, target);

            cost = this.layers.get(this.layers.size() - 1).getOutputCosts(target).getItem(0, 0);

            writer.write(String.format("%s", cost));
            writer.write("\n");

            long temps = System.currentTimeMillis() - temp;

            System.out.printf("%s: %s\n", nbIterations, from_millisecondes(temps));

            nbIterations += 1;
        }

        writer.close();

        this.toFile();

        System.out.printf("Temps total: %s\n", from_millisecondes(System.currentTimeMillis() - start));
        System.out.println(LocalTime.now());

    }

    public long trainFromExternalData(Matrice dataMatrice, Matrice targetMatrice, int iteration, int frequence) throws IOException, DimensionError {
        long start = System.currentTimeMillis();

        this.feedForward(dataMatrice);
        this.backPropagation(dataMatrice, targetMatrice);

        if (iteration % frequence == 0) {
            this.toFile();
        }

        long temps = System.currentTimeMillis() - start;

        System.out.printf("%s: %s\n", iteration, from_millisecondes(temps));

        return temps;
    }
}
