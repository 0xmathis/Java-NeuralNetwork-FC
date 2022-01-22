package Classes;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;


public class Database {
    private final File weightsFile;
    private final File biasesFile;
    private final int[] shape;
    private final ArrayList<Matrice> weightsArray;
    private final ArrayList<Matrice> biasesArray;


    public Database(File weightsFile, File biasesFile, int[] shape) throws IOException {
        this.weightsFile = weightsFile;
        this.biasesFile = biasesFile;
        this.shape = shape;

        if (this.biasesFile.createNewFile() && this.weightsFile.createNewFile()) { // si on créé le fichier
            this.biasesArray = new ArrayList<>();
            this.weightsArray = new ArrayList<>();
        } else {
            this.biasesArray = this.getBiasesArrayFromFile(biasesFile);
            this.weightsArray = this.getWeightsArrayFromFile(weightsFile);
        }

    }

    public static ArrayList<ArrayList<Float>> dataFromFile(File file) throws FileNotFoundException {
        Scanner reader = new Scanner(file);
        ArrayList<ArrayList<Float>> end = new ArrayList<>();

        while (reader.hasNextLine()) {
            String[] line = reader.nextLine().split(" ");
            ArrayList<Float> line_ = new ArrayList<>();
            for (String x : line) {
                line_.add(Float.parseFloat(x));
            }
            end.add(line_);
        }

        return end;
    }

    public void biasesMatricesToFile(ArrayList<Matrice> matriceArrayList) throws IOException {
        FileWriter writer = new FileWriter(this.biasesFile);

        for (Matrice matrice : matriceArrayList) {
            for (int i = 0; i < matrice.getRows(); i++) {
                writer.write(String.format("%s", matrice.getItem(i, 0)));
                if (matrice != matriceArrayList.get(matriceArrayList.size() - 1) || i != matrice.getRows() - 1) {
                    writer.write("\n");
                }
            }
        }

        writer.close();
    }

    public void weightsMatricesToFile(ArrayList<Matrice> matriceArrayList) throws IOException {
        FileWriter writer = new FileWriter(this.weightsFile);

        for (Matrice matrice : matriceArrayList) {
            for (int i = 0; i < matrice.getRows(); i++) {
                for (int j = 0; j < matrice.getColumns(); j++) {
                    writer.write(String.format("%s", matrice.getItem(i, j)));
                    if (matrice != matriceArrayList.get(matriceArrayList.size() - 1) || i != matrice.getRows() - 1 || j != matrice.getColumns() - 1) {
                        writer.write("\n");
                    }
                }
            }
        }

        writer.close();
    }

    public Matrice getBiases(int toLayer) {
        return this.biasesArray.get(toLayer - 1);
    }

    private ArrayList<Matrice> getBiasesArrayFromFile(File file) throws FileNotFoundException {
        Scanner scanner = new Scanner(file);
        ArrayList<Matrice> allBiases = new ArrayList<>();

        for (int n = 1; n < this.shape.length; n++) {
            Matrice matrice = Matrice.vide(this.shape[n], 1);

            for (int i = 0; i < this.shape[n]; i++) {
                double data = Double.parseDouble(scanner.nextLine());
                matrice.setItem(i, 0, data);
            }
            allBiases.add(matrice);
        }

        return allBiases;
    }

    public Matrice getWeigths(int toLayer) {
        return this.weightsArray.get(toLayer - 1);
    }

    private ArrayList<Matrice> getWeightsArrayFromFile(File file) throws FileNotFoundException {
        Scanner scanner = new Scanner(file);
        ArrayList<Matrice> allWeights = new ArrayList<>();

        for (int n = 1; n < this.shape.length; n++) {
            Matrice matrice = Matrice.vide(this.shape[n], this.shape[n - 1]);

            for (int i = 0; i < this.shape[n]; i++) {
                for (int j = 0; j < this.shape[n - 1]; j++) {
                    double data = Double.parseDouble(scanner.nextLine());
                    matrice.setItem(i, j, data);
                }
            }
            allWeights.add(matrice);
        }

        return allWeights;
    }

    public boolean isBiasesFileFull() throws FileNotFoundException {
        Scanner reader = new Scanner(this.biasesFile);
        return reader.hasNext();
    }

    public boolean isWeightsFileFull() throws FileNotFoundException {
        Scanner reader = new Scanner(this.weightsFile);
        return reader.hasNext();
    }

    public void toFile(ArrayList<Matrice> biasesList, ArrayList<Matrice> weightsList) throws IOException {
        this.biasesMatricesToFile(biasesList);
        this.weightsMatricesToFile(weightsList);
    }

}
