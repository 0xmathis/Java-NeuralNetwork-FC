package neuralNetworkExceptions;

public class MatriceError extends Exception {
    public MatriceError(String error, String matrice, int badDim, int goodDim) {
        super(String.format("Le nombre de %s de la matrice des %s, ici %s au lieu de %s", error, matrice, badDim, goodDim));
    }
}
