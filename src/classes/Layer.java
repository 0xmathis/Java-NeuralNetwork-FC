package classes;

import matricesExceptions.DimensionError;

public class Layer {
    private final int[] shape;
    private final int column;
    private final double learningRate;
    private final double momentumRate;
    private Matrice weights;
    private Matrice biases;
    private Matrice input;
    private Matrice weightedSum;
    private Matrice output;
    private Matrice deltas;
    private Matrice dCost_dWeights;
    private Matrice previousDeltaBiases;
    private Matrice previousDeltaWeights;
    private Matrice sumDeltaBiases;
    private Matrice sumDeltaWeights;

    public Layer(int[] shape, int column, double learningRate, double momentumRate) {  // si aucun poids n'est enregistré dans le fichier
        this.shape = shape;
        this.column = column;

        this.learningRate = learningRate;
        this.momentumRate = momentumRate;

        this.weights = Matrice.random(shape[column], shape[column - 1], - 1, 1);
        this.biases = Matrice.random(shape[column], 1, - 1, 1);

        this.input = Matrice.vide(shape[column - 1], 1);
        this.weightedSum = Matrice.vide(shape[column - 1], 1);
        this.output = Matrice.vide(shape[column], 1);
        this.deltas = Matrice.vide(shape[column], 1);
        this.dCost_dWeights = Matrice.vide(shape[column], shape[column - 1]);
        this.previousDeltaBiases = Matrice.vide(shape[column], 1);
        this.previousDeltaWeights = Matrice.vide(shape[column], shape[column - 1]);
        this.sumDeltaBiases = Matrice.vide(shape[column], 1);
        this.sumDeltaWeights = Matrice.vide(shape[column], shape[column - 1]);
    }

    public Layer(int[] shape, int column, Matrice weights, Matrice biases, double learningRate, double momentumRate) {
        this.shape = shape;
        this.column = column;

        this.learningRate = learningRate;
        this.momentumRate = momentumRate;

        this.weights = weights;
        this.biases = biases;

        this.input = Matrice.vide(shape[column - 1], 1);
        this.weightedSum = Matrice.vide(shape[column - 1], 1);
        this.output = Matrice.vide(shape[column], 1);
        this.deltas = Matrice.vide(shape[column], 1);
        this.dCost_dWeights = Matrice.vide(shape[column], shape[column - 1]);
        this.previousDeltaBiases = Matrice.vide(shape[column], 1);
        this.previousDeltaWeights = Matrice.vide(shape[column], shape[column - 1]);
        this.sumDeltaBiases = Matrice.vide(shape[column], 1);
        this.sumDeltaWeights = Matrice.vide(shape[column], shape[column - 1]);
    }

    public static double d_sigmoid_dx(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(- x));
    }

    public Matrice feedForward(Matrice data) throws DimensionError {
        this.input = data.copy();
        this.weightedSum = this.weights.mul(this.input).add(this.biases);
        this.output = this.weightedSum.map(Layer::sigmoid);

        return this.output;
    }

    public Matrice getBiases() {
        return this.biases;
    }

    public Matrice getDeltas() {
        return this.deltas;
    }

    public Matrice getOutput() {
        return this.output;
    }

    public Matrice getOutputCosts(Matrice targets) throws DimensionError {
        Matrice error = this.output.sub(targets);

        return error.hp(error).mul(0.5);
    }

    public Matrice getWeightedSum() {
        return this.weightedSum;
    }

    public Matrice getWeights() {
        return this.weights;
    }

    public void setdCost_dWeights(Matrice previousLayerOutput) throws DimensionError {
        this.dCost_dWeights = this.deltas.mul(previousLayerOutput.transpose());
    }

    public void setDeltas(Matrice nextLayerWeights, Matrice nextLayerDeltas) throws DimensionError {
        this.deltas = (nextLayerWeights.transpose().mul(nextLayerDeltas)).hp(this.weightedSum.map(Layer::d_sigmoid_dx));
    }

    public void setOutputDeltas(Matrice target) throws DimensionError {
        Matrice errors = this.output.sub(target);
        this.deltas = errors.hp(this.weightedSum.map(Layer::d_sigmoid_dx)).mul(0.5);
    }

    public void stochasticTuning() throws DimensionError {
        Matrice momentumBiases = this.previousDeltaBiases.mul(1 - this.momentumRate);
        Matrice momentumWeights = this.previousDeltaWeights.mul(1 - this.momentumRate);

        this.biases = this.biases.sub(this.deltas.mul(this.learningRate * this.momentumRate));
        this.biases = this.biases.add(momentumBiases);
        this.weights = this.weights.sub(this.dCost_dWeights.mul(this.learningRate * this.momentumRate));
        this.weights = this.weights.add(momentumWeights);

        this.previousDeltaBiases = this.deltas.copy();
        this.previousDeltaWeights = this.dCost_dWeights.copy();
    }

    public void batchSaveDeltas() throws DimensionError {
        this.sumDeltaBiases = this.sumDeltaBiases.add(this.deltas);
        this.sumDeltaWeights = this.sumDeltaWeights.add(this.dCost_dWeights);
    }

    public void batchTuning() throws DimensionError {
        this.biases = this.biases.sub(this.sumDeltaBiases.mul(this.learningRate));
        this.weights = this.weights.sub(this.sumDeltaWeights.mul(this.learningRate));

    }

    public void batchResetVariables() {
        this.sumDeltaBiases = Matrice.vide(shape[column], 1);
        this.sumDeltaWeights = Matrice.vide(shape[column], shape[column - 1]);
    }

}
