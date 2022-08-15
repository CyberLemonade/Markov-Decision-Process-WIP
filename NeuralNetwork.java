public class NeuralNetwork
{
    final int[] LAYERS;
    final float LEARNING_RATE = 0.001f;
    
    Matrix weights[];
    Vector values[];
    Vector errors[];
    
    NeuralNetwork(final int[] LAYERS)
    {
        this.LAYERS = LAYERS;
        this.weights = new Matrix[LAYERS.length - 1];
        for (int i = 0; i < LAYERS.length -1; i++)
        {
            weights[i] = new Matrix(LAYERS[i+1], LAYERS[i], -1.0f, 1.0f);
        }
    }
    
    public NeuralNetwork clone()
    {
        NeuralNetwork clone = new NeuralNetwork(this.LAYERS);
        for (int i = 0; i < LAYERS.length -1; i++)
        {
            clone.weights[i] = this.weights[i].clone();
        }
        return clone;
    }
    
    void debug()
    {
        for (int i = 0; i < LAYERS.length - 1; i++) {
            System.err.println("weights["+i+"] = ");
            System.err.println(weights[i]);
        }
    }
    
    float[] guess(final float[] inputs)
    {
        this.values = new Vector[LAYERS.length];
        this.values[0] = new Vector(inputs);
        feedForward();
        return values[LAYERS.length-1].toArray();
    }
    
    void train(final float[] inputs, final float[] targets)
    {
        this.values = new Vector[LAYERS.length];
        this.errors = new Vector[weights.length];
        float guesses[] = guess(inputs);
        this.errors[errors.length-1] = (Vector) Matrix.add(new Vector(targets), Matrix.scale(new Vector(guesses), -1.0f));
        backPropagate();
        tweak();
    }
    
    void feedForward()
    {
        for (int i = 0; i < values.length - 1; i++)
        {
            values[i] = (Vector) Matrix.addRow(values[i]);
            values[i+1] = (Vector) Matrix.product(weights[i], values[i]);
            values[i+1].sigmoid();
        }
    }

    void backPropagate()
    {
        for (int i = errors.length - 1; i > 0; i--)
        {
            errors[i-1] = (Vector) Matrix.product(weights[i].transpose(),errors[i]);
        }
    }
    
    void tweak()
    {
        errors[errors.length - 1] = (Vector) Matrix.addRow(errors[errors.length - 1]);
        values[values.length - 1] = (Vector) Matrix.addRow(values[values.length - 1]);
        
        for (int i = 0; i < weights.length; i++)
        {
            Matrix gradient = errors[i].clone();
            Matrix derivative = Matrix.dsigmoid(values[i+1]);
            gradient.multiply(derivative);
            gradient.scale(LEARNING_RATE);
            gradient = Matrix.delRow(gradient);

            Matrix delta = Matrix.product(gradient, Matrix.transpose(values[i]));
        
            weights[i].add(delta);
        }
    }
}