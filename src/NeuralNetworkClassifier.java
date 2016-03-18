import java.util.ArrayList;


public class NeuralNetworkClassifier {
	
	private int numberOfInputs;
	private int numberOfMiddles;
	private int numberOfOutputs;
	
	private int trainingIterations;
	private int seed;
	private double learningRate;
	
	private ArrayList<Record> records;
	
	//now for the weight and theta matrices
	private double[] input;
	private double[] middle;
	private double[] output;
	
	private double[] errorMiddle;
	private double[] errorOutput;
	
	private double[] thetaMiddle;
	private double[] thetaOutput;
	
	private double[][] weightsMiddle;
	private double[][] weightsOutput;
	
}
































