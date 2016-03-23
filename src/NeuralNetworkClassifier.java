import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeSet;


public class NeuralNetworkClassifier {

	// list of the type of the variables in records (ordinal, continuous, etc)
	//private ArrayList<String> headerList;
	private ArrayList<Record> records;
	
	private int numberOfInputs;
	private int numberOfMiddles;
	private int numberOfOutputs;
	
	private int trainingIterations;
	private int seed;
	private double learningRate;
	
	//now for the weight and theta matrices
	private double[] input;
	private double[] middle;
	private double output;
	
	private double[] errorMiddle;
	private double errorOutput;
	
	private double[] thetaMiddle;
	private double thetaOutput;
	
	private double[][] weightsMiddle;
	private double[] weightsOutput;
	
	public NeuralNetworkClassifier(ArrayList<Record> records) {
		this.records = records;
		this.numberOfMiddles = 3;
		this.seed = 478239;
		this.trainingIterations = 10000;
		this.learningRate = 0.5;
	}
	
	private boolean hasSetParameters = false;
	/**
	 * Allows the user to set the parameters of the neural network other than the default.
	 * Only allows user to set the parameters once to avoid shenanigans from the user.
	 * @param numberOfHiddenNodes
	 * @param seed
	 * @param iterations
	 * @param learningRate
	 * @return
	 */
	public boolean setParameters(int numberOfHiddenNodes, int seed, int iterations, double learningRate){
		if(hasSetParameters == false){
			this.numberOfMiddles = numberOfHiddenNodes;
			this.seed = seed;
			this.trainingIterations = iterations;
			this.learningRate = learningRate;
			return true;
		}else{
			return false;
		}
	}
	/**
	 * Initialized the weights and the theta values of the network to random values
	 * between -1 and 1
	 */
	public void initializeNetwork(){
		Random rng = new Random(seed);
		
		this.input = new double[this.numberOfInputs];
		this.middle = new double[this.numberOfMiddles];
		this.output = 0;
		
		this.errorMiddle = new double[this.numberOfMiddles];
		this.errorOutput = 0.0;
		
		//initialize theta middle and theta out to random values between -1 and 1
		this.thetaMiddle = new double[this.numberOfMiddles];
		for(int i = 0; i < this.thetaMiddle.length; i++){
			this.thetaMiddle[i] = 2 * rng.nextDouble() - 1;
		}
		this.thetaOutput = 2 * rng.nextDouble() - 1;
		
		//initialize weights between input and middle (hidden) nodes
		for(int input_i = 0; input_i < this.numberOfInputs; input_i++){
			for(int middle_j = 0; middle_j < this.numberOfMiddles; middle_j++){
				weightsMiddle[input_i][middle_j] = 2 * rng.nextDouble() - 1;
			}
		}
		
		//initialize weights between middle (hidden) and output node (only one output node)
		for(int middle_j = 0; middle_j < this.numberOfMiddles; middle_j++){
			weightsOutput[middle_j] = 2 * rng.nextDouble() - 1;
		}
	}
	
	public void train(){
		for(int iteration = 0; iteration < this.trainingIterations; iteration++){
			for(Record record: this.records){
				feedForward(record.getAttrList());
				backPropagate(record.getLabel());
			}
		}
	}
	
	private void feedForward(double[] attrList) {
		// TODO Auto-generated method stub
		
	}
	
	private void backPropagate(double label) {
		// TODO Auto-generated method stub
		
	}
	
	public void printRecords(){
		for(Record record : this.records){
			System.out.println(record);
		}
	}
}
































