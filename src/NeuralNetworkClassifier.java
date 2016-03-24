import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.TreeSet;


public class NeuralNetworkClassifier {

	// list of the type of the variables in records (ordinal, continuous, etc)
	//private ArrayList<String> headerList;
	private ArrayList<Record> records;
	private TreeSet<Double> crypticLabels = new TreeSet<>();
	
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
	
	private double[][] weightsMiddle;//[input][hidden]
	private double[] weightsOutput;//[hidden]
	
	public NeuralNetworkClassifier(ArrayList<Record> records) {
		this.records = records;
		if(records.size() > 0){
			this.numberOfInputs = records.get(0).getAttrList().length;
			for(Record record: this.records){
				if(crypticLabels.contains(record.getLabel()) == false){
					crypticLabels.add(record.getLabel());
				}
			}
			System.out.println("number of cryptic labels: " + crypticLabels.size());
		}else{
			System.out.println("cannot build a neural net with zero records.");
		}
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
	
	boolean hasInitializedNeuralNetwork = false;
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
		
		this.weightsMiddle = new double[this.numberOfInputs][this.numberOfMiddles];
		//initialize weights between input and middle (hidden) nodes
		for(int input_i = 0; input_i < this.numberOfInputs; input_i++){
			for(int middle_j = 0; middle_j < this.numberOfMiddles; middle_j++){
				weightsMiddle[input_i][middle_j] = 2 * rng.nextDouble() - 1;
			}
		}
		
		this.weightsOutput = new double[this.numberOfMiddles];
		//initialize weights between middle (hidden) and output node (only one output node)
		for(int middle_j = 0; middle_j < this.numberOfMiddles; middle_j++){
			weightsOutput[middle_j] = 2 * rng.nextDouble() - 1;
		}
	}
	
	public void train(){
		if(hasInitializedNeuralNetwork == false){
			this.initializeNetwork();
			hasInitializedNeuralNetwork = true;
		}
		for(int iteration = 0; iteration < this.trainingIterations; iteration++){
			for(Record record: this.records){
				
				feedForward(record.getAttrList());
				backPropagate(record.getLabel());
			}
		}
	}
	
	private void feedForward(double[] attrList) {
		//feed inputs of record
		for(int i = 0; i < attrList.length; i++){
			this.input[i] = attrList[i];
		}
		
		//for each hidden node, get the sum of each input times the weight into that hidden node
		for(int hiddenIndex = 0; hiddenIndex < this.numberOfMiddles; hiddenIndex++){
			double sum = 0.0; //weighted sum of inputs into hidden node
			//must loop through the inputs
			for(int inputIndex = 0; inputIndex < this.numberOfInputs; inputIndex++){
				sum += this.input[inputIndex] * this.weightsMiddle[inputIndex][hiddenIndex];
			}
			sum += this.thetaMiddle[hiddenIndex]; // sum will be part of exponent in normalized softmax function
			//compute output at hidden node
			this.middle[hiddenIndex] = 1 / (1 + Math.exp(-sum));
		}
		
		//calculate the output for the output node
		double outputSum = 0.0;
		//compute inputs to the output node
		for(int hiddenIndex = 0; hiddenIndex < this.numberOfMiddles; hiddenIndex++){
			outputSum += this.middle[hiddenIndex] * this.weightsOutput[hiddenIndex];
		}
		//add theta value to summ
		outputSum += this.thetaOutput;
		this.output = 1 / (1 + Math.exp(-outputSum));
	}
	
	private void backPropagate(double label) {
		//compute error at the output node
		this.errorOutput = this.output * (1 - this.output) * (label - this.output);
		//this.errorOutput = this.output * (1 - this.output) * (this.output - label);
		//compute error at each hidden node
		for(int hiddenIndex = 0; hiddenIndex < this.numberOfMiddles; hiddenIndex++){
			double errorAtHiddenNode = this.errorOutput * weightsOutput[hiddenIndex];
			this.errorMiddle[hiddenIndex] = 
					this.middle[hiddenIndex] * (1 - this.middle[hiddenIndex]) * errorAtHiddenNode;
		}
		
		//update weights between hidden/output nodes
		for(int hiddenIndex = 0; hiddenIndex < this.numberOfMiddles; hiddenIndex++){
			weightsOutput[hiddenIndex] += this.learningRate * middle[hiddenIndex] * this.errorOutput;
		}
		
		//update weights between input/output nodes
		for(int inputsIndex = 0; inputsIndex < this.numberOfInputs; inputsIndex++){
			for(int hiddenIndex = 0; hiddenIndex < this.numberOfMiddles; hiddenIndex++){
				this.weightsMiddle[inputsIndex][hiddenIndex] += 
						this.learningRate * this.input[inputsIndex] * this.errorMiddle[hiddenIndex];
			}
		}
		
		//update the theta at the output node
		this.thetaOutput += this.learningRate * this.errorOutput;
		
		//update the thetas at the hidden nodes
		for(int hiddenIndex = 0; hiddenIndex < this.numberOfMiddles; hiddenIndex++){
			this.thetaMiddle[hiddenIndex] += this.learningRate * this.errorMiddle[hiddenIndex];
		}
	}
	
	public ArrayList<Double> classifyRecords(ArrayList<Record> records){
		ArrayList<Double> classifications = new ArrayList<>();
		for(Record record: records){
			double classification = classify(record);
			classifications.add(classification);
		}
		return classifications;
	}
	
	private double classify(Record record) {
		feedForward(record.getAttrList());//result of feed foreward will be in this.output
		return this.output;
	}
	
	public double calculateTrainingError(){
		int numberOfMisclassifiedRecords = 0;
		for(Record record: this.records){
			double classification = classify(record);
			double closestValue = findClosestCrypticLabelForClassification(classification);
			if((closestValue == record.getLabel()) == false){
					// || Math.abs(closestValue - record.getLabel()) > .00001){
				/*
				System.out.println("actual value: " + theClass);
				System.out.println("closest Value: " + closestValue);
				System.out.println("records label: " + record.getLabel());
				System.out.println();
				*/
				numberOfMisclassifiedRecords++;
			}
		}
		return (double)numberOfMisclassifiedRecords / this.records.size();
	}
	
	public double findClosestCrypticLabelForClassification(double classification){
		double minDistance = Double.MAX_VALUE;
		double closestValue = -1.0;
		for(Double value: crypticLabels){
			double dist = Math.abs(classification - value);
			if(dist < minDistance){
				closestValue = value;
				minDistance = dist;
			}
		}
		return closestValue;
	}
	
	
	public void printInputToHiddenWeigths(){
		System.out.println("InputToHiddenWeigths:");
		for(int i = 0; i < this.numberOfInputs; i++){
			for(int j = 0; j < this.numberOfMiddles; j++){
				System.out.println("i: " + i + ", j: " + j + ", value: " + this.weightsMiddle[i][j]);
			}
		}
	}
	public void printOutputWeights(){
		System.out.println("output weights: ");
		for(int i = 0; i < this.numberOfMiddles; i ++){
			System.out.println("i: " + i + ", value: " + this.weightsOutput[i]);
		}
	}
	public void printInputToMiddleThetas(){
		System.out.println("Input to middle thetas:");
		for(int i = 0; i < this.numberOfMiddles; i ++){
			System.out.println("i: " + i + ", value: " + this.thetaMiddle[i]);
		}
	}
	public void printOutputThetas(){
		System.out.println("output theta: " + this.thetaOutput);
	}

	public void printRecords(){
		for(Record record : this.records){
			System.out.println(record);
		}
	}
	
}
































