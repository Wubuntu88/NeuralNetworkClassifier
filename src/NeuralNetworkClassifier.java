import java.util.ArrayList;
import java.util.Arrays;
import java.util.TreeSet;


public class NeuralNetworkClassifier {
	
	public static final String BINARY = "binary";
	public static final String CATEGORICAL = "categorical";
	public static final String ORDINAL = "ordinal";
	public static final String CONTINUOUS = "continuous";
	public static final String LABEL = "label";
	public static final TreeSet<String> attributeTypes = new TreeSet<String>(
			Arrays.asList(BINARY, CATEGORICAL, ORDINAL, CONTINUOUS, LABEL));

	// list of the type of the variables in records (ordinal, continuous, etc)
	private ArrayList<String> headerList;
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
	private double[] output;
	
	private double[] errorMiddle;
	private double[] errorOutput;
	
	private double[] thetaMiddle;
	private double[] thetaOutput;
	
	private double[][] weightsMiddle;
	private double[][] weightsOutput;
	
}
































