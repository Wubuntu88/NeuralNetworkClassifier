import java.util.ArrayList;

public class Driver1_NN {
	
	public static void main(String[] args) {
		
		String trainingFileName = "part2/myTrain1";
		String testFileName = "part2/test1";
		String validationFileName = "part2/validate1";
		String validationOutputFileName = "part2/validationOutput1";
		String outputFile = "part2/output1";
		NeuralNetworkIO nnIO = new NeuralNetworkIO();
		NeuralNetworkClassifier nnc = null;
		try {
			nnc = nnIO.instantiateNNClassifierWithTrainingData(trainingFileName);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//nnc.printRecords();
		int numberOfHiddenNodes = 4;
		int seed = 123456;
		int iterations = 10000;
		double learningRate = .7;
		System.out.println("part2 of neural network");
		System.out.println("parametets:");
		System.out.println("# hidden nodes:" + numberOfHiddenNodes);
		System.out.println("seed: " + seed);
		System.out.println("iterations: " + iterations);
		System.out.println("learning rate: " + .7);
		nnc.setParameters(numberOfHiddenNodes, seed, iterations, learningRate);
		nnc.train();
		
		/* A Classification of records in the test file */
		ArrayList<Record> testRecords = null;
		try {
			testRecords = nnIO.readTestRecordsFromFile(testFileName);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		ArrayList<Double> classifications = nnc.classifyRecords(testRecords);
		nnIO.writeRecordsToFile(outputFile, testRecords, classifications);
		/* B computing training error of file */
		double trainError = nnc.calculateTrainingError();
		System.out.println("training error: " + trainError);
		
		/* C validation error */
		ArrayList<Record> validationRecords = null;
		try {
			validationRecords = nnIO.readTestRecordsFromFile(validationFileName);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		ArrayList<Double> valClassifications = nnc.classifyRecords(validationRecords);
		double validationError = nnIO.findValidationErrorRate(validationRecords, valClassifications);
		System.out.println("validation error: " + validationError);
		
		nnIO.writeRecordsToFile(validationOutputFileName, validationRecords, valClassifications);

		/* D printing the weights and the theta values */
		nnc.printInputToHiddenWeigths();
		nnc.printOutputWeights();
		nnc.printInputToMiddleThetas();
		nnc.printOutputThetas();
	}
}

























