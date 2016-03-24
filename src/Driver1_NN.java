import java.util.ArrayList;

public class Driver1_NN {
	
	public static void main(String[] args) {
		
		String trainingFileName = "part2/myTrain1";
		String testFileName = "part2/test1";
		String outputFile = "part2/output1";
		NeuralNetworkIO nnIO = new NeuralNetworkIO();
		NeuralNetworkClassifier nnc = null;
		try {
			nnc = nnIO.instantiateNNClassifierWithTrainingData(trainingFileName);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		nnc.printRecords();
		int numberOfHiddenNodes = 4;
		int seed = 123456;
		int iterations = 10000;
		double learningRate = .7;
		nnc.setParameters(numberOfHiddenNodes, seed, iterations, learningRate);
		
		nnc.train();
		double trainError = nnc.calculateTrainingError();
		System.out.println("training error: " + trainError);
		
		ArrayList<Record> testRecords = null;
		try {
			testRecords = nnIO.readTestRecordsFromFile(testFileName);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		ArrayList<Double> classifications = nnc.classifyRecords(testRecords);
		for(Record record: testRecords){
			System.out.println(record);
		}
		for(Double d: classifications){
			System.out.println(d);
		}
		
		nnIO.writeRecordsToFile(outputFile, testRecords, classifications);
		
		//nnc.initializeNetwork();
		
		/*
		nnc.printInputToHiddenWeigths();
		nnc.printOutputWeights();
		nnc.printInputToMiddleThetas();
		nnc.printOutputThetas();
		*/
	}
}

























