
public class Driver1_NN {
	
	public static void main(String[] args) {
		
		String fileName = "part2/myTrain1";
		NeuralNetworkIO nnIO = new NeuralNetworkIO();
		NeuralNetworkClassifier nnc = null;
		try {
			nnc = nnIO.instantiateNNClassifierWithTrainingData(fileName);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		nnc.printRecords();
		int numberOfHiddenNodes = 3;
		int seed = 123456;
		int iterations = 1000;
		double learningRate = 0.01;
		nnc.setParameters(numberOfHiddenNodes, seed, iterations, learningRate);
		
		nnc.train();
		//nnc.initializeNetwork();
		
		nnc.letsPrintTheClassifiedRecords();
		
		nnc.printInputToHiddenWeigths();
		nnc.printOutputWeights();
		nnc.printInputToMiddleThetas();
		nnc.printOutputThetas();
	}
}

























