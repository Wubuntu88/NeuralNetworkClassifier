
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
	}
}

























