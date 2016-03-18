
public class Driver1_NN {
	
	public static void main(String[] args) {
		
		String fileName = "part2/myTrain1";
		NeuralNetworkIO nnIO = new NeuralNetworkIO();
		try {
			nnIO.instantiateNNClassifierWithTrainingData(fileName);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}

























