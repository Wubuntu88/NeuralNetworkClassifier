import java.util.ArrayList;
public class RegressionPart1_Driver {
	public static void main(String[] args) {
		String fileName = "part3/train1";
		String testFileName = "part3/test1";
		String testOutputName = "part3/testOutput1";
		String validationFileName = "part3/validate1";
		
		RegressionIO regIO = new RegressionIO();
		NeuralNetworkClassifier nnc = null;
		try {
			nnc = regIO.instantiateNNClassifierWithTrainingData(fileName);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		nnc.train();
		double error = nnc.calculateTrainingError();
		System.out.println("error: " + error);
		
		ArrayList<Record> tests = regIO.readTestRecords(testFileName);
		System.out.println(tests);
		ArrayList<Double> testOutputs = nnc.classifyRecords(tests);
		regIO.writeClassifiedRecordsToFile(testOutputName, tests, testOutputs);
		
		ArrayList<Record> validationRecords = regIO.readValidationRecords(validationFileName);
		ArrayList<Double> valOutputs = nnc.classifyRecords(validationRecords);
	}
}