import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class RegressionIO {
	double[] maxesAtInputs;
	double[] minsAtInputs;
	double[] maxesAtOutputs;
	double[] minsAtOutputs;
	int numberOfInputs;
	int numberOfOutputs;
	ArrayList<Record> records = new ArrayList<>();
	
	public NeuralNetworkClassifier instantiateNNClassifierWithTrainingData(
			String fileName) throws Exception {
		ArrayList<Record> recordsToReturn = new ArrayList<>();
		
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		// first line ( these is the attribute types )
		String[] componentsOfFirstLine = lines.get(0).split(whitespace);
		numberOfInputs = Integer.parseInt(componentsOfFirstLine[1]);
		numberOfOutputs = Integer.parseInt(componentsOfFirstLine[2]);
		maxesAtInputs = new double[numberOfInputs];
		minsAtInputs = new double[numberOfInputs];
		maxesAtOutputs = new double[numberOfOutputs];
		minsAtOutputs = new double[numberOfOutputs];
		
		for(int i = 1; i < lines.size(); i++){
			String[] comps = lines.get(i).split(whitespace);
			for(int j = 0; j < numberOfInputs; j++){
				double num = Double.parseDouble(comps[j]);
				if(maxesAtInputs[j] < num){
					maxesAtInputs[j] = num;
				}
				if(minsAtInputs[j] > num){
					minsAtInputs[j] = num;
				}
			}
			for(int q = 0, index = 0; q < numberOfOutputs;q++, index++){
				double num = Double.parseDouble(comps[q+numberOfInputs]);
				if(maxesAtOutputs[index] < num){
					maxesAtOutputs[index] = num;
				}
				if(minsAtOutputs[index] > num){
					minsAtOutputs[index] = num;
				}
			}
		}
		
		for(int i = 1; i < lines.size(); i++){
			String[] comps = lines.get(i).split(whitespace);
			double[] inputAttrs = getInputsWithComps(comps);
			double[] outputAttrs = getOutputsWithComps(comps);
			Record record = new Record(inputAttrs, outputAttrs[0]);
			records.add(record);
		}

		NeuralNetworkClassifier nnc = new NeuralNetworkClassifier(records);
		return nnc;
	}
	
	private double[] getInputsWithComps(String[] comps){
		double[] inputAttrs = new double[numberOfInputs];
		for(int j = 0; j < inputAttrs.length; j++){
			double num = Double.parseDouble(comps[j]);
			inputAttrs[j] = convertInputValueAtCol(j, num);
		}
		return inputAttrs;
	}
	private double[] getOutputsWithComps(String[] comps){
		double[] outputAttrs = new double[numberOfOutputs];
		for(int j = 0; j < outputAttrs.length; j++){
			double num = Double.parseDouble(comps[j+numberOfInputs]);
			outputAttrs[j] = convertOutputValueAtCol(j, num);
		}
		return outputAttrs;
	}
	
	public double calculateTrainingError(ArrayList<Double> classifications){
		return calculateErrorClassifications(records, classifications);
	}
	
	public double calculateErrorClassifications(ArrayList<Record> records,
			ArrayList<Double> classifications){
		assert records.size() == classifications.size();
		double meanSquaredError = 0;
		for(int i = 0; i < records.size(); i++){
			double num1 = records.get(i).getLabel();
			double num2 = classifications.get(i);
			double diff = num1 - num2;
			meanSquaredError += Math.pow(diff, 2);
		}
		return Math.sqrt(meanSquaredError);
	}
	
	public ArrayList<Record> readTestRecords(String fileName){
		ArrayList<Record> records = new ArrayList<>();
		String whitespace = "[ ]+";
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(fileName),
					Charset.defaultCharset());
		} catch (IOException e) {
			e.printStackTrace();
		}
		for(int i = 0; i < lines.size(); i++){
			String[] comps = lines.get(i).split(whitespace);
			double[] inputs = getInputsWithComps(comps);
			Record record = new Record(inputs, -1.0);
			records.add(record);
		}
		return records;
	}
	
	public ArrayList<Record> readValidationRecords(String fileName){
		ArrayList<Record> records = new ArrayList<>();
		String whitespace = "[ ]+";
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(fileName),
					Charset.defaultCharset());
		} catch (IOException e) {
			e.printStackTrace();
		}
		for(int i = 0; i < lines.size(); i++){
			String[] comps = lines.get(i).split(whitespace);
			double[] inputs = getInputsWithComps(comps);
			double[] outputs = getOutputsWithComps(comps);
			Record record = new Record(inputs, outputs[0]);
			records.add(record);
		}
		return records;
	}
	
	public void writeClassifiedRecordsToFile(String outFileName, 
			ArrayList<Record> recordsToPrint, ArrayList<Double> classifications){
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(outFileName);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}
		
		StringBuffer sBuffer = new StringBuffer("");
		int index = 0;
		for(Record record: recordsToPrint){
			Double classification = classifications.get(index);
			Record rec = new Record(record.getAttrList(), classification);
			for(int index1 = 0; index1 < rec.getAttrList().length; index1++){
				
				sBuffer.append(String.format("%.2f", rec.getAttrList()[index1]) + ", ");
			}
			sBuffer.replace(sBuffer.length() - 2, sBuffer.length(), "");
			
			sBuffer.append(" || " + String.format("%.2f", classifications.get(index)) + "\n");
			index++;
		}
		sBuffer.replace(sBuffer.length() - 1, sBuffer.length(), "");
		pw.write(sBuffer.toString());
		pw.close();
	}
	
	private double convertInputValueAtCol(int colIndex, double value){
		double max = maxesAtInputs[colIndex];
		double min = minsAtInputs[colIndex];
		return (value - min) / (max - min);
	}
	
	private double convertOutputValueAtCol(int colIndex, double value){
		double max = maxesAtOutputs[colIndex];
		double min = minsAtOutputs[colIndex];
		return (value - min) / (max - min);
	}
	
	private double convertInputAtColumnBack(int colIndex, double value){
		double max = maxesAtInputs[colIndex];
		double min = minsAtInputs[colIndex];
		return value * (max - min) + min;
	}
	
	private double convertOutputAtColumnBack(int colIndex, double value){
		double max = maxesAtOutputs[colIndex];
		double min = minsAtOutputs[colIndex];
		return value * (max - min) + min;
	}
}


















