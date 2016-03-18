import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class NeuralNetworkIO {
	//for continuous data
	private HashMap<Integer, double[]> rangeAtColumn = new HashMap<>();
	//for binary, categorical, ordinal data
	private HashMap<Integer, HashMap<String, Double>> symbolToValueAtColumn = new HashMap<>();
	
	//list of attr types: continuous, oridinal, categorical, etc.
	private ArrayList<String> attributeList = new ArrayList<>();
	public NeuralNetworkClassifier instantiateNNClassifierWithTrainingData(
			String fileName) throws Exception {
		ArrayList<Record> recordsToReturn = new ArrayList<>();

		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		// first line ( these is the attribute types )
		String[] componentsOfFirstLine = lines.get(0).split(whitespace);
		for (String attrType : componentsOfFirstLine) {
			if (NeuralNetworkClassifier.attributeTypes.contains(attrType) == true) {
				attributeList.add(attrType);
			} else {
				throw new Exception(
						"attribute in file not one of the correct attributes");
			}
		}
		
		String[] listOfRanges = lines.get(1).split(whitespace);
		for (int colIndex = 0; colIndex < listOfRanges.length; colIndex++) {
			// range symbols are low to high
			String[] strRange = listOfRanges[colIndex].split(",");
			double[] dubRange = new double[strRange.length];
			String typeOfAttrAtIndex = attributeList.get(colIndex);
			if (typeOfAttrAtIndex.equals(NeuralNetworkClassifier.CONTINUOUS) == false) {
				HashMap<String, Double> symbolToVal = new HashMap<>(10);
				int denominator = dubRange.length - 1;
				int counter = 0;
				for (String symbol : strRange) {
					symbolToVal.put(symbol, (double)counter / denominator);
					counter++;
				}
				symbolToValueAtColumn.put(colIndex, symbolToVal);
			}else{// if it is continuous
				assert strRange.length == 2;
				dubRange[0] = Double.parseDouble(strRange[0]);
				dubRange[1] = Double.parseDouble(strRange[1]);
				rangeAtColumn.put(colIndex, dubRange);
			}
		}
		return null;
	}
	
	private Record translateLineComponentsIntoRecords(String[] comps){
		// TODO
		return null;
	}
	
}

























