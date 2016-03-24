import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.TreeSet;

public class NeuralNetworkIO {
	
	public static final String BINARY = "binary";
	public static final String CATEGORICAL = "categorical";
	public static final String ORDINAL = "ordinal";
	public static final String CONTINUOUS = "continuous";
	public static final String LABEL = "label";
	public static final TreeSet<String> attributeTypeSet = new TreeSet<String>(
			Arrays.asList(BINARY, CATEGORICAL, ORDINAL, CONTINUOUS, LABEL));
	
	//for continuous data
	private HashMap<Integer, double[]> rangeAtColumn = new HashMap<>();
	//for binary, categorical, ordinal data
	private HashMap<Integer, HashMap<String, Double>> symbolToValueAtColumn = new HashMap<>();
	
	//list of attr types: continuous, oridinal, categorical, etc.
	private ArrayList<String> attributeTypeList = new ArrayList<>();
	
	public NeuralNetworkClassifier instantiateNNClassifierWithTrainingData(
			String fileName) throws Exception {
		ArrayList<Record> recordsToReturn = new ArrayList<>();
		
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName),
				Charset.defaultCharset());
		// first line ( these is the attribute types )
		String[] componentsOfFirstLine = lines.get(0).split(whitespace);
		for (String attrType : componentsOfFirstLine) {
			if (NeuralNetworkIO.attributeTypeSet.contains(attrType) == true) {
				attributeTypeList.add(attrType);
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
			String typeOfAttrAtIndex = attributeTypeList.get(colIndex);
			if (typeOfAttrAtIndex.equals(NeuralNetworkIO.CONTINUOUS) == false) {
				HashMap<String, Double> symbolToVal = new HashMap<>(10);
				int denominator = dubRange.length - 1;
				int counter = 0;
				for (String symbol : strRange) {
					symbolToVal.put(symbol, (double)counter / denominator);
					counter++;
				}
				symbolToValueAtColumn.put(colIndex, symbolToVal);//for non continuous columns
			}else{// if it is continuous
				assert strRange.length == 2;
				dubRange[0] = Double.parseDouble(strRange[0]);
				dubRange[1] = Double.parseDouble(strRange[1]);
				rangeAtColumn.put(colIndex, dubRange);//for continuous columns
			}
		}
		
		//now I create the records after initializing the hasmaps
		ArrayList<Record> records = new ArrayList<>();
		for(int i = 2; i < lines.size(); i++){
			String[] comps = lines.get(i).split(whitespace);
			Record record = translateLineComponentsIntoRecords(comps);
			records.add(record);
		}
		NeuralNetworkClassifier nnc = new NeuralNetworkClassifier(records);
		return nnc;
	}
	
	
	private Record translateLineComponentsIntoRecords(String[] comps){
		double[] attrs = null;
		double label = -1;
		boolean hasLabelIncluded = comps.length == this.attributeTypeList.size();
		if(hasLabelIncluded){// if the training or test includes the label
			attrs = new double[comps.length - 1];
			HashMap<String, Double> labelToValue = symbolToValueAtColumn.get(comps.length - 1);
			double value = labelToValue.get(comps[comps.length - 1]);
			label = value;
		}else{//comps does not include the label
			attrs = new double[comps.length];
		}
		int lenToIterateTo = hasLabelIncluded ? comps.length - 1 : comps.length;
		for(int colIndex = 0; colIndex < lenToIterateTo; colIndex++){//don't do label in for
			String attrType = attributeTypeList.get(colIndex);
			if(attributeTypeList.get(colIndex).equals(CONTINUOUS)){//for continous columns
				double number = Double.parseDouble(comps[colIndex]);
				double normalizedNumber = normalizeContinuousVariableAtColumn(number, colIndex);
				attrs[colIndex] = normalizedNumber;
			}else{//for non-continous data
				HashMap<String, Double> symbolToValue = symbolToValueAtColumn.get(colIndex);
				
				double value = symbolToValue.get(comps[colIndex]);
				
				attrs[colIndex] = value;
			}
		}
		Record record = new Record(attrs, label);
		return record;
	}
	
	public ArrayList<Record> readTestRecordsFromFile(String fileName) throws Exception{
		ArrayList<Record> recordsToReturn = new ArrayList<>();
		String whitespace = "[ ]+";
		List<String> lines = Files.readAllLines(Paths.get(fileName), Charset.defaultCharset());;
		for(String line: lines){
			String[] comps = line.split(whitespace);
			Record record = translateLineComponentsIntoRecords(comps);
			System.out.println("@@" + record);
			recordsToReturn.add(record);
		}
		return recordsToReturn;
	}
	
	
	public void writeRecordsToFile(String fileName, 
			ArrayList<Record> recordsToPrint, ArrayList<Double> classifications){
		assert recordsToPrint.size() == classifications.size();
		
		PrintWriter pw = null;
		try {
			pw = new PrintWriter(fileName);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			return;
		}
		
		StringBuffer sBuffer = new StringBuffer("");
		int index = 0;
		for(Record record: recordsToPrint){
			
			Double classification = classifications.get(index);
			Record rec = new Record(record.getAttrList(), classification);
			System.out.println(rec);
			String recordDescriptionForHuman = convertRecordToHumanReadableString(rec);
			//System.out.println(recordDescriptionForHuman);
			sBuffer.append(recordDescriptionForHuman);
			//sBuffer.append(String.format("; confidence: %.2f\n", ci.getConfidenceLevel()));
			index++;
		}
		sBuffer.replace(sBuffer.length() - 1, sBuffer.length(), "");
		pw.write(sBuffer.toString());
		pw.close();
	}
	
	private String convertRecordToHumanReadableString(Record record){
		StringBuffer sBuffer = new StringBuffer("");
		int lenOfRecordAttrList = record.getAttrList().length;
		double[] attrList = record.getAttrList();
		for(int index = 0; index < lenOfRecordAttrList; index++){
			if(this.attributeTypeList.get(index).equals(this.CONTINUOUS)){
				sBuffer.append(attrList[index] + ", ");
			}else{
			HashMap<String, Double> hMap = symbolToValueAtColumn.get(index);
				for(String key: hMap.keySet()){
					double valForKey = hMap.get(key);
					if(valForKey == attrList[index]){
						sBuffer.append(key + ", ");
						break;
					}else{
						System.out.println("error in printing record; found nothing equal");
					}
				}
			}
		}
		sBuffer.replace(sBuffer.length() - 2, sBuffer.length(), "");
		
		//now for the label
		HashMap<String, Double> hMap = symbolToValueAtColumn.get(attributeTypeList.size() - 1);
		double minDistance = Double.MAX_VALUE;
		String closestLabel = null;
		if(hMap == null){
			System.out.println("hmap is null");
		}
		for(String labelKey: hMap.keySet()){
			double valForKey = hMap.get(labelKey);
			double dist = Math.abs(record.getLabel() - valForKey);
			if(dist < minDistance){
				minDistance = dist;
				closestLabel = labelKey;
			}
		}
		sBuffer.append(" || " + closestLabel + "\n");
		return sBuffer.toString();
	}
	
	private double normalizeContinuousVariableAtColumn(double number, int column){
		double[] theRange = rangeAtColumn.get(column);
		//(currentNumber - min) / (max - min)
		double translation = (number - theRange[0]) / (theRange[1] - theRange[0]);
		return translation;
	}
	
}

























