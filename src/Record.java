
public class Record {

	private double[] attrList;
	public double[] getAttrList() {
		return attrList;
	}
	private int label;
	public int getLabel() {
			return label;
	}
	public Record(double[] attrList, int label) {
		this.attrList = attrList;
		this.label = label;
	}
	
	public int numberOfAttributes(){
		return attrList.length;
	}

	@Override
	public String toString() {
		StringBuffer sBuffer = new StringBuffer("");
		for (double dub : this.attrList) {
			sBuffer.append(String.format("%.2f", dub) + ", ");
		}
		sBuffer.replace(sBuffer.length() - 2, sBuffer.length(), " || ");
		sBuffer.append(this.label);
		return sBuffer.toString();
	}

}
