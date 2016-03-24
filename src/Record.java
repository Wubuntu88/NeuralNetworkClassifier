
public class Record {

	private double[] attrList;
	public double[] getAttrList() {
		return attrList;
	}
	private double label;
	public double getLabel() {
			return label;
	}
	public Record(double[] attrList, double label) {
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
		sBuffer.append(String.format("%.2f", this.label));
		return sBuffer.toString();
	}

}
