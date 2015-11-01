package KMeans;

import java.util.List;

public class Point {
	private List<Double> data;
	//private String classifier;
	
	public Point(List<Double> data) {
		this.data = data;
		//this.classifier = classifier;
	}

	public List<Double> getData() {
		return data;
	}

	public void setData(List<Double> data) {
		this.data = data;
	}

	/*public String getClassifier() {
		return classifier;
	}

	public void setClassifier(String classifier) {
		this.classifier = classifier;
	}*/
	
	
}
