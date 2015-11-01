package KMeans;

import java.util.List;

public class Point {
	private List<Double> data;
	private int cluster = -1;
	private int classifier = -1;
	public Point(List<Double> data) {
		this.data = data;
		//this.cluster = cluster;
		//this.classifier = classifier;
	}

	public static double dist(Point p1, Point p2){
		List<Double> p1_data = p1.getData();
		List<Double> p2_data = p2.getData();
		double distance = 0;
		if(p1_data.size() == p2_data.size()){
			for (int i = 0; i < p1_data.size(); i++){
				double diff = p1_data.get(i) - p2_data.get(i);
				distance += Math.pow(diff, 2);
			}
			distance = Math.sqrt(distance);
		}
		return distance;
	}
	
	public List<Double> getData() {
		return data;
	}

	public void setData(List<Double> data) {
		this.data = data;
	}

	public int getCluster() {
		return cluster;
	}

	public void setCluster(int cluster) {
		this.cluster = cluster;
	}

		
	public int getClassifier() {
		return classifier;
	}

	public void setClassifier(int classifier) {
		this.classifier = classifier;
	}

	@Override
	public String toString() {
		return "Point [data=" + data + ", cluster=" + cluster + ", classifier=" + classifier + "]";
	}


}
