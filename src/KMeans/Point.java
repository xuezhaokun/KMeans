package KMeans;

import java.util.List;
/**
 * point class for each data read from dataset
 * @author Zhaokun Xue
 *
 */
public class Point {
	private List<Double> data; // data value in the point
	private int cluster = -1; // data cluster from kmeans computation
	private int classifier = -1; // data class label from dataset
	/**
	 * constructor for point
	 * @param data 
	 */
	public Point(List<Double> data) {
		this.data = data;
	}

	/**
	 * calculate distance between two points
	 * @param p1
	 * @param p2
	 * @return distance
	 */
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
	
	// getters and setters
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
