package KMeans;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class KMeans {
	private int k;
	private String filename;
	private List<Cluster> clusters;
	
	public KMeans(int k, String filename) {
		this.k = k;
		this.filename = filename;
		this.clusters = new ArrayList<Cluster>();
	}
	
	/**
	 * read data from file using weka instance feature
	 * @param filename input file name
	 * @return a list of KnnData
	 * @throws IOException
	 */
	public static List<Point> readDataFile(String filename) throws IOException {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		Instances read_data = new Instances(inputReader);
		
		List<Point> list_data = new ArrayList<Point>();
		for(int i = 0; i < read_data.numInstances(); i ++){
			List<Double> attributes = new ArrayList<Double>();
			Instance current_instance = read_data.instance(i);
			int num_attributes = current_instance.numAttributes() ;
			String classifier = Double.toString(current_instance.value(num_attributes - 1));
			for(int j = 0; j < num_attributes - 1; j++){
				attributes.add(current_instance.value(j));
			}
			Point knn_data = new Point(attributes); 
			list_data.add(knn_data);
		}
		return list_data;
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

	public List<Point> initCenters(List<Point> points, int k){
		Collections.shuffle(points);
		List<Point> centers = new ArrayList<Point>();
		for(int i = 0; i < k; i++){
			centers.add(points.get(i));
		}
		return centers;
	}
	
	public List<Cluster> initClusters(List<Point> points, List<Point> centers){
		double maxDist = Double.MAX_VALUE;
		double minDist = maxDist;
		
		for(Point point : points){
			for()
		}
	}
	
	
	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	public List<Cluster> getClusters() {
		return clusters;
	}

	public void setClusters(List<Cluster> clusters) {
		this.clusters = clusters;
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
