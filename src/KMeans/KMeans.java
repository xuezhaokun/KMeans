package KMeans;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

public class KMeans {
	private int k;
	private String filename;
	//private HashMap<String, List<Cluster>> clusters;
	private List<Cluster> clusters;
	
	public KMeans(int k, String filename) {
		this.k = k;
		this.filename = filename;
		//this.clusters = new HashMap<String, List<Cluster>>();
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


	public void initClusters(List<Point> points, int k){
		Collections.shuffle(points);
		List<Point> centers = new ArrayList<Point>();
		for(int i = 0; i < k; i++){
			centers.add(points.get(i));
		}
		List<Cluster> initClusters = new ArrayList<Cluster>();
		for(Point center : centers){
			Cluster cluster = new Cluster(center);
			initClusters.add(cluster);
		}
		this.setClusters(initClusters);
		//return initClusters;
	}
	
	public void assignPointsToClusters(List<Point> points){
		int numClusters = this.getClusters().size();
		for(Point point : points){
			double minDist = Double.MAX_VALUE;
			int cluster = -1;
			for(int i = 0; i < numClusters; i++){
				Cluster currentCluster = this.getClusters().get(i);
				Point currentCenter =currentCluster.getCenter();
				double tmpDist = Point.dist(point, currentCenter);
				if(tmpDist < minDist){
					minDist = tmpDist;
					cluster = i;
				}
			}
			point.setCluster(cluster);
			this.getClusters().get(cluster).getPoints().add(point);
		}
	}
	
	public void updateClusters(List<Point> points){
		
	}
	
	public int getK() {
		return k;
	}

	public void setK(int k) {
		this.k = k;
	}

	/*public HashMap<String, List<Cluster>> getClusters() {
		return clusters;
	}

	public void setClusters(HashMap<String, List<Cluster>> clusters) {
		this.clusters = clusters;
	}*/

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
