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
	private List<Cluster> clusters;
	private boolean isConvergent = true;
	
	public KMeans(int k) {
		this.k = k;
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
			int classifier = (int) current_instance.value(num_attributes - 1);
			//System.out.println(classifier);
			for(int j = 0; j < num_attributes - 1; j++){
				attributes.add(current_instance.value(j));
			}
			Point point_data = new Point(attributes); 
			point_data.setClassifier(classifier);
			list_data.add(point_data);
		}
		return list_data;
	}


	public void initClusters(List<Point> points){
		Collections.shuffle(points);
		List<Point> centers = new ArrayList<Point>();
		int k = this.getK();
		for(int i = 0; i < k; i++){
			centers.add(points.get(i));
		}
		List<Cluster> initClusters = new ArrayList<Cluster>();
		for(Point center : centers){
			Cluster cluster = new Cluster(center);
			initClusters.add(cluster);
		}
		this.setClusters(initClusters);
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
			if(point.getCluster() != cluster){
				this.setConvergent(false);
			}
			point.setCluster(cluster);
			this.getClusters().get(cluster).getPoints().add(point);
		}
	}
	
	public void updateClusters(List<Point> points){
		int numClusters = this.getClusters().size();
		List<Cluster> currentClusters = this.getClusters();
		for(int i = 0; i < numClusters; i++){
			Cluster currentCluster = currentClusters.get(i);
			List<Point> pointsInCurrentCluster = currentCluster.getPoints();
			currentCluster.updateCenter(pointsInCurrentCluster);
			currentCluster.getPoints().clear();
		}
		this.setConvergent(true);
		this.assignPointsToClusters(points);
	}
	
	public void updateToConvergence(List<Point> points){
		//int i = 0;
		while(!this.isConvergent()){
			//System.out.println(i);
			this.updateClusters(points);
			//i++;
		}
	}
	
	public double calCS(){
		double cs = 0;
		for(Cluster cluster : this.getClusters()){
			double clusterCS = cluster.calClusterCS();
			cs += clusterCS;
		}
		return cs;
	}
	
	public int[][] setUpContingencyTable(List<Point> points){
		int k = this.getK();
		int[][] contingencyTable = new int[k][k];
		int sum=  0;
		for (int row=0; row < k; row++){
		    for (int col=0; col < k; col++){
		    	for(Point point : points){
		    		if(point.getClassifier() == col && point.getCluster() == row){
		    			contingencyTable[row][col]++;
		    		}
		    	}
		    }
		}
		
		/*
		for (int row=0; row < contingencyTable.length; row++){
		    for (int col=0; col < contingencyTable[row].length; col++){
		    	System.out.print(contingencyTable[row][col] + " ");
		    }
		    System.out.println("");
		}
		
		for(int j = 0; j < k; j ++){
			System.out.println("row: " + rowSum[j]);
			System.out.println("col: " + colSum[j]);
		}*/
		return contingencyTable;
	}
	
	public double calNMI(int[][] contingencyTable){
		int k = this.getK();
		int[] rowSum = new int[k];
		int[] colSum = new int[k];
		double h_row = 0; //clutser
		double h_col = 0; //classifier
		double i_row_col = 0;
		double nmi = 0;
		int totalData = 0;
		for (int row = 0; row < k; row++){
			for (int col=0; col < k; col++){
				rowSum[row] += contingencyTable[row][col];
			}
		}
		
		for(int col = 0; col < k; col++){
			for(int row = 0; row < k; row++){
				colSum[col] += contingencyTable[row][col];
			}
		}
		
		for(int sum : colSum){
			totalData += sum;
		}
		
		for(int i = 0; i < k; i++){
			h_row += -(rowSum[i]/totalData)*Math.log((rowSum[i]/totalData));
			h_col += -(colSum[i]/totalData)*Math.log((colSum[i]/totalData));
		}
		
		for(int row = 0; row < k; row++){
			for(int col = 0; col < k; col++){
				if((contingencyTable[row][col]/totalData) == 0){
					i_row_col += 0;
				}else{
					
					i_row_col += (contingencyTable[row][col]/totalData)*Math.log((contingencyTable[row][col]/totalData)/(rowSum[row]*colSum[col]/totalData));
				}
			}

		}
		
		nmi = (2*i_row_col)/(h_row + h_col);
		return nmi;
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

	public boolean isConvergent() {
		return isConvergent;
	}

	public void setConvergent(boolean isConvergent) {
		this.isConvergent = isConvergent;
	}

	public static void main(String[] args) throws IOException {
		List<Point> dataPoints = KMeans.readDataFile("data/artdata.arff");
		KMeans k1 = new KMeans(3);
		for(int k = 0; k < 10; k++){
			k1.initClusters(dataPoints);
			k1.assignPointsToClusters(dataPoints);
			k1.updateToConvergence(dataPoints);
			List<Cluster> pointsClusters = k1.getClusters();
		
			/*for(Point point : dataPoints){
				System.out.println(point.toString());
			}*/
			double cs = k1.calCS(); 
			System.out.println("*******************");
			System.out.println(cs);
		}
		int[][] table = k1.setUpContingencyTable(dataPoints);
		double nmi = k1.calNMI(table);
		int c = 0;
		for(Point point : dataPoints){
			if(point.getClassifier() == 1 && point.getCluster() == 1){
				c++;
			}
			//System.out.println(point.toString());
		}
		System.out.println("NMI: " + nmi);
	}

}
