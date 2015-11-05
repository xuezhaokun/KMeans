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
/**
 * Implementing k means
 * @author Zhaokun Xue
 *
 */
public class KMeans {
	private int k; // k value
	private List<Cluster> clusters; // clusters in kmeans
	private boolean isConvergent = false; // initialize convergent to false
	private int numClasses = 0; // number of classes in the dataset
	/**
	 * constructor for KMeans class
	 * @param k // k value
	 */
	public KMeans(int k) {
		this.k = k;
		this.clusters = new ArrayList<Cluster>(); // initialize clusters to an empty array
	}
	
	/**
	 * read data from file using weka instance feature
	 * @param filename input file name
	 * @return a list of KnnData
	 * @throws IOException
	 */
	public List<Point> readDataFile(String filename) throws IOException {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		Instances read_data = new Instances(inputReader);
		int numOfClasses = read_data.numDistinctValues(read_data.numAttributes()-1);
		this.setNumClasses(numOfClasses);// set the number of classes in current dataset
		List<Point> list_data = new ArrayList<Point>();
		for(int i = 0; i < read_data.numInstances(); i ++){
			List<Double> attributes = new ArrayList<Double>();
			Instance current_instance = read_data.instance(i);
			int num_attributes = current_instance.numAttributes() ;
			int classifier = (int) current_instance.value(num_attributes - 1);
			for(int j = 0; j < num_attributes - 1; j++){
				attributes.add(current_instance.value(j));
			}
			Point point_data = new Point(attributes); 
			point_data.setClassifier(classifier);// set classifier in for each point based on the info in dataset
			list_data.add(point_data);
		}
		inputReader.close();
		return list_data;
	}

	/**
	 * Initialize clusters, set clusters' centers
	 * @param points all points in dataset
	 */
	public void initClusters(List<Point> points){
		Collections.shuffle(points); // shuffle the points randomly
		List<Point> centers = new ArrayList<Point>();
		int k = this.getK();
		// pick the first k centers after shuffling, i.e. randomly select k centers
		for(int i = 0; i < k; i++){
			centers.add(points.get(i));
		}

		List<Cluster> initClusters = new ArrayList<Cluster>();
		// create new cluster based on centers we randomly selected
		for(Point center : centers){
			Cluster cluster = new Cluster(center);
			initClusters.add(cluster);
		}
		// set clusters in our kmeans object
		this.setClusters(initClusters);
	}
	
	/**
	 * assign all points to corresponding clusters based on the distances to clusters
	 * @param points all points in dataset
	 */
	public void assignPointsToClusters(List<Point> points){
		int numClusters = this.getClusters().size();
		// iterate each point in the dataset and assign them to corresponding cluster
		for(Point point : points){
			double minDist = Double.MAX_VALUE; // current min dist to cluster
			int cluster = -1; // initialize assigned cluster to -1
			// iterate clusters to decide which clutser we should assign current point to
			for(int i = 0; i < numClusters; i++){
				Cluster currentCluster = this.getClusters().get(i); // get current cluster
				Point currentCenter =currentCluster.getCenter(); // get current cluster's center point
				double tmpDist = Point.dist(point, currentCenter); // calculate dist from current point to current center
				if(tmpDist < minDist){
					minDist = tmpDist; // update mid dist to cluster
					cluster = i; // update which cluster should the point be assigned to
				}
			}
			// check whether the point current cluster is the same we update above
			// if  any of the point's cluster is not the same as the one we compute above, we set convergent to false
			if(point.getCluster() != cluster){
				this.setConvergent(false); 
			}
			// update current point's cluster
			point.setCluster(cluster);
			// add current point to its cluster's points set
			this.getClusters().get(cluster).getPoints().add(point);
		}
	}
	
	/**
	 * update clusters, update cluster centers and reassign points
	 * @param points all points in dataset
	 */
	public void updateClusters(List<Point> points){
		int numClusters = this.getClusters().size();// number of clusters
		List<Cluster> currentClusters = this.getClusters();
		// iterate current clusters to update their centers
		for(int i = 0; i < numClusters; i++){
			Cluster currentCluster = currentClusters.get(i);// get current cluster
			List<Point> pointsInCurrentCluster = currentCluster.getPoints(); // get the points in current cluster
			// if some cluster does not have any point, we regard it as an error and reinitialize clusters
			if(pointsInCurrentCluster.isEmpty()){
				this.initClusters(points);// reinitialize centers
				this.assignPointsToClusters(points); // reassign pointers to clusters
				this.updateClusters(points); // update again
				break; // break current loop
			}else{
				currentCluster.updateCenter(pointsInCurrentCluster); // update center
				currentCluster.getPoints().clear(); // clear the points in current cluster
			}
		}
		this.setConvergent(true); // set convergent to true for assignPointsToCluster function
		this.assignPointsToClusters(points); //reassign poins to current clusters based on new centers
	}
	
	/**
	 * continue update clusters until we reach a convergent status
	 * @param points
	 */
	public void updateToConvergence(List<Point> points){
		// update clusters until convergent results
		while(!this.isConvergent()){
			this.updateClusters(points);
		}
	}
	
	/**
	 * calculate cluster scatter
	 * @return cluster scatter
	 */
	public double calCS(){
		double cs = 0;
		// calculate the cluster scatter for each cluster and sum them up
		for(Cluster cluster : this.getClusters()){
			double clusterCS = cluster.calClusterCS();
			cs += clusterCS;
		}
		return cs;
	}
	
	/**
	 * set up the contingency table with columns on class labels (classifier) and rows on our clusters   
	 * @param points all points in dataset
	 * @return the contingency table
	 */
	public int[][] setUpContingencyTable(List<Point> points){
		int numOfClasses = this.getNumClasses();
		int k = this.getK();
		// since we set k to the same number of data class lables, we will have a k*k contingency table
		// column is the class label of the point and row is the cluster of the point
		int[][] contingencyTable = new int[k][numOfClasses]; //initialize all entrance to 0
		// iterate each row
		for (int row=0; row < k; row++){
			// iterate each column
		    for (int col=0; col < numOfClasses; col++){
		    	// iterate point
		    	for(Point point : points){
		    		// check if the point class lable == col num and cluster == row, we add 1 to that entry in the table
		    		if(point.getClassifier() == col && point.getCluster() == row){
		    			contingencyTable[row][col]++;
		    		}
		    	}
		    }
		}
		
		return contingencyTable;
	}
	
	/**
	 * calculate NMI
	 * @param contingencyTable the contingency table made from above
	 * @return the NMI 
	 */
	public double calNMI(int[][] contingencyTable){
		int k = this.getK();
		int numOfClasses = this.getNumClasses();
		int[] rowSum = new int[k]; // sum for each row  
		int[] colSum = new int[numOfClasses]; // sum of each column
		double h_row = 0; //entropy for row
		double h_col = 0; //entropy for column
		double i_row_col = 0; // mutual information of row and col
		double nmi = 0; // init NMI to 0
		double totalData = 0;

		// calculate sum for each row
		for (int row = 0; row < k; row++){
			for (int col=0; col < numOfClasses; col++){
				rowSum[row] += contingencyTable[row][col];
			}
		}
		
		// calculate sum for each column
		for(int col = 0; col < numOfClasses; col++){
			for(int row = 0; row < k; row++){
				colSum[col] += contingencyTable[row][col];
			}
		}
		// total data is the sum of the column or row
		for(int sum : colSum){
			totalData += sum;
		}
		
		// calculate the entropy for row 
		for(int i = 0; i < k; i++){
			double p_row = rowSum[i]/totalData; // (sum of each row)/(total N) 
			h_row += -p_row*Math.log(p_row);
		}
		
		// calculate the entropy for column
		for(int i = 0; i < numOfClasses; i++){
			double p_col = colSum[i]/totalData; // (sum of each column)/(total N) 
			h_col += -p_col*Math.log(p_col);
		}
		
		// calculate the mutual information for row and column
		for(int row = 0; row < k; row++){
			for(int col = 0; col < numOfClasses; col++){
				double p_cell = contingencyTable[row][col]/totalData; // (each entry)/(total N)
				if(p_cell == 0){ // if 0, don't calculate logrithm part, just add 0, i.e. continue to next iterate
					continue;
				}else{
					i_row_col += p_cell*Math.log(p_cell/(rowSum[row]*colSum[col]/(Math.pow(totalData, 2))));
				}
			}
		}
		nmi = (2*i_row_col)/(h_row + h_col);
		return nmi;
	}
	
	/**
	 * getter for k value
	 * @return k value
	 */
	public int getK() {
		return k;
	}

	/**
	 * setter for k
	 * @param k
	 */
	public void setK(int k) {
		this.k = k;
	}

	/**
	 * getter for clusters
	 * @return clusters
	 */
	public List<Cluster> getClusters() {
		return clusters;
	}

	/**
	 * setter for clusters
	 * @param clusters
	 */
	public void setClusters(List<Cluster> clusters) {
		this.clusters = clusters;
	}

	/**
	 * getter for convergent
	 * @return true or false
	 */
	public boolean isConvergent() {
		return isConvergent;
	}

	/**
	 * setter for convergent
	 * @param isConvergent
	 */
	public void setConvergent(boolean isConvergent) {
		this.isConvergent = isConvergent;
	}
	
	/**
	 * getter for numClasses
	 * @return number of classes
	 */
	public int getNumClasses() {
		return numClasses;
	}

	/**
	 * setter for number of clusters
	 * @param numClasses
	 */
	public void setNumClasses(int numClasses) {
		this.numClasses = numClasses;
	}

	public static void main(String[] args) throws IOException {
		String [] filenames = new String [] {"artdata.arff", "ionosphere.arff", "iris.arff", "seeds.arff"};
		for(String filename : filenames){
			System.out.println("********** Results for dataset: " + filename + " **********");
			System.out.println("Part 3.2 with k = number of classes in the dataset");
			int k_value = 0;
			if(filename == "ionosphere.arff"){ // if dataset is ionosphere, set k_value to 2
				k_value = 2;
			}else{
				k_value = 3;
			}
			KMeans k1 = new KMeans(k_value); // construct k1 instance
			List<Point> dataPoints = k1.readDataFile("data/"+filename); //get data points
			for(int i = 0; i < 10; i++){// repeat 10 runs
				k1.initClusters(dataPoints);
				k1.assignPointsToClusters(dataPoints);
				k1.updateToConvergence(dataPoints);
				double cs = k1.calCS();
				int[][] table = k1.setUpContingencyTable(dataPoints);
				double nmi = k1.calNMI(table);
				System.out.println("Round: "+ i + " CS: " + cs +  " NMI: " + nmi);
			}
			System.out.println("------------- Part3.3 knee criterion -------------");
			for(int k = 1; k < 16; k++){// k value from 1 to 15
				double minCS = Double.MAX_VALUE; // minimal cluster scatter for current k
				for(int j = 0; j < 10; j++){ // repeat 10 runs
					KMeans k2 = new KMeans(k); // construct instance k2
					List<Point> data = k2.readDataFile("data/"+filename);
					k2.initClusters(data);
					k2.assignPointsToClusters(data);
					k2.updateToConvergence(data);
					double cs = k2.calCS();
					if(cs < minCS){ // update min cluster scatter
						minCS = cs;
					}
				}
				System.out.println("k:" + k + " CS: " + minCS);
			}
		}

	}

}
