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
	private List<Cluster> clusters;
	private boolean isConvergent = false;
	private int numClasses = 0;
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
	public List<Point> readDataFile(String filename) throws IOException {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		Instances read_data = new Instances(inputReader);
		int numOfClasses = read_data.numDistinctValues(read_data.numAttributes()-1);
		this.setNumClasses(numOfClasses);
		//System.out.println(read_data.numDistinctValues(read_data.numAttributes()-1));
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
			point_data.setClassifier(classifier);
			list_data.add(point_data);
		}
		inputReader.close();
		return list_data;
	}


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
	
	public void updateClusters(List<Point> points){
		int numClusters = this.getClusters().size();
		List<Cluster> currentClusters = this.getClusters();
		// iterate current clusters to update their centers
		for(int i = 0; i < numClusters; i++){
			Cluster currentCluster = currentClusters.get(i);// get current cluster
			List<Point> pointsInCurrentCluster = currentCluster.getPoints(); // get the points in current cluster
			if(pointsInCurrentCluster.isEmpty()){
				this.initClusters(points);
				this.assignPointsToClusters(points);
				this.updateClusters(points);
				break;
			}else{
				currentCluster.updateCenter(pointsInCurrentCluster); // update center
				currentCluster.getPoints().clear(); // clear the points in current cluster
			}
		}
		this.setConvergent(true); // set convergent to true for assignPointsToCluster function
		this.assignPointsToClusters(points); //reassign poins to current clusters based on new centers
	}
	
	public void updateToConvergence(List<Point> points){
		// update clusters until convergent results
		while(!this.isConvergent()){
			this.updateClusters(points);
		}
	}
	

	public double calCS(){
		double cs = 0;
		// calculate the cluster scatter for each cluster and sum them up
		for(Cluster cluster : this.getClusters()){
			double clusterCS = cluster.calClusterCS();
			cs += clusterCS;
		}
		return cs;
	}
	
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
		
		/*for (int row=0; row < contingencyTable.length; row++){
		    for (int col=0; col < contingencyTable[row].length; col++){
		    	System.out.print(contingencyTable[row][col] + " ");
		    }
		    System.out.println("");
		}*/
		
		return contingencyTable;
	}
	
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
		
		// calculate the entropy for row and column
		for(int i = 0; i < k; i++){
			double p_row = rowSum[i]/totalData; // (sum of each row)/(total N) 
			h_row += -p_row*Math.log(p_row);
			//System.out.println("total: "+ totalData +" row: " + rowSum[i]  + " h_row: " + h_row );
		}
		
		for(int i = 0; i < numOfClasses; i++){
			double p_col = colSum[i]/totalData; // (sum of each column)/(total N) 
			h_col += -p_col*Math.log(p_col);
			//System.out.println("total: "+ totalData + " col: " +colSum[i] + " h_col: " + h_col);
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
				//System.out.println(p_cell + " " +i_row_col);
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

	public int getNumClasses() {
		return numClasses;
	}

	public void setNumClasses(int numClasses) {
		this.numClasses = numClasses;
	}

	public static void main(String[] args) throws IOException {
		KMeans k1 = new KMeans(3);
		List<Point> dataPoints = k1.readDataFile("data/iris.arff");
		for(int i = 0; i < 10; i++){
			k1.initClusters(dataPoints);
			k1.assignPointsToClusters(dataPoints);
			k1.updateToConvergence(dataPoints);
			double cs = k1.calCS(); 
			//System.out.println("*******************");
			System.out.println("CS: " + cs);
			int[][] table = k1.setUpContingencyTable(dataPoints);
			double nmi = k1.calNMI(table);
			System.out.println("NMI: " + nmi);
		}
		System.out.println("------------- knee criterion -------------");
		for(int k = 1; k < 16; k++){
			double minCS = Double.MAX_VALUE;
			for(int j = 0; j < 10; j++){
				KMeans kmeans = new KMeans(k);
				List<Point> data = kmeans.readDataFile("data/iris.arff");
				kmeans.initClusters(data);
				kmeans.assignPointsToClusters(data);
				kmeans.updateToConvergence(data);
				double cs = kmeans.calCS();
				if(cs < minCS){
					minCS = cs;
				}
			}
			System.out.println("CS: " + minCS);
		}

	}

}
