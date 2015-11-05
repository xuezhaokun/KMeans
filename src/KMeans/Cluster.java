package KMeans;

import java.util.ArrayList;
import java.util.List;

/**
 * cluster object for kmeans
 * @author Zhaokun XUe
 *
 */
public class Cluster {
	private List<Point> points;
	private Point center;
	
	/**
	 * constructor for cluster
	 * @param center center for the cluster
	 */
	public Cluster(Point center) {
		this.points = new ArrayList<Point>();
		this.center = center;
	}
	
	/**
	 * update cluster's center
	 * @param points points in current cluster
	 */
	public void updateCenter(List<Point> points){
		List<Double> newCenterData = points.get(0).getData(); //get the first data in the current cluster as the start calculation point for new center
		int totalNumPoints = points.size(); // total number of points in current cluster
		points.remove(0);// remove the first point, since we already add this data to our calculation for new center
		for(Point point : points){ // iterate the points after removing the first one
			List<Double> currentPointData = point.getData(); // current point's data
			for(int i = 0; i < currentPointData.size(); i++){ // iterate each value in the data
				double tmp = newCenterData.get(i) + currentPointData.get(i); // calculate new value for current value positin in new center
				newCenterData.set(i, tmp);// add the value to new data center 
			}
		}
		// iterate each value in new center's data
		for(int j = 0; j < newCenterData.size(); j++){
			double avg = newCenterData.get(j)/totalNumPoints; // calculate the mean value for each value
			newCenterData.set(j, avg); // set new center value
		}
		Point newCenter = new Point(newCenterData);// construct new center point
		this.setCenter(newCenter);
	}
	
	/**
	 * calculate the cluster scatter for this cluster
	 * @return return cluster scatter value
	 */
	public double calClusterCS(){
		Point center = this.getCenter();
		double clusterCS = 0;
		for(Point point : this.getPoints()){
			double tmp = Math.pow(Point.dist(point, center), 2);
			clusterCS += tmp;
		}
		return clusterCS;
	}
	
	// getters and setters
	public List<Point> getPoints() {
		return points;
	}

	public void setPoints(List<Point> points) {
		this.points = points;
	}

	public Point getCenter() {
		return center;
	}

	public void setCenter(Point center) {
		this.center = center;
	}
	
	
	
}
