package KMeans;

import java.util.ArrayList;
import java.util.List;

public class Cluster {
	private List<Point> points;
	private Point center;
	
	public Cluster(Point center) {
		this.points = new ArrayList<Point>();
		this.center = center;
	}
	
	public void updateCenter(List<Point> points){
		List<Double> newCenterData = points.get(0).getData();
		int totalNumPoints = points.size();
		points.remove(0);
		for(Point point : points){
			List<Double> currentPointData = point.getData();
			for(int i = 0; i < currentPointData.size(); i++){
				double tmp = newCenterData.get(i) + currentPointData.get(i);
				newCenterData.set(i, tmp);
			}
		}
		for(int j = 0; j < newCenterData.size(); j++){
			double avg = newCenterData.get(j)/totalNumPoints;
			newCenterData.set(j, avg);
		}
		Point newCenter = new Point(newCenterData);
		this.setCenter(newCenter);
	}
	
	public double calClusterCS(){
		Point center = this.getCenter();
		double clusterCS = 0;
		for(Point point : this.getPoints()){
			double tmp = Math.pow(Point.dist(point, center), 2);
			clusterCS += tmp;
		}
		return clusterCS;
	}
	
		
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
