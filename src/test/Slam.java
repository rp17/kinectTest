package test;

import java.awt.Component;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class Slam extends Component{
	private ArrayList<MatOfPoint> newObject;
	private ArrayList<Landmark> foundLandmarks;
	private Mat depth;
	private Mat areaMap;
	private int rover_x;
	private int rover_y;
	private Point roverPoint;
	private Scalar green;
	private Scalar black;
	private Scalar red;
	private Scalar blue;
	private Size map;
	private BufferedImage featImage;
	private JKalmanFilter kf;
	
	public Slam(){
		newObject = new ArrayList<MatOfPoint>();
		foundLandmarks = new ArrayList<Landmark>();
		green = new Scalar(0,255,0);
		black = new Scalar(0,0,0);
		red = new Scalar(0,0,255);
		blue = new Scalar(255,0,0);
		map = new Size(500,500);
		areaMap = new Mat(map, CvType.CV_8UC3,black);
		depth = new Mat();
		rover_x = 250;
		rover_y = 100;
		roverPoint = new Point(rover_x,rover_y);
		try{
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void findNewObjects(ArrayList<MatOfPoint> foundObjects, Mat depthInfo){
		depth = depthInfo;
		if(foundLandmarks.isEmpty()){
			for(int idx = 0; idx < foundObjects.size(); idx++){
				MatOfPoint object = foundObjects.get(idx);
				calcLandmarkData(object);
			}
			initKalman();
		}
		else{
			for(int idx = 0; idx < foundObjects.size(); idx++){
				MatOfPoint object = foundObjects.get(idx);
				matchObject(object);
			}
			kalmanFilter();
			for(int idx = foundLandmarks.size() - 1; idx >= 0; idx--){
				if(!foundLandmarks.get(idx).observed){
					foundLandmarks.remove(idx);
				}
			}
		}

		if(!newObject.isEmpty()){
			for(int idx = 0; idx < newObject.size(); idx ++){
				MatOfPoint object = newObject.get(idx);
				calcLandmarkData(object);
			}
			
			initKalman();
		}
	}
	
	public void matchObject(MatOfPoint object){
		double min = 0.1;
		int foundIdx = -1;
		for(int idx = 0; idx < foundLandmarks.size(); idx++){
			double match = Imgproc.matchShapes(object, foundLandmarks.get(idx).getContour(), Imgproc.CV_CONTOURS_MATCH_I3, 0);
			if(match < min){
				min = match;
				foundIdx = idx;
			}
		}
		if(foundIdx != -1){
			updateLandmark(foundIdx, object);
		}
		else{
			newObject.add(object);
		}
	}
	
	public void calcLandmarkData(MatOfPoint object){
		ArrayList<Point> objectPoints = new ArrayList<Point>(object.toList());
		double min = objectPoints.get(0).x;
		double max = objectPoints.get(0).x;
		Point leftmost = objectPoints.get(0);
		Point rightmost = objectPoints.get(0);
		for(int idx = 0; idx < objectPoints.size(); idx++){
			if(objectPoints.get(idx).x < min){
				min = objectPoints.get(idx).x;
				leftmost = objectPoints.get(idx);
			}
			if(objectPoints.get(idx).x > max){
				max = objectPoints.get(idx).x;
				rightmost = objectPoints.get(idx);
			}
			
		}
		Point center = new Point((leftmost.x + rightmost.x)/2,(leftmost.y + rightmost.y)/2);
		double[] depthRatio = depth.get((int)center.x,(int)center.y);
		if(depthRatio == null)
			return;
		double distance = ((depthRatio[0]/255) * 3600)+400;
		double pixelLength = Math.sqrt(Math.pow(rightmost.x-leftmost.x,2)+Math.pow(rightmost.y-leftmost.y,2));
		double range = Math.tan(28.5*Math.PI/180)*distance*2;
		double length = (range/640)*pixelLength;
		double fromCenter = (center.x - 320)*(range/640);
		
		Point location = new Point(rover_x+(fromCenter/100),rover_y+(distance/100));
		Landmark data = new Landmark(location, length/100, object);
		foundLandmarks.add(data);
		drawLandmark(data);
	}
	
	public void updateLandmark(int landmark, MatOfPoint contour){
		ArrayList<Point> objectPoints = new ArrayList<Point>(contour.toList());
		double min = objectPoints.get(0).x;
		double max = objectPoints.get(0).x;
		Point leftmost = objectPoints.get(0);
		Point rightmost = objectPoints.get(0);
		for(int idx = 0; idx < objectPoints.size(); idx++){
			if(objectPoints.get(idx).x < min){
				min = objectPoints.get(idx).x;
				leftmost = objectPoints.get(idx);
			}
			if(objectPoints.get(idx).x > max){
				max = objectPoints.get(idx).x;
				rightmost = objectPoints.get(idx);
			}
			
		}
		Point center = new Point((leftmost.x + rightmost.x)/2,(leftmost.y + rightmost.y)/2);
		double[] depthRatio = depth.get((int)center.x,(int)center.y);
		if(depthRatio == null)
			return;
		double distance = ((depthRatio[0]/255) * 3600)+400;
		double range = Math.tan(28.5*Math.PI/180)*distance*2;
		double fromCenter = (center.x - 320)*(range/640);
		Point location = new Point(rover_x+(fromCenter/100),rover_y+(distance/100));
		foundLandmarks.get(landmark).updateLocation(location);
		foundLandmarks.get(landmark).isObserved();
		
	}
	
	public void initKalman(){
		if(!foundLandmarks.isEmpty()){
			int dynamic = foundLandmarks.size() * 2 + 2;
			int measured = foundLandmarks.size() * 2;
			try {
				kf = new JKalmanFilter(dynamic, measured);
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			Mat postState = new Mat(1, dynamic, CvType.CV_64F);
			postState.put(0, 0, rover_x);
			postState.put(0, 1, rover_y);
			for(int i = 0; i < foundLandmarks.size(); i++){
				double x = foundLandmarks.get(i).mapLocation.x;
				double y = foundLandmarks.get(i).mapLocation.y;
				postState.put(0,i*2+2,x);
				postState.put(0,i*2+3, y);
			}
			kf.setState_post(postState);
			
		}
		
		
	}
	
	public void kalmanFilter(){
		Mat predicted = new Mat();
		predicted = kf.Predict();
		
		int measured = foundLandmarks.size() * 2;
		Mat measurement = new Mat(1, measured, CvType.CV_64F);

		for(int i = 0; i < foundLandmarks.size(); i++){
			if(foundLandmarks.get(i).updatedLocation == null){
				double x = foundLandmarks.get(i).mapLocation.x;
				double y = foundLandmarks.get(i).mapLocation.y;
				measurement.put(0,i*2,x);
				measurement.put(0,i*2+1, y);
			}
			else{
				double x = foundLandmarks.get(i).updatedLocation.x;
				double y = foundLandmarks.get(i).updatedLocation.y;
				measurement.put(0,i*2,x);
				measurement.put(0,i*2+1, y);
			}
			
		}
		Mat corrected = new Mat();
		corrected = kf.Correct(measurement);

	}
	
	public void drawLandmark(Landmark data){
			Core.circle(areaMap, data.mapLocation, (int)data.length/2, red);
		}
		
	
	public Dimension getPreferredSize() {
        return new Dimension(500, 500);
    }
	
	public BufferedImage getFeatureImg(){
		BufferedImage featImage = null;
		MatOfByte pixels = new MatOfByte();
    	Highgui.imencode(".png", areaMap, pixels);
    	byte[] byteArray = pixels.toArray();
    	
    	try {
            InputStream in = new ByteArrayInputStream(byteArray);
            featImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
		return featImage;
	}
	
	 public void paint(Graphics g) {
		 BufferedImage featImage = null;
			MatOfByte pixels = new MatOfByte();
	    	Highgui.imencode(".png", areaMap, pixels);
	    	byte[] byteArray = pixels.toArray();
	    	
	    	try {
	            InputStream in = new ByteArrayInputStream(byteArray);
	            featImage = ImageIO.read(in);
	        } catch (Exception e) {
	            e.printStackTrace();
	        }
	        g.drawImage(featImage, 0, 0, null);
	    }
	
	public class Landmark{
		public Point mapLocation;
		public Point updatedLocation = null;
		public double length;
		public MatOfPoint contour;
		public boolean observed;
		
		public Landmark(Point loc,double len, MatOfPoint contourlines){
			this.mapLocation = loc;
			this.length = len;
			this.contour = contourlines;
			this.observed = false;
		}
		
		public void updateLocation(Point updated){
			this.updatedLocation = updated;
		}
		
		public Point getNewLocation(){
			return updatedLocation;
		}
		
		public MatOfPoint getContour(){
			return contour;
		}
		
		public void isObserved(){
			this.observed = true;
		}
	}
	

}
