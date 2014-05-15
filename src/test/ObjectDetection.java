package test;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

public class ObjectDetection {
	private Mat imageToDetect;
	private Mat grayImage;
	private Mat bwImage;
	private ArrayList<MatOfPoint> contours;
	private ArrayList<MatOfPoint> objects;
	private BufferedImage featImage;
	private Scalar color_green;
	
	
	public ObjectDetection(){
		imageToDetect = new Mat();
		grayImage = new Mat();
		bwImage = new Mat();
		color_green = new Scalar(0,255,0);
		contours = new ArrayList<MatOfPoint>();
		objects = new ArrayList<MatOfPoint>();
	}
	
	public void detectObjs(Mat img){
		contours.clear();
		objects.clear();
		imageToDetect = img;
		Mat hierarchy = new Mat();
		Imgproc.cvtColor(imageToDetect,grayImage,Imgproc.COLOR_RGB2GRAY);
		MatOfDouble mean = new MatOfDouble(), sigma = new MatOfDouble();
		Core.meanStdDev(grayImage, mean, sigma);
		double[] avg = mean.toArray();
		double[] stdiv = sigma.toArray();
		Imgproc.Canny(grayImage, bwImage, avg[0]-stdiv[0], avg[0]+stdiv[0]);
		Imgproc.findContours(bwImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
		
		for(int idx = 0; idx < contours.size(); idx++){
			MatOfPoint current = contours.get(idx);
			MatOfPoint2f curve = new MatOfPoint2f(current.toArray());
			MatOfPoint2f approx = new MatOfPoint2f();
			double epsilon = Imgproc.arcLength(curve, true)*0.01;
			Imgproc.approxPolyDP(curve, approx, epsilon, true);
			
			MatOfPoint foundObject = new MatOfPoint(approx.toArray());
			if(Math.abs(Imgproc.contourArea(current)) > 300)
				objects.add(current);
			
		}
		Imgproc.drawContours(imageToDetect, objects, -1, color_green);
		
		MatOfByte pixels = new MatOfByte();
    	Highgui.imencode(".png", imageToDetect, pixels);
    	byte[] byteArray = pixels.toArray();
    	
    	try {
            InputStream in = new ByteArrayInputStream(byteArray);
            featImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
	
	public ArrayList<MatOfPoint> getObjectList(){
		return objects;
	}
	
	public BufferedImage getFeatureImg(){
		return featImage;
	}

}
