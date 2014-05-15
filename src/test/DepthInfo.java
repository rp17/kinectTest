package test;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;

public class DepthInfo {
	private Mat depthMat;
	private Mat depthf;
	
	public void interpDepth(Mat depth){
		depthMat = depth;
		depthf = new Mat();
		Mat temp1 = new Mat();
		Mat temp2 = new Mat();
		Mat smallDepthf = new Mat();
		Core.MinMaxLocResult minMax = new Core.MinMaxLocResult();
		depthMat.convertTo(temp1, CvType.CV_64FC1);
		minMax = Core.minMaxLoc(temp1);
		temp1.convertTo(depthf, CvType.CV_8UC1,255.0/minMax.maxVal);
		Imgproc.resize(depthf, smallDepthf, new Size(), 0.2, 0.2, Imgproc.INTER_LINEAR);
		
		Mat smallDepthMask = new Mat(smallDepthf.size(), CvType.CV_8UC1);
		for(int i = 0; i < smallDepthf.rows(); i++){
			for(int j = 0; j < smallDepthf.cols(); j++){
				double[] cell = new double[1];
				double[] zero = {0.0};
				double[] max = {255.0};
				cell = smallDepthf.get(i, j);
				if(cell[0] == 0){
					smallDepthMask.put(i,j,max);
				}
				else
					smallDepthMask.put(i,j,zero);
			}
		}
		
		Photo.inpaint(smallDepthf, smallDepthMask, temp1, 5.0, Photo.INPAINT_TELEA);
		Imgproc.resize(temp1, temp2, depthf.size());
		
		Mat depthfMask = new Mat(depthf.size(), CvType.CV_8UC1);
		for(int i = 0; i < depthf.rows(); i++){
			for(int j = 0; j < depthf.cols(); j++){
				double[] cell = new double[1];
				double[] zero = {0.0};
				double[] max = {255.0};
				cell = depthf.get(i, j);
				if(cell[0] == 0)
					depthfMask.put(i, j, max);
				else
					depthfMask.put(i, j, zero);
			}
		}
		//Photo.inpaint(depthf, depthfMask, temp2, 5.0, Photo.INPAINT_TELEA);
		temp2.copyTo(depthf, depthfMask);
	}
	
	public Mat getInterpDepth(){
		return depthf;
	}
	
	public BufferedImage getFeatureImg(){
		BufferedImage featImage = null;
		MatOfByte pixels = new MatOfByte();
    	Highgui.imencode(".png", depthf, pixels);
    	byte[] byteArray = pixels.toArray();
    	
    	try {
            InputStream in = new ByteArrayInputStream(byteArray);
            featImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
		return featImage;
	}

}
