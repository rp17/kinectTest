package test;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.openni.*;
import java.nio.ByteBuffer;
import java.util.List;
import java.awt.*;
import java.awt.image.*;

public class KinectViewer extends Component {
	private static final long serialVersionUID = 1L;;
	private BufferedImage featImage;
    public static int width, height;
    private VideoFrameRef lastFrame;
    private VideoFrameRef depthFrame;
    private VideoStream videoStream;
    private VideoStream depthStream;
    private Mat image;
    private Mat depthInfo;
 
    public KinectViewer() {
    	OpenNI.initialize();
        List<DeviceInfo> devicesInfo = OpenNI.enumerateDevices();
        if (devicesInfo.isEmpty()) {
            System.out.print("No device is connected\n");
            return;
        }
        
        Device device = Device.open(devicesInfo.get(0).getUri());

        SensorType type = SensorType.COLOR;
        SensorType depth = SensorType.DEPTH;
        depthStream = VideoStream.create(device, depth);
        videoStream = VideoStream.create(device, type);
        videoStream.start();
        depthStream.start();
        depthFrame = depthStream.readFrame();
        lastFrame = videoStream.readFrame();
        width = lastFrame.getWidth();
        height = lastFrame.getHeight();
        image = new Mat(lastFrame.getHeight(),lastFrame.getWidth(),CvType.CV_8UC3);
        depthInfo = new Mat(depthFrame.getHeight(),depthFrame.getWidth(),CvType.CV_16UC1);
    }
    
    
    public Dimension getPreferredSize() {
        return new Dimension(width, height);
    }
    
    
    public void writePixels(ByteBuffer pixels, int width, int height) {
        int bufferInd = 0;
        byte[][] rgbbytes = new byte[3][4];
        byte[] pixel = new byte[3];
        for (int row = 0; row < height; row++) {
            for (int col = width; col > 0; col--) {
                int R, G, B;
                B = pixels.get(bufferInd++);
                G = pixels.get(bufferInd++);
                R = pixels.get(bufferInd++);
                
                for(int i = 0; i < 4; i++){
                	rgbbytes[0][i] = (byte)(R >>> (i * 8));
                	rgbbytes[1][i] = (byte)(G >>> (i * 8));
                	rgbbytes[2][i] = (byte)(B >>> (i * 8));
                }
                pixel[0] = rgbbytes[0][0];
                pixel[1] = rgbbytes[1][0];
                pixel[2] = rgbbytes[2][0];
                image.put(row, col, pixel);
            }
        }
        
    }
    
    public void writeDepth(ByteBuffer pixels, int width, int height) {
        int bufferInd = 0;
        byte first;
        byte second;
        short[] data = new short[1];
        for (int row = 0; row < height; row++) {
            for (int col = width; col > 0; col--) {
                first = pixels.get(bufferInd++);
                second = pixels.get(bufferInd++);
                
                data[0] = twoBytesToShort(second,first);
                depthInfo.put(row, col, data);
            }
        }
        
    }
    
    public static short twoBytesToShort(byte b1, byte b2) {
        return (short) ((b1 << 8) | (b2 & 0xFF));
}
    
    public void update(){
        lastFrame = videoStream.readFrame();
        depthFrame = depthStream.readFrame();
        writePixels(lastFrame.getData(),lastFrame.getWidth(),lastFrame.getHeight());
        writeDepth(depthFrame.getData(),depthFrame.getWidth(),depthFrame.getHeight());
        
    }
    
    public Mat getImageMat(){
    	return image;
    }
    
    public Mat getDepthMat(){
    	return depthInfo;
    }
    
    public void setFeatures(BufferedImage img){
    	this.featImage = img;
    }
    
    public void exit(){
    	videoStream.stop();
    	depthStream.stop();
    }
    

    public void paint(Graphics g) {
        g.drawImage(featImage, 0, 0, null);
    }
}
		



