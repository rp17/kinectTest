package test;

import javax.swing.JFrame;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import org.opencv.core.*;


public class KinectViewerApp implements Runnable{
	private KinectViewer viewer;
	private ObjectDetection detector;
	private DepthInfo depth;
	private Slam slam;
	private JFrame frame;
	private boolean shouldRun = true;
	
    public KinectViewerApp (JFrame frame)
    {
        	this.frame = frame;
        	frame.addKeyListener(new KeyListener()
    		{
    			@Override
    			public void keyTyped(KeyEvent arg0) {}
    			@Override
    			public void keyReleased(KeyEvent arg0) {}
    			@Override
    			public void keyPressed(KeyEvent arg0) {
    				if (arg0.getKeyCode() == KeyEvent.VK_ESCAPE)
    				{
    					shouldRun = false;
    				}
    			}
    		});
    }

    public static void main(String s[]) {
    	System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        JFrame f = new JFrame("Kinect Viewer");
        JFrame g = new JFrame("Map");
        f.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {System.exit(0);}
        });
        g.addWindowListener(new WindowAdapter() {
           public void windowClosing(WindowEvent e) {System.exit(0);}
        });
        KinectViewerApp app = new KinectViewerApp(f);
        app.viewer = new KinectViewer();
        app.detector = new ObjectDetection();
        app.depth = new DepthInfo();
        app.slam = new Slam();
        f.add("Center", app.viewer);
        g.add("Center", app.slam);
        f.pack();
        g.pack();
        f.setVisible(true);
        g.setVisible(true);
        
        app.run();
    }

    public void run() {
        while(shouldRun){
            viewer.update();
            detector.detectObjs(viewer.getImageMat());
            depth.interpDepth(viewer.getDepthMat());
            slam.findNewObjects(detector.getObjectList(), depth.getInterpDepth());
            viewer.setFeatures(detector.getFeatureImg());
            viewer.repaint();
            slam.repaint();
            try{
            	Thread.sleep(1000);
            }catch(InterruptedException e){
            	e.printStackTrace();
            }
            }
        viewer.exit();
        frame.dispose();
    }
    static { 
    	try { 
    	System.load("D:\\tools\\openni2\\redist\\OpenNI2.dll"); 
    	} 
    	catch (Exception e) 
    	{ 
    	e.printStackTrace(); 
    	} 
    	} 
}