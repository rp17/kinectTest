package radar;

import org.jfree.ui.RefineryUtilities;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class Tracker {
	private static double posp = 0;
	private static double pos;
	private static double vel;
	private static double alt;
	private static boolean firstRun = true;
	
	private static Mat R;
	private static Mat a;
	private static Mat q;
	private static Mat x;
	private static Mat p;
	

	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		double dt = 0.05;
		int nSamples = (int) (20/dt)+1;
		double[] t = new double[nSamples];
		
		for(int i = 0; i < nSamples; i++)
			t[i] = dt*i;
		
		Mat xSaved = Mat.zeros(nSamples, 3, CvType.CV_64F);
		Mat zSaved = Mat.zeros(nSamples, 1, CvType.CV_64F);
		Mat rSaved = Mat.zeros(nSamples, 1, CvType.CV_64F);
		
		for(int i = 0; i < nSamples; i++){
			double r = getRadar(dt);
			radarEKF(r, dt);
			
			xSaved.put(i, 0, pos);
			xSaved.put(i, 1, vel);
			xSaved.put(i, 2, alt);
			
			zSaved.put(i, 0, r);
			
			rSaved.put(i, 0, Math.sqrt(pos*pos + alt*alt));
		}
		
		final Plotter results = new Plotter("EKF Results", rSaved, zSaved, t);
        results.pack();
        RefineryUtilities.centerFrameOnScreen(results);
        results.setVisible(true);

	}
	
	public static void radarEKF(double r, double dt){
		if(firstRun){
			//a is the equation of motion or the Jacobian of the function f(x)
			a = Mat.eye(3, 3, CvType.CV_64F);
			a.put(0, 1, dt);
			
			//q is the processing noise covariance
			q = Mat.zeros(3, 3, CvType.CV_64F);
			q.put(1,1,0.001);
			q.put(2,2,0.001);
			
			//R is the measurement noise covariance
			R = Mat.ones(1, 1, CvType.CV_64F);
			Core.multiply(R, new Scalar(10), R);
			
			//x is the initial position with guessed values
			x = Mat.zeros(1, 3, CvType.CV_64F);
			x.put(0, 1, 90);
			x.put(0, 2, 1100);
			x = x.t();
			
			//p is the error covariance matrix
			p = Mat.eye(3, 3, CvType.CV_64F);
			Core.multiply(p, new Scalar(10), p);
			
			firstRun = false;
		}
		//Calculate the Jacobian of x to predict the motion model
		Mat h = Hjacob(x);
		
		//Predict the state from the previous motion model a
		Mat xp = new Mat();
		Core.gemm(a, x, 1, new Mat(), 0, xp);
		
		//Predict the error covariance
		Mat temp1 = new Mat();
		Core.gemm(a,p,1,new Mat(),0,temp1);
		Mat Pp = new Mat();
		Core.gemm(temp1,a.t(),1,q,1,Pp);
		
		//Compute the Kalman gain
		Mat temp2 = new Mat();
		Core.gemm(Pp, h.t(), 1, new Mat(), 0, temp2);
		Mat temp3 = new Mat();
		Core.gemm(h, temp2, 1, R, 1, temp3);
		Mat K = new Mat();
		Core.gemm(temp2, temp3.inv(), 1, new Mat(), 0, K);
		
		//Compute the estimate
		Mat temp4 = new Mat();
		Core.multiply(K, new Scalar(r - hx(xp)), temp4);
		Core.add(xp, temp4, x);
		
		//Compute the error covariance
		Mat temp5 = new Mat();
		Core.gemm(K, h, 1, new Mat(), 0, temp5);
		Mat temp6 = new Mat();
		Core.gemm(temp5, Pp, 1, new Mat(), 0, temp6);
		Core.subtract(Pp, temp6, p);
		
		double[] value = x.get(0, 0);
		pos = value[0];
		value = x.get(1, 0);
		vel = value[0];
		value = x.get(2, 0);
		alt = value[0];
		
		
	}
	
	public static Mat Hjacob(Mat xp){
		Mat H = Mat.zeros(1, 3, CvType.CV_64F);
		
		double[] x1 = xp.get(0, 0);
		double[] x3 = xp.get(2, 0);
		
		H.put(0, 0, (x1[0]/(Math.sqrt(x1[0]*x1[0] + x3[0]*x3[0]))));
		H.put(0, 2, (x3[0]/(Math.sqrt(x1[0]*x1[0] + x3[0]*x3[0]))));
		return H;
	}
	
	public static double hx(Mat xhat){
		double[] x1 = xhat.get(0, 0);
		double[] x3 = xhat.get(2, 0);
		
		double zp = Math.sqrt(x1[0]*x1[0] + x3[0]*x3[0]);
		return zp;
	}
	
	public static double getRadar(double dt){
		double vel = 100 + 5*Math.random();
		double alt = 1000 + 10*Math.random();
		
		double pos = posp + vel*dt;
		
		double v = 0 + pos*0.05*Math.random();
		
		double r = Math.sqrt(pos*pos + alt*alt) + v;
		
		posp = pos;
		return r;
	}
	

}
