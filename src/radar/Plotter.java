package radar;
import java.awt.Color;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.opencv.core.Mat;

@SuppressWarnings("serial")
public class Plotter extends ApplicationFrame {


    public Plotter(final String title, Mat rResults, Mat zResults, double[] t) {
        super(title);
        final XYDataset dataset = createDataset(rResults, zResults, t);
        final JFreeChart chart = createChart(dataset);
        final ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 500));
        setContentPane(chartPanel);

    }
	    
	    /**
	     * Creates a sample dataset.
	     * 
	     * @return a sample dataset.
	     */
    private XYDataset createDataset(Mat rResults, Mat zResults, double[] t) {
	        
	        final XYSeries series1 = new XYSeries("Estimated");
	        final XYSeries series2 = new XYSeries("Measured");
	        
	        for(int i = 0; i < t.length; i++){
	        	double[] value = rResults.get(i, 0);
	        	series1.add(t[i], value[0] );
	        	value = zResults.get(i, 0);
	        	series2.add(t[i], value[0]);
	        }
	        

	        final XYSeriesCollection dataset = new XYSeriesCollection();
	        dataset.addSeries(series1);
	        dataset.addSeries(series2);
	                
	        return dataset;
	        
	    }
	    
	    /**
	     * Creates a chart.
	     * 
	     * @param dataset  the data for the chart.
	     * 
	     * @return a chart.
	     */
	    private JFreeChart createChart(final XYDataset dataset) {
	        
	        // create the chart...
	        final JFreeChart chart = ChartFactory.createXYLineChart(
	            "Real and Predicted Measurements",      // chart title
	            "Time",                      // x axis label
	            "Distance",                      // y axis label
	            dataset,                  // data
	            PlotOrientation.VERTICAL,
	            true,                     // include legend
	            true,                     // tooltips
	            false                     // urls
	        );

	        // NOW DO SOME OPTIONAL CUSTOMISATION OF THE CHART...
	        chart.setBackgroundPaint(Color.white);

//	        final StandardLegend legend = (StandardLegend) chart.getLegend();
	  //      legend.setDisplaySeriesShapes(true);
	        
	        // get a reference to the plot for further customisation...
	        final XYPlot plot = chart.getXYPlot();
	        plot.setBackgroundPaint(Color.lightGray);
	    //    plot.setAxisOffset(new Spacer(Spacer.ABSOLUTE, 5.0, 5.0, 5.0, 5.0));
	        plot.setDomainGridlinePaint(Color.white);
	        plot.setRangeGridlinePaint(Color.white);
	        
	        final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
	        renderer.setSeriesShapesVisible(1, false);
	        renderer.setSeriesShapesVisible(0, false);
	        plot.setRenderer(renderer);

	        // change the auto tick unit selection to integer units only...
	        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
	        rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
	        // OPTIONAL CUSTOMISATION COMPLETED.
	                
	        return chart;
	        
	    }

}
