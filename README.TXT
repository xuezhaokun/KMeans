!!!****this project does not include the data. If I include the all the data, the size of the zip file will be larger than the max allowbale size for provide system.****

How to setup:
	import the .zip file as existing archive project to workspace
	create a folder under the project called "data" and import all the data files
	build configure path
	import weka.jar file as referenced lib

How to get the results:
	When you run the KMeans.java, it will output the instructions and results for part 3.2 and 3.3 in console.
	And there will be words to explain each part of results in console.
	
How to read data:
	I stored all the results data in results folder.
	There are three folders under results folder. They are rawdata, data_for_plots and plots
		rawdata: This folder contains five files. output.txt is the output from console log when you run the project. And I separate the output results to
				corresponding dataset.
		data_for_plots: This folder contains all the data files cleaned from rawdata folder for plotting in gnuplot.
						The first part of filename represents the dataset name and the second part relates to the question part. 
						For example, "artdata_32.dat" is the data results of artdata for question part 3.2
		plots: contains all the plots results from gnuplot based on data files in data_for_plots folder
For analysis of the project, check report.pdf in results folder.