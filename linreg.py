# linreg.py
#
# Vamsi Krishna Kovuru
# vkovuru@uncc.edu
#
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np
from pyspark import SparkContext

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print >> sys.stderr, "Usage: linreg <datafile>"
		exit(-1)
	sc = SparkContext(appName="LinearRegression")
	yxlines = sc.textFile(sys.argv[1])\
				.map(lambda line: line.split(','))		# Input yx file has y_i as the first element of each line
	

	#
	# Defining function to create matrices
	#
	def Xmatrix(line):
		lines=line
		lines[0]=1
		X = np.array([lines], dtype=float)   			# Preparing X matrix with only x values
		XT = X.transpose()                   			# Obtaining the Transpose matrix for X
		return X,XT

	#
	# Definig the function to calculate XTX for every rdd
	#
	def FindXXT(line):
		x,y=Xmatrix(line)
		return (np.dot(y, x))             				# dot product of XT * X
	

	#
	# Implementation of XTX on the evry rdd using Map function 
	#
	# Reduce method to get the matrix
	#
	out1 = yxlines.map(FindXXT)\
		  		  .reduce(lambda a, b: np.add(a, b)) 
	XIN = np.linalg.inv(out1) 
	
	# Defining the XTY function 
	def xtransy(line):
		return (float(line[0])*Xmatrix(line)[1])        # Calculating XT * Y

	#
	# Implemntation of XTY
	#
	# Map method to calculate the XTY using xtransy function
	#
	# Reduce method to get  B matrix = XT * Y
	#
	XTY = yxlines.map(xtransy)\
	.reduce(lambda a, b: np.add(a, b))  
	beta = np.dot(XIN, XTY)								# Calculating the Beta using least mean square error
	

	print ("XIN:",XIN)
	print ("XTY:",XTY)     
	
	# Print the linear regression coefficients in desired output format
	print ("beta: ")
	for coeff in beta:
		print (coeff)
	sc.stop()


