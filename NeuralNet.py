#!/usr/bin/python

#An implementation of a single sigmoid unit
#Jeffrey Brown

from operator import mul
from math import exp

def error(actual,predicted):
	return (actual-predicted)

def sigmoid(x):
	return 1/(1+exp(-x))

def d_sigmoid(x):
	return (sigmoid(x) * (1 - sigmoid(x)))

def prediction(weights,inputs):
	return sigmoid(sum([mul(x,y) for x,y in zip(weights,inputs)]))

def update(weights,inputs,output,learningRate):
	return ([(w + learningRate * error(output,prediction(weights,inputs)) * x * d_sigmoid(sum([mul(x,y) for x,y in zip(weights,inputs)]))) for w,x in zip(weights,inputs)])

def readFile(fileName):
	labels = []
	trainingExamples = []

	ifile = open(fileName,'r')
	labels = ifile.readline().strip().split()

	for line in ifile:
		if line.strip():
			trainingExamples.append(list(map(int,line.strip().split())))

	return (labels,trainingExamples)

def buildUnit(trainFile,learningRate,iterations):
	labels,data = readFile(trainFile)
	weights = [0 for x in data[:-1]]
	correct = 0

	for n in range(0,iterations):
		weights = update(weights,data[n%len(data)][:-1],data[n%len(data)][-1],learningRate)
		print("After iteration %d: "%(n+1), ", ".join(["w(%s) = %.4f"%(x,y) for x,y in zip(labels[:-1],weights)]),", output = %.4f"%prediction(weights,data[n%len(data)][:-1]),sep="")
		if prediction(weights,data[n%len(data)][:-1]) >= 0.5 and data[n%len(data)][-1] == 1:
			correct += 1
		elif prediction(weights,data[n%len(data)][:-1]) < 0.5 and data[n%len(data)][-1] == 0:
			correct += 1
	
	print ("accuracy: ",correct/iterations)
