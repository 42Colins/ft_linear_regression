import csv
import matplotlib.pyplot as plt
import numpy as np

def parse(dataname) :
	i = 0
	j = 0
	with open('data.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			reader = csv.DictReader(csvfile)
			i += 1
			
	tab = np.zeros((i, 2), dtype=int) 
	with open('data.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			tab[j, 0] = int(row['km'])
			tab[j, 1] = int(row['price'])
			j += 1
	return tab

def drawGraph(km, price) :
	plt.scatter(km, price)
	plt.plot(km, F_model(X, theta), c='black')
	plt.ylabel('kms')
	plt.xlabel('price')
	plt.show()

def getThetaValues() :
	thetas = open("thetaValues.txt")
	values = thetas.read().splitlines()
	return values

def estimatePrice(theta0, theta1, mileage) :
	return (theta0 + (theta1 * mileage))

def F_model(X, theta) :
	return (np.dot(X, theta))
	
def F_cout(X, theta, price) :
	return (1/(2*len(X)) * np.sum((F_model(X, theta) - price) ** 2))

def gradient(X, theta, price) :
	return (1/len(X) * X.T.dot(F_model(X, theta) - price))
	
def gradientDescent(X, theta, price, learningRate) :
	i = 0
	for i in range (10000)

tab = parse('data.csv')
i = 0
km = np.zeros((len(tab), 1), dtype=int)
price = np.zeros((len(tab), 1), dtype=int)
for i in range(len(tab)) :
	km[i] = (tab[i, 0])
	price[i] = (tab[i, 1])
X = np.hstack((km, np.ones(km.shape)))
theta = np.random.rand(2, 1)
drawGraph(km, price)
print(F_cout(X, theta, price))