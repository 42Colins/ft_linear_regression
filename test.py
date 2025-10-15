import csv
import matplotlib.pyplot as plt
import numpy as np

def parse(dataname) :
	"""Parse the data.csv in the format I want"""
	i = 0
	j = 0
	with open('data.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			reader = csv.DictReader(csvfile)
			i += 1
			
	tab = np.zeros((i, 2), dtype=float) 
	with open('data.csv', newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			tab[j, 0] = float(row['km'])
			tab[j, 1] = float(row['price'])
			j += 1
	return tab

def drawGraph(km, price, theta, X) :
	"""Draw the graph that proves our linear regression"""
	plt.scatter(km, price)
	plt.plot(km, F_model(X, theta), c='black')
	plt.ylabel('price')
	plt.xlabel('km')
	plt.show()

def getThetaValues() :
	thetas = open("thetaValues.txt")
	values = thetas.read().splitlines()
	return values

def F_model(X, theta) :
	"""f(x) = ax+b"""
	return (np.dot(X, theta))
	
def F_cout(X, theta, price) :
	""""Cost function"""
	return ((1/(2*len(X))) * np.sum((F_model(X, theta) - price) ** 2))

def gradient(X, theta, price) :
	return ((1/len(X)) * (X.T.dot(F_model(X, theta) - price)))
	
def gradientDescent(X, theta, price, learningRate) :
	"""Do the gradient descent"""
	newTheta = theta - 1
	while F_cout(X, theta, price) != F_cout(X, newTheta, price) :
		newTheta = theta
		theta = theta - learningRate * gradient(X, theta, price)
	return theta

def unnormalize_theta(theta_norm, mean_km, std_km, mean_price, std_price):
	"""Convert normalized theta back to original scale"""
	theta_original = np.zeros_like(theta_norm)
	theta_original[0] = theta_norm[0] * (std_price / std_km)	
	theta_original[1] = mean_price + std_price * theta_norm[1] - theta_original[0] * mean_km

	return theta_original	

def saveThetas(theta) :
	"""Save the values of theta0 and theta1 in thetaValues.txt"""
	with open('thetaValues.txt', 'w') as file:
	    text1 = f"{theta[1]}"
	    file.write(text1[1:-1] + "\n")
	    text2 = f"{theta[0]}"
	    file.write(text2[1:-1] + "\n")

# def isFileAccessible()	

def init():
	"""Initialize every needed value"""
	learningRate = 0.00001
	tab = parse('data.csv')
	km = np.zeros((len(tab), 1), dtype=float)
	price = np.zeros((len(tab), 1), dtype=float)
	for i in range(len(tab)) :
		km[i] = (tab[i, 0])
		price[i] = (tab[i, 1])
	mean_km = km.mean()
	std_km = km.std()
	mean_price = price.mean()
	std_price = price.std()
	i = 0
	X = np.hstack((km, np.ones(km.shape)))
	X_normalized = X.copy()
	X_normalized[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
	Y_normalized = price.copy()
	Y_normalized[:, 0] = (price[:, 0] - price[:, 0].mean()) / price[:, 0].std()
	theta = np.random.rand(2, 1)
	return X_normalized, Y_normalized, learningRate, mean_km, std_km, mean_price, std_price, km, price, X, theta


def linearRegression() :
	X_normalized, Y_normalized, learningRate, mean_km, std_km, mean_price, std_price, km, price, X, theta = init()
	theta = gradientDescent(X_normalized, theta, Y_normalized, learningRate)
	finaltheta = unnormalize_theta(theta, mean_km, std_km, mean_price, std_price)
	drawGraph(km, price, finaltheta, X)
	saveThetas(finaltheta)

if (__name__ == "__main__") :
	"""MAIN FUNCTION"""
	linearRegression()