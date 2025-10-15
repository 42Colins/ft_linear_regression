def getThetaValues() :
	thetas = open("thetaValues.txt")
	values = thetas.read().splitlines()
	return values
	
def estimatePrice(mileage, theta0, theta1) :
	return (theta0 + (theta1 * mileage))

if __name__ == "__main__" :
	try :
		miles = float(input("What is the mileage of your car ?\n"))
		if (miles < 0) :
			raise Exception ("The mileage can't be a negative value !")
		thetas = getThetaValues()
		price = estimatePrice(float(miles), float(thetas[0]), float(thetas[1]))
		print("The price should be", int(price))
	except Exception as e :
		print(e)