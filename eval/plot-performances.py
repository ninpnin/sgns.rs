import pickle
import matplotlib.pyplot as plt


all_performances = pickle.load(open("performances.pkl","rb"))
print(all_performances.keys())

selected = ["1000_k100", "1000_k25"]

for run in selected:
	run += "/"

	Y = all_performances[run]
	X = range(len(Y))

	plt.plot(X, Y)


plt.show()