import numpy as np
import pickle

# Script to convert the 1D array to embedding matrix

dimensionality = 96

f = open("unigram.txt", "r").readlines()

vocab_dict = {}
vocab = []
for line in f:
	s = line.split()
	vocab.append(s[-1])
	vocab_dict[s[-1]] = int(s[0])

vocab_len = len(vocab)
print(vocab_dict)

pickle.dump(vocab_dict, open("fits/dictionary.pkl", "wb"))

w = np.load("word.npy")

print(w)

def word(i):
	start = dimensionality * i
	end = dimensionality * (i+1)

	return w[start:end]

def angle(i,j):
	w1 = word(i)
	w2 = word(j)

	norm1 = np.linalg.norm(w1)
	norm2 = np.linalg.norm(w2)

	dot = w1 @ w2
	return dot / (norm1 * norm2)



for comparison_ix in range(10):
	comparison_word = vocab[comparison_ix]

	#print("Compare word '", comparison_word, "' at index", comparison_ix)
	top1 = 0.0
	ix = 0

	for i in range(1, 10):
		#print("word 0 and word", i, ":", angle(0,i))

		new_angle = angle(comparison_ix, i)
		if new_angle > top1 and new_angle < 0.9999:
			top1 = new_angle
			ix = i

	print(comparison_word, vocab[ix], ", ", top1)


print("Convert to 2D...")
embedding = np.zeros((vocab_len, dimensionality))

print(word(0))
print(word(1))
for w_ix in range(vocab_len):
	embedding[w_ix] = word(w_ix)

np.save("fits/rhos_at_0.npy", embedding)
print("Converted.")