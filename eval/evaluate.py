#!/usr/bin/env python3
#
# Copyright (c) 2017-present, All rights reserved.
# Written by Julien Tissier <30314448+tca19@users.noreply.github.com>
#
# This file is part of Dict2vec.
#
# Dict2vec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Dict2vec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License at the root of this repository for
# more details.
#
# You should have received a copy of the GNU General Public License
# along with Dict2vec.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import math
import argparse
import numpy as np
import scipy.stats as st
import pickle

RUN = "fits/"
FILE_DIR = "../"
EVAL_DIR = "data"
missed_pairs = dict()
missed_words = dict()

word_dictionary = {}
embedding_matrix = np.array([1])

performances = {}

def tanimotoSim(v1, v2):
    """Return the Tanimoto similarity between v1 and v2 (numpy arrays)"""
    dotProd = np.dot(v1, v2)
    return dotProd / (np.linalg.norm(v1)**2 + np.linalg.norm(v2)**2 - dotProd)


def cosineSim(v1, v2):
    """Return the cosine similarity between v1 and v2 (numpy arrays)"""
    dotProd = np.dot(v1, v2)
    return dotProd / (np.linalg.norm(v1) * np.linalg.norm(v2))


def init_results():
    results = dict()

    print("List test datasets...")
    """Read the filename for each file in the evaluation directory"""
    for filename in os.listdir(EVAL_DIR):
        if not filename in results:
            results[filename] = []
    print(results)
    print("Done.")
    return results

def init_we(matrix_loc):
    global word_dictionary
    global embedding_matrix
    print("Load word embeddings...")
    word_dictionary = pickle.load(open(FILE_DIR + RUN + "dictionary.pkl", "rb"))
    embedding_matrix = np.load(FILE_DIR + RUN + matrix_loc)
    print("Done.")

def evaluate(results):
    # Compute Spearman rank coefficient for each evaluation file

    wordToNum = {}

    for word, ix in word_dictionary.items():
        #print(ix, word)
        wordToNum[word] = ix

    #print(word_dictionary.items()[:10])
    mat = embedding_matrix

    # step 1 : iterate over each evaluation data file and compute spearman
    for filename in results:
        pairs_not_found, total_pairs = 0, 0
        words_not_found, total_words = 0, 0
        with open(os.path.join(EVAL_DIR, filename)) as f:
            file_similarity = []
            embedding_similarity = []
            for line in f:
                w1, w2, val = line.split()
                w1, w2, val = w1.lower(), w2.lower(), float(val)
                total_words += 2
                total_pairs += 1
                if not w1 in wordToNum:
                    words_not_found += 1
                if not w2 in wordToNum:
                    words_not_found += 1

                if not w1 in wordToNum or not w2 in wordToNum:
                    pairs_not_found += 1
                else:
                    v1, v2 = mat[wordToNum[w1]], mat[wordToNum[w2]]
                    cosine = cosineSim(v1, v2)
                    file_similarity.append(val)
                    embedding_similarity.append(cosine)

                    #tanimoto = tanimotoSim(v1, v2)
                    #file_similarity.append(val)
                    #embedding_similarity.append(tanimoto)

            rho, p_val = st.spearmanr(file_similarity, embedding_similarity)
            results[filename].append(rho)
            missed_pairs[filename] = (pairs_not_found, total_pairs)
            missed_words[filename] = (words_not_found, total_words)

    return results


def stats(results):
    """Compute statistics on results"""
    title = "{}| {}| {}| {}| {}| {} ".format("Filename".ljust(16),
                              "AVG".ljust(5), "MIN".ljust(5), "MAX".ljust(5),
                              "STD".ljust(5), "Missed words/pairs")
    print(title)
    print("="*len(title))

    weighted_avg = 0
    total_found  = 0

    for filename in sorted(results.keys()):
        average = sum(results[filename]) / float(len(results[filename]))
        minimum = min(results[filename])
        maximum = max(results[filename])
        std = sum([(results[filename][i] - average)**2 for i in
                   range(len(results[filename]))])
        std /= float(len(results[filename]))
        std = math.sqrt(std)

        # For the weighted average, each file has a weight proportional to the
        # number of pairs on which it has been evaluated.
        # pairs evaluated = pairs_found = total_pairs - number of missed pairs
        pairs_found = missed_pairs[filename][1] - missed_pairs[filename][0]
        weighted_avg += pairs_found * average
        total_found  += pairs_found

        # ratio = number of missed / total
        ratio_words = missed_words[filename][0] / missed_words[filename][1]
        ratio_pairs = missed_pairs[filename][0] / missed_pairs[filename][1]
        missed_infos = "{:.0f}% / {:.0f}%".format(
                round(ratio_words*100), round(ratio_pairs*100))

        print("{}| {:.3f}| {:.3f}| {:.3f}| {:.3f}| {} ".format(
              filename.ljust(16),
              average, minimum, maximum, std, missed_infos.center(20)))

    print("-"*len(title))
    print("{0}| {1:.3f}".format("W.Average".ljust(16),
                                weighted_avg / total_found))
    return (weighted_avg / total_found)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
             description="Evaluate semantic similarities of word embeddings.",
             )

    #parser.add_argument('filenames', metavar='FILE', nargs='+',
    #                    help='Filename of word embedding to evaluate.')

    current_dir = os.listdir("./")

    all_performances = {}

    if "performances.pkl" in current_dir:
        all_performances = pickle.load(open("performances.pkl","rb"))


    args = parser.parse_args()

    arr = os.listdir(FILE_DIR + RUN)

    matrix_locs = []
    for filename in arr:
        print(filename)
        if "rhos" in filename:
            matrix_locs.append(filename)

    #["trained_rhos.npy", "rhos_at_0.npy", "rhos_at_1.npy", "rhos_at_2.npy", "rhos_at_3.npy", "rhos_at_4.npy",  "rhos_at_5.npy"]

    performance_list = [0.0] * 20

    for matrix_loc in matrix_locs:
        init_we(matrix_loc)
        results = init_results()
        #for f in args.filenames:
        results = evaluate(results)
        performance = stats(results)

        performances[matrix_loc] = performance

        for file, performance in performances.items():
            print(file, performance)

            if "rhos_at_" in file:
                index = file.split("rhos_at_")[-1].split(".")[0]
                index = int(index)

                performance_list[index] = performance


        print("BEST: ", max(performances))
        print(performance_list)

    all_performances[RUN] = performance_list

    pickle.dump(all_performances, open('performances.pkl','wb'))
