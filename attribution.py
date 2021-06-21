"""Authorship Attribution

Usage:
  attribution.py --words <filename>
  attribution.py --chars=<n> <filename>
  attribution.py (-h | --help)
  attribution.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --words
  --chars=<kn>  Length of char ngram [default: 3].

"""

import sys
import os
import math
from utils import process_document_words, process_document_ngrams, get_documents, extract_vocab, top_cond_probs_by_author
from docopt import docopt
import numpy as np
from obfuscator import Obfuscator
import seaborn as sns
import pandas as pd
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Authorship Attribution 1.1')


'''Default values for hyperparameters'''
feature_type = "words"
ngram_size = 3
testfile = "data/test/persuasion.txt"

if arguments["--words"]:
    feature_type = "words"
elif arguments["--chars"]:
    feature_type = "chars"
    ngram_size = int(arguments["--chars"])
testfile = arguments["<filename>"]


alpha = 0.0001
mu, sigma = 0, 0.2 # mean and standard deviation
classes = ["Austen", "Carroll", "Grahame", "Kipling"]
documents = get_documents(feature_type, ngram_size)

def count_docs(documents):
    return len(documents)

def count_docs_in_class(documents, c):
    count=0
    for values in documents.values():
        if values[0] == c:
            count+=1
    return count

def concatenate_text_of_all_docs_in_class(documents,c):
    words_in_class = {}
    for d,values in documents.items():
        if values[0] == c:
            words_in_class.update(values[2])
    return words_in_class

def train_naive_bayes(classes, documents):
    vocabulary = extract_vocab(documents)
    conditional_probabilities = {}
    for t in vocabulary:
        conditional_probabilities[t] = {}
    priors = {}
    print("\n\n***\nCalculating priors and conditional probabilities for each class...\n***")
    for c in classes:
         priors[c] = count_docs_in_class(documents,c) / count_docs(documents)
         print("\nPrior for",c,priors[c])
         class_size = count_docs_in_class(documents, c)
         print("In class",c,"we have",class_size,"document(s).")
         words_in_class = concatenate_text_of_all_docs_in_class(documents,c)
         #print(c,words_in_class)
         print("Calculating conditional probabilities for the vocabulary.")
         denominator = sum(words_in_class.values())
         for t in vocabulary:
             if t in words_in_class:
                 conditional_probabilities[t][c] = (words_in_class[t] + alpha) / (denominator * (1 + alpha))
                 # print(t,c,words_in_class[t],denominator,conditional_probabilities[t][c])
             else:
                conditional_probabilities[t][c] = (0 + alpha) / (denominator * (1 + alpha))
    return vocabulary, priors, conditional_probabilities

def apply_naive_bayes(classes, vocabulary, priors, conditional_probabilities, test_document):
    scores = {}

    if feature_type == "chars":
        author, doc_length, words = process_document_ngrams(test_document,ngram_size)
    elif feature_type == "words":
        author, doc_length, words = process_document_words(test_document)
    for c in classes:
        scores[c] = math.log(priors[c])
        for t in words:
            if t in conditional_probabilities:
                for i in range(words[t]):
                    scores[c] += math.log(conditional_probabilities[t][c])
    print("\n\nNow printing scores in descending order:")
    probabilities = []
    for author in sorted(scores, key=scores.get, reverse=True):
        probabilities.append((author,scores[author]))
        print(author,"score:",scores[author])
    return probabilities

def plot_results(data, exp, xlabel, ylabel):
    df = pd.DataFrame(data)
    sns_plot = sns.relplot(data=df, kind="line")
    if feature_type == "chars":
        sns_plot.fig.suptitle("feature type: " + feature_type + " " + str(ngram_size), fontsize=11)
        file_name = feature_type + "_" + str(ngram_size)
    else:
        sns_plot.fig.suptitle("feature type: " + feature_type, fontsize=11)
        file_name = feature_type
    sns_plot.fig.subplots_adjust(top=0.9);
    sns_plot.set(xlabel=xlabel, ylabel=ylabel)
    sns_plot.savefig("results/output_" + exp + "_" + file_name + ".png")

def experiment_1(classes, file, conditional_probabilities, feature_type, vocabulary, priors):
    test_features = [0, 10, 100, 250, 500]
    obf_successes = []
    prob_results = {author: {0: None, 10: None, 100: None, 250: None, 500: None} for author in classes}
    obf = Obfuscator(file, "Austen", conditional_probabilities, feature_type)
    obf_file = "encrypted_doc.txt"

    for num in test_features:
        obf.obfuscate(type="encrypt", n_features=num)
        probs = apply_naive_bayes(classes, vocabulary, priors, conditional_probabilities, obf_file)
        obf_successes.append(obf.success(probs))
        for p in probs:
            prob_results[p[0]][num] = p[1]

    # plotting results
    plot_results(prob_results, "exp1", "number of encrypted features", "probability")

    return obf_successes


def experiment_2(file, conditional_probabilities, feature_type, classes, vocabulary, priors):
    obf = Obfuscator(file, "Austen", conditional_probabilities, feature_type)
    languages = [["nl"],["fr"],["ru"],["zh-cn"],["nl","fr"],["ru","zh-cn"]]

    # change the index of languages[index] in order to translate in the corresponding language
    # comment the next line to skip the obfuscation(translation) part and to use already existing obfuscated files
    obf.obfuscate(type="translate",file=file,languages=languages[0])

    obf_files = [f for f in listdir("translated") if isfile(join("translated", f))]
    prob_results = {author: {"original": None, "nl": None, "fr": None, "ru": None, "zh-cn": None} for author in classes}
    obf_successes = []

    # testing on the original file and storing probabilities
    original_probs = apply_naive_bayes(classes, vocabulary, priors, conditional_probabilities, file)
    for p in original_probs:
        prob_results[p[0]]["original"] = p[1]

    # testing the obfuscated files and storing probabilities
    for file in obf_files:
        probs = apply_naive_bayes(classes, vocabulary, priors, conditional_probabilities, "translated/"+file)
        obf_successes.append(obf.success(probs))
        for p in probs:
            lang = file.split("(")[1].split(")")[0]
            prob_results[p[0]][lang] = p[1]

    # plotting results
    plot_results(prob_results,"exp2","languages","probability")

    return obf_successes


vocabulary, priors, conditional_probabilities = train_naive_bayes(classes, documents)

for author in classes:
    print("\nBest features for",author)
    top_cond_probs_by_author(conditional_probabilities, author, 10)

# N.B. without obfuscation
apply_naive_bayes(classes, vocabulary, priors, conditional_probabilities, testfile)

# running experiments
results_exp1 = experiment_1(classes,testfile,conditional_probabilities,feature_type,vocabulary,priors)
results_exp2 = experiment_2(testfile,conditional_probabilities,feature_type,classes,vocabulary,priors)

print("SUCCESSES EXP1: ", results_exp1)
print("SUCCESSES EXP2: ", results_exp2)
