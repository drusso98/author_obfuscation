# Authorship obfuscation

## Obfuscation

The opposite of authorship attribution is obfuscation. You are Jane Austen and you don't want to have your texts identify you. What can you do to prevent this? Try out different methods and see if you can fool the system. 

Experiments 1 and 2 try to challenge the Naive Bayes classifier using encryption methods and a language translation approach. More details are in the ML4NLP_analysis.pdf file. 


## Requirements

You'll need the docopt package to run the code. If you're on PythonAnywhere, this will be installed by default. If you need to install it on your system, do:

    sudo pip3 install docopt


## Run the code

To run the code, type:

    python3 attribution.py --words data/emma.txt

Or alternatively:

    python3 attribution.py --chars 3 data/emma.txt

The first alternative computes a model over words. The second alternative uses character ngrams of the size given to the system.

The output of the code tells you which operations the classifier is currently performing: computing prior probabilities, conditional probabilities, etc. Then, for illustration, it outputs the 10 features with highest conditional probability for the class under consideration (i.e. for each author). It gives you an idea of which words / ngrams are most important for each author. Finally, you get sorted log figures for the probability of each class. The first entry in that list is the author guessed by the system.
