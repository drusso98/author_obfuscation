from deep_translator import GoogleTranslator
import random


def preprocess_doc(doc):
    file = open(doc, "r", encoding="utf-8")
    text = file.read()
    file.close()
    chunks = text.split("\n\n")
    chunks = [chunk.replace("\n", " ") for chunk in chunks]
    return chunks


chunks = preprocess_doc("data/test/persuasion.txt")
# keeping only the book parts
del chunks[1071:]
del chunks[:19]

# getting random paragraphs from the book
random.seed(2)
sample = []
for i in range(0, 3):
    idx = random.randrange(0, len(chunks))
    sample.append(chunks[idx])
    ++i

# translating paragraphs and storing them in a dict
evaluation = {}
languages = [["nl"], ["fr"], ["ru"], ["zh-cn"], ["nl", "fr"], ["ru", "zh-cn"]]

for i, el in enumerate(sample):
    evaluation[i] = {"original": el}
    for l in languages:
        for language in l:
            translated = GoogleTranslator(target=language).translate(el)
        translated = GoogleTranslator(target='en').translate(translated)
        key = "_".join(l)
        evaluation[i][key] = translated

# saving in a txt file
file = open("evaluation.txt", "a", encoding="utf-8")
for i, el in evaluation.items():
    file.write("\n" + str(i))
    for lang, text in el.items():
        file.write("\n\t" + lang + " :\n\t" + text)
file.close()
