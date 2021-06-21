import re
from deep_translator import GoogleTranslator
from multiprocessing.dummy import Pool as ThreadPool

class Obfuscator:

    def __init__(self,document,author,probabilities,features):
        self.document = document
        self.author = author
        self.probabilities = probabilities
        self.features = features

    def set_document(self,document):
        self.document = document

    def get_document(self):
        return self.document

    def get_main_features(self,author,n):
        cps = {}

        for term, probs in self.probabilities.items():
            cps[term] = probs[author]

        cps = sorted(cps, key=cps.get, reverse=True)
        return list(cps[:n])

    def __encrypt(self,text,shift):
        crypted_text = ""
        for idx in range(len(text)):
            char = text[idx]
            crypted_text += chr(ord(char) + shift)

        return crypted_text

    def __decrypt(self,text,shift):
        original_text = ""
        for idx in range(len(text)):
            char = text[idx]
            original_text += chr(ord(char) - shift)

        return original_text

    def preprocess_doc(self,doc):
        file = open(doc, "r", encoding="utf-8")
        text = file.read()
        file.close()
        chunks = text.split(".")
        return chunks

    def translate(self,languages,text):
        try:
            translated = ""
            for lang in languages:
                translated = GoogleTranslator(target=lang).translate(text)
            translated = GoogleTranslator(target='en').translate(translated)
            return translated
        except:
            return text

    def obfuscate(self, type, n_features=None, file=None, languages=None):

        print("\n### OBFUSCATION PROCESS IN PROGRESS ###\n")

        if type == "encrypt":
            print("\n>>> EXPERIMENT N.1\n")
            features = self.get_main_features("Austen",n_features)
            print("Number of features to encrypt: ",len(features))
            print(">>> processing...\n")
            with open(self.document, "r", encoding="utf-8") as file:
                obf_doc = file.read()
                file.close()

            for feature in features:
                obf_f = self.__encrypt(feature,4)
                if self.features == "words":
                    obf_doc = re.sub(r"\b" + feature + r"\b",obf_f,obf_doc)
                else:
                    obf_doc = re.sub(feature, obf_f, obf_doc)

            with open("encrypted_doc.txt", "w", encoding="utf-8") as file:
                file.write(obf_doc)
                file.close()

        elif type == "translate":
            print("\n>>> EXPERIMENT N.2\n")
            print(">>> Translation pipeline: en -> " + " -> ".join(languages) + " -> en")
            print(">>> Document: " + file)
            print(">>> processing...\n")
            pool = ThreadPool()
            document = self.preprocess_doc(file)
            lang = [languages]*len(document)
            result = pool.starmap(self.translate, zip(lang,document))
            pool.close()
            pool.join()
            ob_text = ".".join(result)
            path = "translated/obf_file(" + "_".join(languages) + ").txt"
            ob_file = open(path, "w", encoding="utf-8")
            ob_file.write(ob_text)
            ob_file.close()
            print(">>> Output file: " + path)

        print("\n### OBFUSCATION PROCESS ENDED ###\n")

    def success(self,probabilities):
        return probabilities[0][0] != self.author




