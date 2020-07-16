#!/usr/bin/python

# Preprocesses raw full texts in the following steps:
#   (1) Case-folding and punctuation removal
#   (2) Lemmatization with WordNet

import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def lemmatize_term(term):

	# Load WordNet lemmatizer from NLTK
	lemmatizer = WordNetLemmatizer()

	acronyms = ["abc", "aai", "adhd", "aids", "atq", "asam", "asi", "aqc", "asi", "asq", "ax", "axcpt", "axdpx", "bees", "bas", "bdm", "bis", "bisbas", "beq", "brief", "cai", "catbat", "cfq", "deq", "dlmo", "dospert", "dsm", "dsmiv", "dsm5", "ecr", "edi", "eeg", "eei", "ema", "eq", "fmri", "fne", "fss", "grapes", "hrv", "iri", "isi", "ius", "jnd", "leas", "leds", "locs", "poms", "meq", "mctq", "sans", "ippa", "pdd", "pebl", "pbi", "prp", "mspss", "nart", "nartr", "nih", "npu", "nrem", "pas", "panss", "qdf", "rbd", "rem", "rfq", "sam", "saps", "soc", "srs", "srm", "strain", "suds", "teps", "tas", "tesi", "tms", "ug", "upps", "uppsp", "vas", "wais", "wisc", "wiscr", "wrat", "wrat4", "ybocs", "ylsi"]
	names = ["american", "badre", "barratt", "battelle", "bartholomew", "becker", "berkeley", "conners", "corsi", "degroot", "dickman", "marschak", "beckerdegrootmarschak", "beery", "buktenica", "beerybuktenica", "benton", "bickel", "birkbeck", "birmingham", "braille", "brixton", "california", "cambridge", "cattell", "cattells", "chapman", "chapmans", "circadian", "duckworth", "duckworths", "eckblad", "edinburgh", "erickson", "eriksen", "eysenck", "fagerstrom", "fitts", "gioa", "glasgow", "golgi", "gray oral", "halstead", "reitan", "halsteadreitan", "hamilton", "hayling", "holt", "hooper", "hopkins", "horne", "ostberg", "horneostberg", "iowa", "ishihara", "kanizsa", "kaufman", "koechlin", "laury", "leiter", "lennox", "gastaut", "lennoxgastaut", "london", "macarthur", "maudsley", "mcgurk", "minnesota", "montreal", "morris", "mullen", "muller", "lyer", "mullerlyer", "munich", "parkinson", "pavlovian", "peabody", "penn", "penns", "piaget", "piagets", "pittsburgh", "porteus", "posner", "rey", "ostereith", "reyostereith", "reynell", "rivermead", "rutledge", "salthouse", "babcock", "spielberger", "spielbergers", "stanford", "binet", "shaver", "simon", "stanfordbinet", "sternberg", "stroop", "toronto", "trier", "yale", "brown", "umami", "uznadze", "vandenberg", "kuse", "vernier", "vineland", "warrington", "warringtons", "wason", "wechsler", "wisconsin", "yalebrown", "zimbardo", "zuckerman"]

	if term not in acronyms + names:
		return lemmatizer.lemmatize(term)
	else:
		return term


# Function for stemming, conversion to lowercase, and removal of punctuation
def preprocess_text(text):

	# Load English stop words from NLTK
	stops = stopwords.words("english")

	# Convert to lowercase, convert slashes to spaces, and remove remaining punctuation except periods
	text = text.replace("-\n", "").replace("\n", " ").replace("\t", " ")
	text = "".join([char for char in text.lower() if char.isalpha() or char.isdigit() or char in [" ", "."]])
	text = text.replace(".", " . ").replace("  ", " ").strip()
	text = re.sub("\. \.+", ".", text)

	# Perform lemmatization, excluding acronyms and names in RDoC matrix
	text = " ".join([lemmatize_term(term) for term in text.split() if term not in stops])

	return text


# Function for consolidating ngrams
def consolidate_ngrams(text, ngrams):

	consolidated_ngrams = [ngram.replace(" ", "_") for ngram in ngrams]
	for i, ngram in enumerate(ngrams):
            text = text.replace(ngram, consolidated_ngrams[i])
	text = re.sub("\. \.+", ".", text)

	return text


# Wrapper function for executing preprocessing steps on a corpus
def run_preproc(path, pmids, ngrams=[], preproc_texts=True, preproc_ngrams=True):
    for pmid in pmids:
        file = "{}/{}.txt".format(path, pmid)
        text = open(file, "r").read()
        if preproc_texts:
            text = preprocess_text(text)
        if preproc_ngrams:
            text = consolidate_ngrams(text, ngrams)
        with open(file, "w+") as outfile:
            outfile.write(text)


# Function to fit a document-term matrix for a corpus of records
def fit_dtm(records, pmids, lexicon, suffix):
    
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    
    vec = CountVectorizer(min_df=0, vocabulary=lexicon)
    dtm = vec.fit_transform(records)
    dtm_df = pd.DataFrame(dtm.toarray(), index=pmids, columns=lexicon)
    dtm_df.to_csv("dtm_{}.csv.gz".format(suffix), compression="gzip")
    
    return dtm_df

    