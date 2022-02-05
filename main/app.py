'''Vector Representations'''

'''As we learned in module 1, machines
are unable to understand raw text, hence
we need to turn it into something they can
analyze. In this project, we'll discuss
vector representations like bag of words
and word embedding models. We'll use those
representations for search, visualization,
and prepare for classification day in the
next project.'''
'''Processing data for ML models usually
means translating the information from docs
into a numerical format. BoW approaches accomplish
this by vectorizing tokenized docs. Vectorization
represents each doc as a row in a dataframe
and creates a column for each unique word
in the corpora (group of docs). The presence
or lack thereof of a given word in a doc is
either a raw count or how many times it appears
or as that word's TF-IDF score.'''

### OBJECTIVE 01- REPRESENT A DOCUMENT
### AS A VECTOR

'''Overview'''

'''In the previous project, we focused
on one of the first steps of nlp, tokenizing.
This step usually includes processing such as
getting rid of stop words and finding stems
and lemmas. When we have our text broken down into
tokens, we can do a more quantitative
analysis such as word counts and ranking the
text by most common words.
But there is only so much we can do with
tokenized text. ML algorithms don't accept
text or tokens as input, so we need addtional
ways to represent them. In this project, we'll
focus on vectorizing text. There are a few
ways to do this, and we'll go over the
advantages and disadvantages of each.'''

'''Text Vectorization'''

'''In linear algebra, a vector is a single
colmn or row that contains a number of
elements. When we want to represent text
as a vector, we are trying to illustrate the
importance of the words in the document
numerically or in a meaningful way.
A simple example is a binary bit vector,
where a 1 indicates the presence of the word.
Obviously there's no way to represent how
often a word appears in a given doc.'''

'''Bag of Words'''

'''Another common way of representing
data is using a BoW model. Like in
text vectorization, you ignore grammar
and order, but this time multiplicity is
recorded. When we represent a text or doc
as a bow, we can use the term frequency to
create a vector for the text. Next we
count each word or term in the text and
construct  vector for each term.'''

'''Document-term Matrix'''

'''Now that we have looked at two
ways of creating vectors from text,
we can generalize the concept of
a doc-term matrix, DTM. We can represent
the numeric characteristics of a doc by
using a matrix that describes the frequency
of terms that appear in a collection of docs.
The rows correspond to docs in the collection
in a DTM, and the columns correspond to the terms.
In the examples we have used so far, the sentences
would be the documents, and the words in those sentences
are the terms. In a BOW model, the value in the
cell is the term frequency. In this module, we'll
use the term frequency-inverse document
frequency (tf-idf).
The tf-idf is calculated by counting how
many times the term occurs in the document
(term frequency) divided by the number of documents
in which that word occurs.'''

'''Follow Along'''
'''Using an example corpus (collection
of docs), we'll implement some of the
above examples in scikit-learn. First,
we'll create a sample corpus with some
randomly generated sentences and then look
at vectorization using binary encoding
(one-hot), term frequency, and tf-idf'''

def freq():
    from sklearn.feature_extraction.text import CountVectorizer
    # random corpus
    corpus = ["Karma, karma, karma, karma, karma chameleon.",
          "The paintbrush was angry at the color the chameleon chose to use.",
          "She stomped on her fruit loops and thus became a cereal killer.",
          "He hated that he loved what she hated about cereal and her chameleon."
         ]
    # instantiate object and count words
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    # print the matrix
    print(vectors.todense())
    return "Function complete."
'''Each vector is the length of the words
in the corpus with an integer count for how
often each word appears in the document. For
the first document (sentence), we only have two
words, and so there are only two non-zero
integers. The word karma appears five times
and so for that column, there is a five for the
first document and a zero for all the other
documents.'''

'''One-hot Encoding'''

'''To binary encode our text, we first
have to count the words to put those
values into the binarizer. The binarizer,
by default, will convert all values above a
zero to 1 and all other values to 0.'''
def ohe():
    '''This function fits the count vectorizer
    to the corpus, then fits it to the binarizer
    and turns it into an array.'''
    # import the binary encoder
    from sklearn.preprocessing import Binarizer
    from sklearn.feature_extraction.text import CountVectorizer
    # initialize vectorizer and get word count
    freq = CountVectorizer()
    # random corpus
    corpus = ["Karma, karma, karma, karma, karma chameleon.",
          "The paintbrush was angry at the color the chameleon chose to use.",
          "She stomped on her fruit loops and thus became a cereal killer.",
          "He hated that he loved what she hated about cereal and her chameleon."
         ]
    corpus_freq = freq.fit_transform(corpus)
    # initialize the binarizer and create the binary encoded vector
    onehot = Binarizer()
    corpus_onehot = onehot.fit_transform(corpus_freq.toarray())
    # display the one-hot encoded vector
    return print(corpus_onehot)
'''The one-hot encoded vector for each doc
in the corpus now contains a one if that word is
present in the doc and a 0 if otherwise. There is
no longer any info about how many times the words appear
in each doc, just if they are present.'''

'''Term frequency-inverse document frequency'''

'''In the scikit-learn feature extraction
module, there is a tf-idf vectorizer function.
Because stop words occur so frequently,
we'll remove them before calculating the
tf-idf terms.'''
def tf():
    # import libraries and modules
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    # instantiate vectorizer object
    tfidf = TfidfVectorizer(stop_words="english",
    max_features=5000)
    # random corpus
    corpus = ["Karma, karma, karma, karma, karma chameleon.",
          "The paintbrush was angry at the color the chameleon chose to use.",
          "She stomped on her fruit loops and thus became a cereal killer.",
          "He hated that he loved what she hated about cereal and her chameleon."
         ]
    # create a vocabulary and get word counts per document
    dtm = tfidf.fit_transform(corpus)
    dtm = pd.DataFrame(dtm.todense(),
        columns = tfidf.get_feature_names())
    # view feature matrix as a dataframe
    return dtm.head()

### OBJECTIVE 02- QUERY DOCUMENTS
### BY SIMILARITY

'''Overview'''
'''In the first objective in this
module, we learned how to make vectors
from text. But now, what do we do with
the vectors? It turns out we can do much
more valuable things in natural language
processing and beyond when we have our
info in numeric form. Of course, vectors
are just vectors, and so the usual
mathematical operations apply. So, we'l
first review a bit of linear algebra
and then look at some examples with
actual text vectors.'''

'''Vector Spaces'''
'''A vector space is the number of
dimensions in the space. For example,
text vectors have dimensions equal
to the number of distinct words in the
corpus. For a small corpus, there isn't
a very high dimensional space. However,
when you look at longer documents and larger
corpora, the dimensions become much
larger. But it's difficult for humans
to interpret areas bigger than three
dimensions. So, to make these examples
easier, we'll consider two dimensional
spaces because they are easy to show
on the screen.
Remember that the math is the same
for two-dimensional spaces to the hundred
dimensional spaces we'll work with in
projects in this unit.'''

'''Cosine Similarity'''
'''We often want to compare docs to
each other. We start by converting a
doc to a vector, then compare the vectors.
The next question is when are two vectors
similar? Using linear algebra concepts,
we could say they are similar if they have
the same length and direction. Remember that
length is the number of unique words in
the corpus, which isn't necessarily helpful
when comparing docs. Next, we need to compare
the properties of the vectors. Comparing is
where the concept of cosine similarity
is useful.
If we want to know the similarity of
two vectors, we can calculate the cosine
similarity- the more similar the vectors,
the smaller the angle between them. The
cosine similarity is given by:
cos0 = (A * B) / |A||B|
this can also be considered the normalized
dot product. The normalized dot product
is where the dot product of the vectors
A and B are divided by their lengths. So
it's a measure of how much the vectors
point in the same direction.'''

'''Follow Along'''
'''Now that we have an understanding
of how to mathematically calculate
the cosine similarity of two vectors,
let's do this with some actual text.
We'll create a corpus and then calculate
the tf-idf vectors. Finally, you can
calculate the cosine similarity with the
scikit-learn cosine_similarity function
from the sklearn.metrics.pairwise module.'''

# create the corpus (text is available in a github repo)
def cos():
    # import module, open and read file
    from urllib.request import urlopen
    # three docs on three different subjects
    link = 'https://raw.githubusercontent.com/nwhoffman/NLP_example_text/master/u4s1m2_similarity.txt'
    f = urlopen(link)
    myfile = f.read()
    mystring = str(myfile, "utf-8")
    # create the corpus
    corpus = mystring.split(";")
    # print first 300 characters for each doc
    for i in [0, 1, 2]:
        print("document:", i)
        print(corpus[i][0:300])
    # create the vectors for each doc
    from sklearn.feature_extraction.text import TfidfVectorizer
    # instantiate vectorizer object
    tfidf = TfidfVectorizer(stop_words="english",
        max_features=5000)
    # create a vocabulary and get tfidf values per doc
    dtm = tfidf.fit_transform(corpus)
    import pandas as pd
    # get feature names to use as DF column headers
    dtm = pd.DataFrame(dtm.todense(), columns = tfidf.get_feature_names())
    # view the feature matrix as a dataframe
    print(dtm.head())
    # find cosine similarity of tf-idf vectors
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim = cosine_similarity(dtm)
    # turn it into a dataframe
    cosine_sim = pd.DataFrame(cosine_sim)
    print(cosine_sim)
    return "Function complete."

'''With the above matrix we can see
that each row is the similarity of the doc
to itself (1.00) and the two
other documents. The three docs were excerpts
from textbooks, 0 on astronomy, 1 on biology,
and 2 on physics. The astronomy and physics
documents are more similar than the bio doc,
which makes sense given the overlap in
the two topics.'''
