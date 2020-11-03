from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

df = pd.read_csv("", sep="\t")

vectorizer = CountVectorizer()
corpus = [
    'Text of first document.',
    'Text of the second document make longer.',
    'Number three.',
    'This is number four.',
]

# store CountVectorizer sparse matrix in X
# The column of matrix is words, rows are documents
X = vectorizer.fit_transform(corpus) #(4, 13)
print(vectorizer.get_feature_names())
print(X.toarray())

# Convert a new document to count representation
res = vectorizer.transform(['This is a new document'])
print("===")
print(res)
