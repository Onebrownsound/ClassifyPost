from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import numpy as np
from datacombiner import DataCombiner


def save_classifier(classifier):
    joblib.dump(classifier, 'classifier.pkl')


def load_classifier(name):
    return joblib.load(name + '.pkl')


def create_classifier(feature_vector, target_vector):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),
                         ])

    text_clf = text_clf.fit(feature_vector, target_vector)
    return text_clf


def test_classifier(classifier, test_documents, data_target_names):
    predicted = classifier.predict(test_documents)

    for doc, j in zip(test_documents, predicted):
        print('%r => %s' % (doc, data_target_names[j]))


tweets = DataCombiner(filenames=['marriage_tweets', 'death_tweets', 'baby_tweets', 'garbage_tweets'],
                      classifications=['Marriage', 'Death', 'Baby', 'Garbage'])

docs_new = ['i will miss you bob RIP ', 'congrats on your marriage becky',
            'you were a very kind friend ill miss you dearly', 'ryan you were an animal grats on the wedding day',
            'so sorry for your loss ryan my condolences rip', 'this is a random post makes no sense',
            'i wish the vikings beat the chargers tonight', 'God damn I hate donald trump',
            'We need to do more for the environment', 'congrats on the marriage Bobby',
            'congratulations Sally on your wedding, i hope it lasts forever',
            "i don't wanna go to school it sucks in da morning", 'congratulations',
            'so excited i heard about your new kid I welcome him into the world',
            'im going to miss that dude he was my best friend',
            'grats to eileen and mark i hope they cherish each other on their special day',
            'good luck']

#text_clf = create_classifier(tweets.data, tweets.target)
text_clf = load_classifier('classifier')
test_classifier(text_clf, docs_new, tweets.classification_names)
save_classifier(text_clf)
