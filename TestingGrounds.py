from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from datacombiner import DataCombiner, Object
import random
import numpy as np


def save_classifier(classifier):
    joblib.dump(classifier, 'classifier.pkl')


def load_classifier(name):
    return joblib.load(name + '.pkl')


def create_classifier(feature_vector, target_vector):
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))),
                         ('tfidf', TfidfTransformer(use_idf=True)),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42,))
                         ])

    text_clf = text_clf.fit(feature_vector, target_vector)
    return text_clf


def test_classifier(classifier, test_documents, data_target_names):
    predicted = classifier.predict(test_documents)

    for doc, j in zip(test_documents, predicted):
        print('%r => %s' % (doc, data_target_names[j]))

def evaluate_classifier(classifier, _data):
    predictions = classifier.predict(_data.data)
    print (np.mean(predictions == _data.target))


def sample_data(population_data, number_samples):
    ax = Object()
    ax.data = []
    ax.target = []
    random_indexes = random.sample(range(len(population_data.data)), number_samples)
    for index in random_indexes:
        ax.data.append(population_data.data[index])
        ax.target.append(population_data.target[index])
    return ax


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
            'good luck', '#rip love you so much', '#congratulations on baby its a boy!',
            '#theysaidido i do todays the big day #ido', 'marriage sucks ill never do it again to hell with bobby','#rip miss you kev']
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }
def grid_search(sample, classifier):
    gs_clf = GridSearchCV(classifier, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(sample.data, sample.target)
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    print (score)



if __name__== '__main__':
    #text_clf = create_classifier(tweets.data, tweets.target)
    text_clf = load_classifier('classifier')
    sample_tweets = sample_data(tweets, 400000)
    #grid_search(sample_tweets,text_clf)
    test_classifier(text_clf, docs_new, tweets.classification_names)
    evaluate_classifier(text_clf,sample_tweets)
    save_classifier(text_clf)


