import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

import matplotlib.pyplot as plt

# Preprocessing of the dataset
words=set(stopwords.words('english'))

df=np.load('data_train.pkl', allow_pickle=True)
# print(labels[16])
valid_size=2000
train_set=[df[0][:-valid_size],df[1][:-valid_size]]
validation_set=[df[0][-valid_size:],df[1][-valid_size:]]
labels=np.unique(df[1])
# Number of labels
n_labels=len(labels)
# Number of exemples
n=len(train_set[0])

vectorizer = TfidfVectorizer(train_set[0], min_df=2, max_df=1400, stop_words=words, strip_accents='ascii')

X_train=vectorizer.fit_transform(train_set[0])
y_train=np.array(train_set[1])
X_val=vectorizer.transform(validation_set[0])
y_val=np.array(validation_set[1])

# Dimensionality reduction using PCA

from sklearn.decomposition import TruncatedSVD
n_features=10000
pca=TruncatedSVD(n_components=n_features)
x_train=pca.fit_transform(X_train)
x_val=pca.transform(X_val)
# Fitting the best model
model = RandomForestClassifier(n_jobs=-3,min_samples_split=10, min_samples_leaf=2,random_state=42, n_estimators=200)
model.fit(x_train,y_train)
print('Accuracy on the validation set: ',model.score(x_val,y_val))
print('Accuracy on the training set: ', model.score(x_train,y_train))

# Uncomment the following lines to see the result with Bagging
# BC=BaggingClassifier(model,n_jobs=-3)
# BC.fit(X_train,y_train)
# print('Accuracy on the validation set: ',BC.score(X_val[:,ind_chi2],y_val))
# print('Accuracy on the training set: ', BC.score(X_train[:,ind_chi2],y_train))


# Prediction of the test examples
test=np.load('data_test.pkl', allow_pickle=True)
X_test=vectorizer.transform(test)
predictions=model.predict(X_test)

# Saving in the right format
submission=pd.DataFrame(enumerate(predictions), columns=['Id','Category'])
submission.to_csv('submissionRF.csv', index=False)