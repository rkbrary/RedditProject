import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
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

# Tuning the smoothing parameter
alpha=np.linspace(0.01,1.,20)
scores=np.zeros(len(alpha))
for i in range(len(alpha)):
    model=MultinomialNB(alpha=alpha[i])
    model.fit(X_train,y_train)
    scores[i]=model.score(X_val,y_val)

# Plotting the validation curve
plt.plot(alpha,scores)
plt.xlabel('alpha')
plt.ylabel('Score')
plt.show()

alpha0=alpha[np.argmax(scores)]

# Fitting the best model
model = MultinomialNB(alpha=alpha0)
model.fit(X_train,y_train)
print('Accuracy on the validation set: ',model.score(X_val,y_val))

# Prediction of the test examples
test=np.load('data_test.pkl', allow_pickle=True)
X_test=vectorizer.transform(test)
predictions=model.predict(X_test)

# Saving in the right format
submission=pd.DataFrame(enumerate(predictions), columns=['Id','Category'])
submission.to_csv('submissionNB.csv', index=False)