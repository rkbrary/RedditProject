import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
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

# Feature Selection using chi2 (was best not to use it after all)

ki2=chi2(X_train,y_train)[1]
ind_chi2=ki2<2 # By using a threshold bigger than 1, it is the same as not using chi2

# ind_chi2=np.ones(X_train.shape[1],dtype=bool)

# Tuning the regularization parameter

# Fitting the best model
model = LogisticRegression(penalty='elasticnet', solver='saga',multi_class='multinomial', l1_ratio=0.5)
model.fit(X_train[:,ind_chi2],y_train)
print('Accuracy on the validation set: ',model.score(X_val[:,ind_chi2],y_val))

# # Prediction of the test examples
test=np.load('data_test.pkl', allow_pickle=True)
X_test=vectorizer.transform(test)
predictions=model.predict(X_test[:,ind_chi2])

# Saving in the right format
submission=pd.DataFrame(enumerate(predictions), columns=['Id','Category'])
submission.to_csv('submissionLogReg.csv', index=False)