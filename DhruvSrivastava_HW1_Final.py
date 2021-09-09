#!/usr/bin/env python
# coding: utf-8

# In[1]:

#install contractions and sklearn, if it's not on the system.

import warnings
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import contractions
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from sklearn.model_selection import train_test_split
nltk.download('stopwords')

warnings.simplefilter(action='ignore',category=FutureWarning)
warnings.filterwarnings(action='ignore',category=UserWarning)

# In[2]:


#! pip install bs4 # in case you don't have it installed
# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz


# ## 1. Read Data

# 1. We read the data directly from the link using the "read_csv" function of pandas. "error_bad_lines=False" is used to Drop any row that contains bad data.
# 
# 2. We use "dropna" to drop any row which contains empty/null values.

# In[3]:


#Initial_Dataset = pd.read_csv("https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz",error_bad_lines=False,sep="\t") 
Initial_Dataset = pd.read_csv("https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz",error_bad_lines=False,sep="\t") 


# In[4]:


Initial_Dataset = Initial_Dataset.dropna()


# ## 2. Keep Reviews and Ratings

# We only need two columns, ( 'review_body', 'star_rating' ), and thus we ignore all the other columns and use only these two.

# In[5]:


Initial_Dataset = Initial_Dataset[['review_body','star_rating']]


# ### Three Sample Reviews and Ratings: 

# In[6]:


#print(Initial_Dataset.sample(3))


# ## 3. Statistics of Ratings:

# In[7]:


Initial_Dataset[["star_rating"]].describe()


# #### Reviews grouped by Star Rating received:
# 

# In[8]:


Agg_Data = Initial_Dataset.groupby(["star_rating"]).count()
Agg_Data = Agg_Data.reset_index()
#print(Agg_Data)
# bring count in review_body


# ### Count for each class of reviews:

# In[9]:


Neutral_Reviews  = Initial_Dataset[Initial_Dataset['star_rating']==3]['star_rating'].count()
Positive_Reviews = Initial_Dataset[Initial_Dataset['star_rating']>3]['star_rating'].count()
Negative_Reviews = Initial_Dataset[Initial_Dataset['star_rating']<3]['star_rating'].count()
#print('Neutral Reviews: ', Neutral_Reviews,',','Positive Reviews: ',Positive_Reviews,',','Negavtive Reviews: ',Negative_Reviews,'\n')


# ## 4. Labelling Reviews:
# ### The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'

# We don't need to consider "Neutral Reviews" and we'll be ignoring them in the following step.

# In[10]:


df = Initial_Dataset[Initial_Dataset['star_rating']!=3]


# Now, we'll add the "Label" column. We'll do labelling in the following way:
# 1. if star_rating > 3 = 1
# 2. if star_rating < 3 = 0

# In[11]:


df['label'] = df['star_rating'].apply(lambda x: 1 if x >3 else 0)


#  ### We select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews.
# 
# 

# 1. Using the "sample" function we'll select 100,000 random values where label = 1 and 100,000 random values where label = 0.
# 
# 2. After selecting random values, we'll join them together to make a smaller data set which is ready for Data Cleaning and Pre-Processing.

# In[12]:


positive_ht = df[df.label == 1].sample(100000)


# In[13]:


negative_ht = df[df.label == 0].sample(100000)


# In[14]:


data_set = pd.concat([positive_ht,negative_ht])


# ## Average length of string BEFORE cleaning:

# In[15]:


temp1 = data_set["review_body"].str.len()


# In[16]:


Avg_len_BeforeCleaning = temp1.mean()
#print('Average length of string BEFORE cleaning: ',Avg_len_BeforeCleaning,'\n')


# ## Three sample readings BEFORE data cleaning and pre-processing:

# In[17]:


#print(data_set.sample(3))


# ## 5. Data Cleaning
# 
# ### 5.a) Convert the all reviews into the lower case.

# "str.lower( )" is used to convert all characters to lowercase.

# In[18]:


data_set['review_body'] = data_set['review_body'].str.lower()


# ### 5.b) remove the HTML and URLs from the reviews

# ### Removing HTML

# "BeautifulSoup" library is used to extract text from HTML/XML files. We use it here to retrieve just the relevant text and ignore HTML text present in the 'review_body' column.

# In[19]:


data_set['review_body']  = [BeautifulSoup(X).getText() for X in data_set['review_body']]


# ### Removing URL 

# 1. We used regular expression ( " http\S+|www.\S+ " ) to filter-out any text which falls under the category of a URL.
# 2. We "replaced" every URL with a an empty ""

# In[20]:


data_set['review_body'] = data_set['review_body'].str.replace('http\S+|www.\S+', '', case=False)


# ### 5.c) remove non-alphabetical characters

# The regular expression r'[^a-zA-Z ]+' represents all strings that contain non-alphabetical characters and we replaced them with "".

# In[21]:


data_set['review_body'] = data_set.review_body.str.replace(r'[^a-zA-Z ]+', '')


# ### 5.d) Remove the extra spaces between the words
# 
# 1. Used "strip" to remove the extra-spaces at the beginning and the end of a string.
# 2. Used the regular expression ( '\s+' ) to remove the extra space between the words.

# In[22]:


data_set['review_body'] = data_set['review_body'].str.strip()
data_set['review_body'] = data_set.review_body.str.replace('\s+', ' ', regex=True)


# ### 5.e) perform contractions on the reviews.
# 
# 1. Used the contractions library to expand all contrations like we'll = we will, you'll = You will...

# In[23]:


data_set['review_body'] = data_set['review_body'].apply(lambda x: contractions.fix(x))


# ### Average length of string AFTER cleaning OR Average length of string BEFORE Pre-processing :
# 

# In[24]:


temp2 = data_set["review_body"].str.len()
Average_length_AFTER_cleaning = temp2.mean()
#print(Average_length_AFTER_cleaning)


# ## 6. Pre-processing

# ### 6.a) Removing Stopwords 

# Removed all the stopwords from the 'review_body' column using the "stopwords" list downloaded from NLTK.

# In[25]:


stop_words = set(stopwords.words('english'))
data_set['review_body'] = data_set['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# ### 6.b) perform lemmatization  
# 
# 1. Lemmatization is the step at which we reduce a word to it's base form. Eg. Studying => Study
# 2. Wordnet is an large, freely and publicly available lexical database for the English language.
# 3. We use WordNetLemmatizer() and call the lemmatize() function on each word of the string.
# 4. The "string_word_lemmetize" function lemmitizes each string (row) passed to it by lemmitizing each word of that string.
# 5. We used WhitespaceTokenizer() to extract the tokens from string of words without whitespaces etc.

# In[26]:


import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def string_word_lemmetize(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


# In[27]:


data_set['review_body'] = data_set.review_body.apply(string_word_lemmetize)


# The "string_word_lemmetize" function returns a list and inorder to convert it into a string, we use the join function.

# In[28]:


data_set['review_body'] = data_set['review_body'].str.join(' ')


# ## Three sample readings AFTER data cleaning and pre-processing:

# In[29]:


#print(data_set[['review_body','star_rating','label']].sample(3))


# ## Average length of string AFTER pre-processing: 

# In[30]:


temp3 = data_set["review_body"].str.len()
Average_length_after_preprocessing = temp3.mean()
#print(Average_length_after_preprocessing)


# ### DATASET Split 80-20
# 
# We split the data into training and testing dataset

# In[34]:


review_body_train,review_body_test,label_train,label_test = train_test_split(data_set["review_body"], data_set["label"], test_size=0.2)


# ## 7. TF-IDF Feature Extraction

# 1. TF-IDF stands for Term Frequency – Inverse Document Frequency and we use it for feature extraction.
# 2. The TF-IDF algorithm is implemented using TfidfVectorizer.

# In[35]:


vectorizer = TfidfVectorizer()


# 1. We usefit_transform( ) method on our training data and transform( ) method on our test data
# 2. We usefit_transform( ) on training data so that we can scale the training data and learn the scaling parameters of that data (Mean, Variance)
# 3. The parameters found using the fit_transform( ) will be used by transform( ) when working on the testing set.
# 4. If we apply fit_transform on testing data as well, then our model would calculate new mean and variance, defeating the purpose of the testing dataset

# In[36]:


review_body_train_final = vectorizer.fit_transform(review_body_train)
review_body_test_final = vectorizer.transform(review_body_test)


# Now we have the datset ready which includes features and labels for the reviews. We have to now just use the data and train different models and test their accuracy. We use 4 different models:
# 
# 1. Perceptron (imported from sklearn.linear_model)
# 2. SVM (imported from sklearn.svm)
# 3. Logistic Regression (imported from sklearn.linear_model)
# 4. Multinomial Naive Bayes (imported from sklearn.naive_bayes)

# In[37]:


print("\n---------------FINAL OUTPUTS---------------\n")

print('Neutral Reviews:',Neutral_Reviews,',','Positive Reviews:',Positive_Reviews,',','Negavtive Reviews:',Negative_Reviews,'\n')
print(Avg_len_BeforeCleaning, Average_length_AFTER_cleaning)
print(Average_length_AFTER_cleaning, Average_length_after_preprocessing)


# ## 7.a) Perceptron

# In[38]:


ppn = Perceptron()
ppn.fit(review_body_train_final, label_train)
pred_ = ppn.predict(review_body_test_final)
pred2 = ppn.predict(review_body_train_final)


# In[39]:


#print('\nPerceptron:')
#print('Testing Data: ',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_test, pred_),precision_score(label_test, pred_), recall_score(label_test, pred_), f1_score(label_test, pred_))) 
#print('Training Data:',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_train, pred2),precision_score(label_train, pred2), recall_score(label_train, pred2), f1_score(label_train, pred2))) 
print(accuracy_score(label_train, pred2),precision_score(label_train, pred2), recall_score(label_train, pred2), f1_score(label_train, pred2), accuracy_score(label_test, pred_),precision_score(label_test, pred_), recall_score(label_test, pred_), f1_score(label_test, pred_),sep=', ')

# ## 7.b) SVM

# In[40]:


lsvc = LinearSVC()
lsvc.fit(review_body_train_final, label_train)
pred_SVM = lsvc.predict(review_body_test_final)
pred2SVM = lsvc.predict(review_body_train_final)


# In[41]:


#print('\nSVM:')
#print('Testing Data: ',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_test, pred_SVM),precision_score(label_test, pred_SVM), recall_score(label_test, pred_SVM), f1_score(label_test, pred_SVM))) 
#print('Training Data:',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_train, pred2SVM),precision_score(label_train, pred2SVM), recall_score(label_train, pred2SVM), f1_score(label_train, pred2SVM))) 
print(accuracy_score(label_train, pred2SVM),precision_score(label_train, pred2SVM), recall_score(label_train, pred2SVM), f1_score(label_train, pred2SVM),accuracy_score(label_test, pred_SVM),precision_score(label_test, pred_SVM), recall_score(label_test, pred_SVM), f1_score(label_test, pred_SVM), sep=', ')

# ## 7.c) Logistic Regression

# In[42]:


logisticRegr = LogisticRegression(max_iter=1000)
logisticRegr.fit(review_body_train_final, label_train)
pred_LR = logisticRegr.predict(review_body_test_final)
pred2_LR = logisticRegr.predict(review_body_train_final)


# In[43]:


#print('\nLogistic Regression:')
#print('Testing Data: ',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_test, pred_LR),precision_score(label_test, pred_LR), recall_score(label_test, pred_LR), f1_score(label_test, pred_LR))) 
#print('Training Data:',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_train, pred2_LR),precision_score(label_train, pred2_LR), recall_score(label_train, pred2_LR), f1_score(label_train, pred2_LR))) 
print(accuracy_score(label_train, pred2_LR),precision_score(label_train, pred2_LR), recall_score(label_train, pred2_LR), f1_score(label_train, pred2_LR),accuracy_score(label_test, pred_LR),precision_score(label_test, pred_LR), recall_score(label_test, pred_LR), f1_score(label_test, pred_LR),sep=', ')

# ## 7.d) Multinomial Naive Bayes

# In[44]:


nb = MultinomialNB()
nb.fit(review_body_train_final, label_train)
pred_NB = nb.predict(review_body_test_final)
pred2_NB = nb.predict(review_body_train_final)


# In[45]:


#print('\nNaive Bayes:')
#print('Testing Data: ',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_test, pred_NB),precision_score(label_test, pred_NB), recall_score(label_test, pred_NB), f1_score(label_test, pred_NB))) 
#print('Training Data:',"%.2f, %.2f, %.2f, %.2f" % (accuracy_score(label_train, pred2_NB),precision_score(label_train, pred2_NB), recall_score(label_train, pred2_NB), f1_score(label_train, pred2_NB))) 
print(accuracy_score(label_train, pred2_NB),precision_score(label_train, pred2_NB), recall_score(label_train, pred2_NB), f1_score(label_train, pred2_NB),accuracy_score(label_test, pred_NB),precision_score(label_test, pred_NB), recall_score(label_test, pred_NB), f1_score(label_test, pred_NB),sep=', ')
