
# coding: utf-8

# ## 3.6 Featurizing text data with TF-IDF weighted word-vectors and Avg.word-vectors

# In[1]:


# We will only operate on the first 100k points as 8GB RAM is not enough.

# Library imports:
import numpy as np
import pandas as pd
from time import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# We can extract word2vec vectors using spacy
# If there are any dependency issues, please folow these links:
# https://github.com/explosion/spaCy/issues/1721
# http://landinghub.visualstudio.com/visual-cpp-build-tools
import spacy


# In[2]:


# We will use the following function to time our code:
def time_taken(start_time):
    print("~> Time taken:",
         round(time()-start_time, 2), "seconds")
    return


# In[ ]:


# We will import more libraries as and when required.


# In[5]:


st = time()

# Import sample from the original dataset:
df = pd.read_csv("../train/train.csv", nrows=100000)

# Encode all the questions to unicode format:
df['question1'] = df['question1'].apply(lambda x: str(x))
df['question2'] = df['question2'].apply(lambda x: str(x))

time_taken(st)
df.shape


# In[6]:


df.head(2)


# ### 3.6.1 Computing TF-IDF weighted Average Word2Vec Vectors

# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Merge texts into a single list:
questions = list(df['question1']) + list(df['question2'])

# Create TfidfVectorizer instance:
tfidf = TfidfVectorizer(lowercase=False)

# Get the parameters in tfidf instance:
print(tfidf)


# In[8]:


st =time()

# Now apply tfidf transform:
tfidf.fit_transform(questions)

time_taken(st)


# In[9]:


st = time()

# We will now take all the tf-idf vectored values into a dictionary:
# key:word and value:tf-idf score
word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

time_taken(st)


# - We have the TF-IDF Scores. We will now convert each question to a weighted average of word2vec vectors by using these TF-IDF Scores.
# - We use a pre-trained GLoVe model which comes free with spacy library. The model itself is trained on Wikipedia data. More about the pre-trained model @ https://spacy.io/usage/vectors-similarity
# - Because it is trained on Wikipedia, the model is strong in terms of word semantics

# In[ ]:


# We can either load and use 'en_vectors_web_lg' or 'en_core_web_sm'.
# The difference between the two is that, 'en_core_web_sm' is a smaller
# model compared to 'en_vectors_web_lg'

# If the system has less RAM, it is better to load 'en_core_web_sm' model
# as the word-vector. The only disadvantage with a smaller model is that
# it will give us similarity vectors with lesser accuracy.

# Note: there are 2 full version models of word2vec models trained on
# wikipedia data. One of them is 'en_core_web_lg' and the other one is
# 'en_vectors_web_lg'. If we want to make changes to the word2vec model,
# then we can load the 'en_vectors_web_lg' word2vec model, otherwise, we
# go with 'en_core_web_lg' word2vec model.

# Note: We can't load these pre-trained word2vec models unless and until
# we download the models through the console of the system.

    ###############################################
    # Syntax: python -m spacy download model_name #
    ###############################################

# Example: python -m spacy download en_core_web_lg


# In[10]:


# Loading the smaller model:
nlp = spacy.load('en_core_web_sm')
st = time()

vector1 = []
# https://github.com/noamraph/tqdm
# tqdm is used to print the progress bar.
for q1 in tqdm(list(df.question1)):
    doc = nlp(q1)
    # 384 dimensional vector
    mean_vec1 = np.zeros([len(doc), 384])
    for word in doc:
        # word2vec:
        vec1 = word.vector

        # Fetch the tf-idf score:
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 0

        # Compute final vector:
        mean_vec1 += vec1 * idf
    mean_vec1 = mean_vec1.mean(axis=0)
    vector1.append(mean_vec1)
df['q1_feats_m'] = list(vector1)


# In[11]:


time_taken(st)


# In[12]:


st = time()

vector2 = []
for q2 in tqdm(list(df.question2)):
    doc = nlp(q2)
    mean_vec2 = np.zeros([len(doc), 384])
    for word in doc:
        # word2vec:
        vec2 = word.vector
        # Fetch idf score:
        try:
            idf = word2tfidf[str(word)]
        except:
            idf = 0
        # Compute final vector:
        mean_vec2 += vec2 * idf
    mean_vec2 = mean_vec2.mean(axis=0)
    vector2.append(mean_vec2)

df['q2_feats_m'] = list(vector2)

time_taken(st)


# In[13]:


# Get the files in the current working directory
files_in_cwd = os.listdir()
index = 0
print("<idx>. <Filename>")
for f in files_in_cwd:
    print("{}. {}".format(index, f))
    index += 1


# In[14]:


# We will read some previously saved .csv files like:
# 1. nlp_features_train.csv
# 2. df_fe_without_preprocessing_train.csv
# and we will use the data generated now and finally merge all of the data into
# a single pandas dataframe. The dataframe size may get really large.

st = time()
# Load the nlp_features_train.csv file into a dataframe:
if os.path.isfile('nlp_features_train.csv'):
    df_nlp = pd.read_csv("nlp_features_train.csv", encoding='latin-1', nrows=100000)
else:
    print('Generate the file by running the code in QQP1.')

# Load the df_fe_without_preprocessing_train.csv file into a dataframe:
if os.path.isfile('df_fe_without_preprocessing_train.csv'):
    df_pre = pd.read_csv('df_fe_without_preprocessing_train.csv', encoding='latin-1', nrows=100000)
else:
    print('Generate the file by running the code in QQP1.')

time_taken(st)


# In[15]:


print(df_nlp.shape)
print(df_pre.shape)
# We will drop the unnecessary features and only keep the required ones:
df1 = df_nlp.drop(['qid1', 'qid2', 'question1', 'question2'], axis=1)

# df1 corresponds to advanced nlp and fuzzy engineered features:
df1.head()


# In[16]:


df2 = df_pre.drop(['qid1', 'qid2', 'question1', 'question2','is_duplicate'],axis=1)

# df2 corresponds to basic engineered features:
df2.head()


# In[17]:


# our original dataset with some additional features:
df.head()


# In[18]:


# We will drop ['qid1','qid2','question1','question2','is_duplicate'] from df:
df3 = df.drop(['qid1','qid2','question1','question2','is_duplicate'], axis=1)
df3.head()


# In[20]:


# q1_feats_m has each row as a list. Therefore, we will extract it into a
# dataframe as:
st = time()
df_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index = df3.index)
time_taken(st)
df_q1.head()


# In[21]:


# q12_feats_m has each row as a list. Therefore, we will extract it into a
# dataframe as:
st = time()
df_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index = df3.index)
time_taken(st)
df_q2.head()


# In[23]:


print("Number of features in nlp dataframe:", df1.shape[1])
print("Number of features in preprocessed dataframe:", df2.shape[1])
print("Number of features in question1 w2v dataframe:", df_q1.shape[1])
print("Number of features in question2 w2v dataframe:", df_q2.shape[1])
print("Number of features in the final dataframe:", df1.shape[1] + df2.shape[1] + df_q1.shape[1] + df_q2.shape[1])


# In[25]:


st = time()

# The following code might take some time to execute, depending on the system
# configuration.
if not os.path.isfile('final_features_100k.csv'):

    # Attach 'id' attribute to question1 and question2 w2v vectors:
    df_q1['id'] = df1['id']
    df_q2['id'] = df1['id']

    # Merge nlp_features with preprocessing_features:
    df1 = df1.merge(df2, on='id', how='left')

    # Merge question1 and question2 w2v vectors and save them in df2 variable:
    df2 = df_q1.merge(df_q2, on='id', how='left')

    # We will now merge df1 and df2 into result:
    result = df1.merge(df2, on='id', how='left')

    # Save as a .csv file to use when applying k-NN to classify the points:
    result.to_csv('final_features_100k.csv')

time_taken(st)
