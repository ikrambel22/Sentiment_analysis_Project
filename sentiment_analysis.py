import warnings
warnings.filterwarnings('ignore')
#importing packages that we need
import numpy as np 
import pandas as pd 
# Data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#pre-processing libraries
from textblob import Word 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
import neattext.functions as nfx
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
lemmatizer = WordNetLemmatizer()



pd.options.mode.chained_assignment = None  
df = pd.read_csv('Tweets.csv',index_col=None)
df=df[['airline_sentiment','text']]
df.head(10)

#different sentiments
sentiments = set(df['airline_sentiment'])

#pre-processing
#we noticed that our data contains insignificant words
import enchant
def check_meaning_words(data,column):
                    d = enchant.Dict("en_US")
                    i=0
                    for sentence in data[column]:
                                l=[]
                                for token in sentence:
                                     if d.check(str(token)):
                                            l+=[token]
                                data[column][i]=l
                                i=i+1 
                    return data[column]    


#convert links to string
import re
def convert_links_to_text(text):
         new = re.sub(r'^https?:\/\/.*[\r\n]*', 'link', text, flags=re.MULTILINE)
         return new
     
        
#convert emails to string
def convert_emails_to_text(text):
         new1 = re.sub(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+', 'email', text, flags=re.MULTILINE)
         return new1

 
import emoji
def convert_emojis_to_text(text):
          new=emoji.demojize(text, delimiters=("", "")) 
          return new
      

def remove_redundant_chars(data,column):
            i=0
            for sentence in data[column]:
                l=[]
                for token in sentence:
                    if 'aa' in token:
                            repeat_pattern = re.compile(r'(\w)\1*')
                            y=repeat_pattern.sub(r'\1',token)
                            #y=re.sub(r"(.)\1\1+", r"\1\1", token)
                    else:
                        y=token
                    l+=[y]
                data[column][i]=l
                i=i+1
            return data[column]    
       
        
import contractions
def fix_contractions(data,column):
        i=0
        for sentence in data[column]:
            expanded_words=[]
            for word in sentence:
                        expanded_words+=[contractions.fix(word)]
            data[column][i]=expanded_words
            i=i+1
        return data[column]
    
    
    


#cleaning fct
def clean_data(data,column):
    #convert text to lower
    data[column]=data[column].str.lower()
    #remove numbers
    data[column]=data[column].apply(nfx.remove_numbers)
    #remove userhandles
    data[column]=data[column].apply(nfx.remove_userhandles)
    #remove punctuations
    data[column]=data[column].apply(nfx.remove_punctuations)
    #remove special characters
    data[column]=data[column].apply(nfx.remove_special_characters)
    #remove hashtags
    data[column]=data[column].apply(nfx.remove_hashtags)
    #remove some words 
    data[column].replace("amp", " ", regex=True, inplace=True)
    #remove multiple space
    data[column]=data[column].apply(nfx.remove_multiple_spaces)
    #lemmatization
    data[column]=data[column].apply(lambda x: " ".join([lemmatizer.lemmatize(word,wordnet.VERB) for word in x.split()]))
    data[column]=data.apply(lambda row: convert_emojis_to_text(row[column]), axis=1)
    data[column]=data.apply(lambda row: convert_links_to_text(row[column]), axis=1)
    data[column]=data.apply(lambda row: convert_emails_to_text(row[column]), axis=1)
    data['tokenized_sents'] = data.apply(lambda row: nltk.word_tokenize(row[column]), axis=1)
    data['tokenized_sents']=check_meaning_words(data,'tokenized_sents')
    data['tokenized_sents']=remove_redundant_chars(data,'tokenized_sents')
    data['tokenized_sents']=fix_contractions(data,'tokenized_sents')
    data['detokenized_sents'] = data.apply(lambda row: TreebankWordDetokenizer().detokenize(row['tokenized_sents']), axis=1)
    data=data[data['detokenized_sents'].str.len()>=4]
    #delete empty rows
    data = data[data['detokenized_sents']!= '']
    #reset data index
    data=data.reset_index().drop('index',axis=1)
    return data


data=clean_data(df,'text')

#split data
from sklearn.model_selection import train_test_split
X=data['detokenized_sents']
y=data['airline_sentiment']
#we used  80% for training data and 20% for testing
train,valid=train_test_split(data[['detokenized_sents','airline_sentiment']],test_size=0.2, random_state=42)

#Easy Data Augmentation

#1- Synonyme Replacement SR
#We define the get_synonyms function to retrieve pre-processed list of synonyms of a given word
from nltk.corpus import wordnet
def get_synonyms(word):
    
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym) 
    if word in synonyms:
        synonyms.remove(word)
    
    return list(synonyms)

from nltk.corpus import stopwords
stop_words = []
for w in stopwords.words('english'):
    stop_words.append(w)
    
import random
def synonym_replacement(words, n):
    
    words = words.split()
    
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n: #only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

aug_syn = {'airline_sentiment':[],'detokenized_sents':[]}
columns = ['airline_sentiment','detokenized_sents']
for i in train.index:
    if train['airline_sentiment'][i]== 'positive' or train['airline_sentiment'][i]== 'neutral':
        new_row=synonym_replacement(train['detokenized_sents'][i],2)
        aug_syn['airline_sentiment'].append(train['airline_sentiment'][i])
        aug_syn['detokenized_sents'].append(new_row)
        
data1 = pd.DataFrame(aug_syn, columns=columns, index=None)
df = train.append(data1,ignore_index=True)

# 2- Random Deletion (RD)

def random_deletion(words, p):
    words = words.split()
    #obviously, if there's only one word, don't delete it
    if len(words) == 1:
        return words

    #randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    #if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    sentence = ' '.join(new_words)
    
    return sentence

aug_syn = {'airline_sentiment':[],'detokenized_sents':[]}
columns = ['airline_sentiment','detokenized_sents']
for i in df.index:
    if df['airline_sentiment'][i]== 'positive' or df['airline_sentiment'][i]== 'neutral':
        new_row=random_deletion(df['detokenized_sents'][i],0.2)
        aug_syn['airline_sentiment'].append(df['airline_sentiment'][i])
        aug_syn['detokenized_sents'].append(new_row)
        
data2 = pd.DataFrame(aug_syn, columns=columns, index=None)
df1 = df.append(data2,ignore_index=True)

#3-Random Swap (RS)
# This will Swap the words
def swap_word(new_words):
    
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        
        if counter > 3:
            return new_words
    
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

def random_swap(words, n):
    
    words = words.split()
    new_words = words.copy()
    # n is the number of words to be swapped
    for _ in range(n):
        new_words = swap_word(new_words)
        
    sentence = ' '.join(new_words)
    
    return sentence

aug_syn = {'airline_sentiment':[],'detokenized_sents':[]}
columns = ['airline_sentiment','detokenized_sents']
for i in range(3000):
    if df1['airline_sentiment'][i]== 'positive' or df1['airline_sentiment'][i]== 'negative':
        new_row=random_swap(df1['detokenized_sents'][i], 1)
        aug_syn['airline_sentiment'].append(df1['airline_sentiment'][i])
        aug_syn['detokenized_sents'].append(new_row)
        
data3 = pd.DataFrame(aug_syn, columns=columns, index=None)
df2 = df1.append(data3,ignore_index=True)

#4-Random Insertion (RI)
def random_insertion(words, n):
    
    words = words.split()
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words)
        
    sentence = ' '.join(new_words)
    return sentence

def add_word(new_words):
    
    synonyms = []
    counter = 0
    
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
        
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)
    
    
aug_syn = {'airline_sentiment':[],'detokenized_sents':[]}
columns = ['airline_sentiment','detokenized_sents']

for i in range(2000):
    if df2['airline_sentiment'][i]== 'positive' :
        new_row=random_insertion(df2['detokenized_sents'][i], 1)
        aug_syn['airline_sentiment'].append(df2['airline_sentiment'][i])
        aug_syn['detokenized_sents'].append(new_row)
        
data4 = pd.DataFrame(aug_syn, columns=columns, index=None)
df3 = df2.append(data4,ignore_index=True)

# TF-IDF
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
tokenizer = TweetTokenizer()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer.tokenize)#it will takes uni_grams,bigrams(max)
vectorizer.fit(list(train['detokenized_sents'].values))
X_train=train['detokenized_sents']
X_test=valid['detokenized_sents']
y_train=train['airline_sentiment']
y_test=valid['airline_sentiment']
train_vectorized = vectorizer.transform(X_train)
test_vectorized = vectorizer.transform(X_test)
#getFreatures
ngramFeatures = vectorizer.get_feature_names()
np.array(ngramFeatures)

#Implementing ML algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,confusion_matrix
### 1- without EDA
#Logistic Regression
lr = LogisticRegression()
lr.fit(train_vectorized,y_train)

#Support Vector Machine (SVM)
from nltk import probability
svm = SVC(probability=True)
svm.fit(train_vectorized,y_train)

#Voting Classifier
#combination svm and LR
estimators = [ ('svm',svm) , ('lr' , lr) ]
clf = VotingClassifier(estimators , voting='soft')
clf.fit(train_vectorized,y_train)

###2- With EDA
#split data into training and validation set
X=df3['detokenized_sents']
Y=df3['airline_sentiment']
X_train1, X_test1, y_train1, y_test1=train_test_split(X,Y,test_size=0.2, random_state=42)
tokenizer1 = TweetTokenizer()
vectorizer1 = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenizer1.tokenize)
X_train2=["".join(review) for review in X_train1.values]
vectorizer1.fit(X_train2)
X_test2=["".join(review) for review in X_test1.values]
vectorizer1.fit(X_test2)
#vectorization
train_vectorized1 = vectorizer1.transform(X_train2)
test_vectorized1 = vectorizer1.transform(X_test2)

#Logistic Regression
lr = LogisticRegression()
lr.fit(train_vectorized1,y_train1)

#Support Vector Machine (SVM)
svm = SVC(probability=True)
svm.fit(train_vectorized1,y_train1)

#Voting Classifier
#combination svm and LR
estimators = [ ('svm',svm) , ('lr' , lr) ]
clf = VotingClassifier(estimators , voting='soft')
clf.fit(train_vectorized1,y_train1)

#Best Model Selection using ROC-AUC Curve
# Import label encoder
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
# Encode labels
y_test2= label_encoder.fit_transform(y_test1)

def roc_auc_curve():
              pred_prob1 = lr.predict_proba(test_vectorized1) #Logistic regression
              pred_prob2 = svm.predict_proba(test_vectorized1)#svm
              pred_prob3 = clf.predict_proba(test_vectorized1)#voting classifier
              
              #for the positive class
              # roc curve for models
              fpr1, tpr1, _ = roc_curve(y_test2, pred_prob1[:,1], pos_label=1)
              fpr2, tpr2, _ = roc_curve(y_test2, pred_prob2[:,1], pos_label=1)
              fpr3, tpr3, _ = roc_curve(y_test2, pred_prob3[:,1], pos_label=1)
              
              # roc curve for tpr = fpr
              random_probs = [0 for i in range(len(y_test2))]
              p_fpr, p_tpr, _ = roc_curve(y_test2, random_probs, pos_label=1)
              # auc scores
              auc_score1 = roc_auc_score(y_test2, pred_prob1,multi_class='ovr')# logistic regression
              auc_score2 = roc_auc_score(y_test2, pred_prob2,multi_class='ovr')# svm
              auc_score3 = roc_auc_score(y_test2, pred_prob3,multi_class='ovr')# voting classifier
             

              plt.style.use('seaborn')
              # plot roc curves
              plt.plot(fpr1, tpr1, linestyle='--',color='orange',label="Logistic Regression: AUC="+str(auc_score1))
              plt.plot(fpr2, tpr2, linestyle='--',color='green', label='SVM: AUC='+str(auc_score2))
              plt.plot(fpr3, tpr3, linestyle='--',color='red', label='Voting Classifier : AUC='+str(auc_score3))
          

              # plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
              plt.plot([0, 1], [0, 1],'r--')

              # title
              plt.title('ROC curve: ')
              # x label
              plt.xlabel('False Positive Rate')
              # y label
              plt.ylabel('True Positive rate')
              plt.legend(loc='best')
              plt.savefig('ROC',dpi=300)
              plt.show(); 

roc_auc_curve()

#Test More data
###1-using validation subset
#vectorize the text
test =vectorizer1.transform(valid['detokenized_sents'])

###2-using new phrases
#create a function  that takes a text as input and return the suitable sentiment 
def detect_sentment(text):
            #vectorize the text
            test = vectorizer1.transform([text])
            l=svm.predict(test)
            #Check for the prediction probability
            pred_proba=svm.predict_proba(test)
            pred_percentage_for_all=dict(zip(svm.classes_,pred_proba[0]))
            print("Prediction using SVM:  : {} , Prediction Score : {}".format(l[0],np.max(pred_proba)))
            print()
            print(pred_percentage_for_all)              

    
detect_sentment("i took a flight today!")

    