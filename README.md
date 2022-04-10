# Sentiment_analysis_Project
Preforming Sentiment detection on several texts using machine learning algorithms.

methodology followed in the project is represented in this map:


<img  alt="image" src="https://user-images.githubusercontent.com/66137466/162644007-3c30fc0e-e109-4329-a339-3152b0545037.PNG">

**1-Data unserstanding**
We used the Aireline sentiment analysis from kaggle we've focused on two features:
airline_sentiment
Text 

**2-Data preprocessing**
This is one of the most important steps in any modelisation probleme, data preprocessing plays a crucial role since the modelisation technics are not equipped to process non-structed data especially in our case, where we're dealing with textual data. 

**3-Data Augmentation**
Data augmentation techniques are used to generate additional, synthetic data using the data you have. Augmentation methods are super popular in computer vision applications but they are just as powerful for NLP. but it should be done carefully due to the grammatical structure of the text. In our case we use EDA(Easy Data Augmentation) method.The method is used before training. A new augmented dataset is generated beforehand and later fed into data loaders to train the model.

![Capture](https://user-images.githubusercontent.com/66137466/162643350-5b7c6677-d4bb-4024-a1af-8b15d3fb41df.PNG)


**4-Modeling**
After getting our data ready, and compatible with machine learning algorithms inputs, we're ready tobuild our model, the challenge here is that we have several types of algorithms and we will have to chose which one preforms the best in our case. 

**5-Evaluation**
After building our models we move to evaluationg them using different technics.

After succesfully cleaning our dataset, we move to building the matrix how is it done?


#  **Vectorization**
To move on to the creation of machine learning models, we must first transform the text into a data matrix that corresponds to the processing by ML algorithms,while trying to minimizing the loss of information as much as possible.
each line in our dataset will represent the lines of our matrix hence we speak of a vector presentation, but in order to determine the features or the indexes we will use the TF-IDF vectorizer.



#  **Models**

For this problem, we used 3 classification models:
 ### Logistic Regression
 ### Support Vector Machine
 ### Voting Classifier
 
 **Result**:
Using ROC-AUC curve, we found that SVM  the best model, because it was able to distinguish more or less between the Sentiments.

![Capture](https://user-images.githubusercontent.com/66137466/162643830-d1418ae1-685b-411d-baef-5f8bf8adb0c4.PNG)


