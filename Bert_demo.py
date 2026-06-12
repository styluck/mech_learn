# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:21:44 2024

@author: lich5

pip install pywaffle
pip install tensorflow_hub, or
pip install --upgrade tf-hub-nightly
pip install wordcloud==1.9.1.1

"""

import pandas as pd 
import matplotlib.pyplot as plt 
 
import seaborn as sns 
import numpy as np
from tensorflow.keras import Sequential, layers

import tensorflow_hub as hub
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from wordcloud import WordCloud
import tensorflow_text as text
from pywaffle import Waffle

df = pd.read_csv('youtoxic_english_1000.csv')
df.head()

df.shape
df.info()

# EDA
diff_of_gen = df.groupby(df['IsToxic'])['Text'].count().reset_index()
labels = ['IsToxic', 'not-IsToxic']
explode = (0, 0.1)
plt.pie(diff_of_gen['Text'], labels=labels, autopct='%1.1f%%', explode=explode, startangle=90)
plt.title('Distribution of IsToxic vs not-IsToxic Text')
plt.show()


fig = plt.figure(FigureClass = Waffle,
                  rows = 20, columns = 30, #pass the number of rows and columns for the waffle 
                  values = diff_of_gen['Text'], #pass the data to be used for display
                  cmap_name = 'tab20', #color scheme
                  legend = {'labels': [f"{k}, ({v})" for k, v in zip(diff_of_gen.IsToxic.values,diff_of_gen.Text)],
                            'loc': 'lower left', 'bbox_to_anchor':(0,-0.5),'ncol': 1}
                  #notice the use of list comprehension for creating labels 
                  #from index and total of the dataset
                )
plt.show()
#Display the waffle chart


text_gen = ' '.join(df[df['IsToxic']==1]['Text'])[0:df.shape[0]]
text_not_gen = ' '.join(df[df['IsToxic']==0]['Text'])[0:df.shape[0]]

text_cloud_gen = WordCloud().generate(text_gen)
text_cloud_not_gen = WordCloud().generate(text_not_gen)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Most frequent words in IsToxic text')
plt.imshow(text_cloud_gen, interpolation='bilinear')
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Mostb frequent words in not-IsToxic text')
plt.imshow(text_cloud_not_gen, interpolation='bilinear')
plt.axis('off')

plt.show()

rand_seed = 40
X_train, X_val, y_train, y_val = train_test_split(df['Text'], df['IsToxic'], 
                                                  test_size=0.2, 
                                                  random_state=rand_seed)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape )


class MyBertModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        PRE_HUB_URL = "https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-preprocess/versions/3"
        HUB_URL='https://www.kaggle.com/models/tensorflow/bert/frameworks/TensorFlow2/variations/en-uncased-l-12-h-768-a-12/versions/4'

        self.Input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        self.bert_preprocessor = hub.KerasLayer(PRE_HUB_URL)
        
        self.bert_encoder = hub.KerasLayer(HUB_URL, trainable=False)
        
        self.Linear = tf.keras.Sequential()
        self.Linear.add(tf.keras.layers.Dense(128, activation="relu"))
        self.Linear.add(tf.keras.layers.Dense(32, activation="relu"))
        self.Linear.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        
        Preprocessor = self.bert_preprocessor(self.Input)
        x1 = self.bert_encoder(Preprocessor)
        output = self.Linear(x1["pooled_output"])
        self.model=tf.keras.Model(inputs=[self.Input], outputs=[output])
        
    def call(self, inputs):
        
        return self.model(inputs)
    
model = MyBertModel()

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy',Precision(), Recall()])


hist = model.fit(X_train, y_train, epochs=1, batch_size=32 ,validation_data=(X_val, y_val))

model.summary()

hist.history.keys()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.title('Loss vs Validation loss performance')
plt.xlabel('Epoch')
plt.ylabel('error')
plt.show()

model.evaluate(X_val, y_val)

y_predict=model.predict(X_val)
y_pred=np.where(y_predict>0.5,1,0)

cr = classification_report(y_val, y_pred)
print(cr)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred)

# Print the confusion matrix
print("Confusion Matrix:\n", conf_matrix)


# demostrate the example result
test_df = pd.read_csv("test.csv")
test = test_df['text']

n = 10
idx = np.random.randint(1e4,size=n)

Demo = test[idx]
y_predict_demo = (model.predict(Demo)>.5)

Output = dict(zip(Demo.values, np.squeeze(y_predict_demo)))


# [EOF]