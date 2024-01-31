#!/usr/bin/env python
# coding: utf-8

# In[387]:


import numpy as np
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import csv


# In[388]:


with open ('C:\\Users\joe62\\OneDrive - Aberystwyth University\\Apps\\Desktop\\Research on demographic and health\\deep_learning_n.csv', 'r') as f:
      reader = csv.reader(f)
      deep_learning_n = list(reader)


# In[389]:


data = np.array(deep_learning_n, dtype=int)


# In[390]:


Description of the data
The data is from Demographic and health survey, Nigeria dataset 2018. The dependent varaible (1st coloumn of the dataset) was to
determine if a child bearing woman ever had terminated pregnancy, "0" was no as reponse while "1" referrred to "yes" as response, 
indepedent variables were the weight of the respondent, wealth index, number of household and no of children given
birth to by the respondent in column 2 to 5 respectively.


# In[391]:


print (data)


# In[392]:


len(data)


# Spliting data into training and test data 

# In[393]:


data = shuffle (data)


# In[394]:


training, test = np.array_split(data, [int(0.8*len(data))])


# In[395]:


len(training)


# In[396]:


print(training)


# In[397]:


len(test)


# Preparing target (dependent variable)

# In[398]:


dependent_train_data = training[0:,0]


# In[399]:


print(dependent_train_data)


# Reshape train data

# In[400]:


dependent_train_data= dependent_train_data.reshape(83846,1)


# In[401]:


print(dependent_train_data)


# In[402]:


len(dependent_train_data)


# Preparing target (independent variable)

# In[403]:


independent_variabls_train_data = training[:,1:]


# In[404]:


print(independent_variabls_train_data)


# In[405]:


len(independent_variabls_train_data)


# Reshape indepedent train data

# In[406]:


independent_variabls_train_data = independent_variabls_train_data.reshape(83846,4)


# In[407]:


print (independent_variabls_train_data )


# Normalization of features (independent variables)

# In[408]:


normalized_independent_train_samples = normalize(independent_variabls_train_data, axis=0)


# In[409]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_crossentropy


# In[410]:


print(normalized_independent_train_samples)


# Model

# In[411]:


model = Sequential([
      
     Dense(units=32, input_shape=(4,), activation = "relu"),
     Dense(units=64, activation ="relu"),
     Dense(units=128, activation ="relu"),
     Dropout(0.5),
     Dense(units=1, activation = "sigmoid")
])


# In[412]:


model.summary()


# Training the model

# In[413]:


model.compile(optimizer = Adam (learning_rate = 0.01), loss = "binary_crossentropy", metrics = ["binary_accuracy"])


# Model compile and ready for training with validation sample of 10%

# In[414]:


model.fit(x=normalized_independent_train_samples, y = dependent_variabl_train_data , validation_split = 0.1, batch_size = 32, epochs = 5, verbose = 2)


# Use model to predict

# Preparing dependent_test_data

# In[415]:


dependent_test_data = test[0:,0]


# In[416]:


len(dependent_test_data)


# In[417]:


print(dependent_test_data)


# Preparing independent variables

# In[418]:


independent_variabls_test_data = test[:,1:]


# In[419]:


normalized_independent_test_samples = normalize(independent_variabls_test_data , axis=0)


# In[420]:


print(normalized_independent_test_samples)


# Predict

# In[421]:


predictions = model.predict(x=independent_variabls_test_data, batch_size=10, verbose=0 )


# In[422]:


len(predictions)


# In[423]:


rounded_predictions = np.argmax(predictions, axis=1)


# In[424]:


print(rounded_predictions)


# In[425]:


rounded_predictions = rounded_predictions.reshape(20962, 1)


# In[426]:


print(rounded_predictions)


# In[427]:


len(rounded_predictions)


# In[428]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


# In[429]:


cm = confusion_matrix(y_true = dependent_test_data, y_pred = rounded_predictions )


# In[430]:


print(cm)


# In[431]:


def plot_confusion_matrix(cm, classes, 
                          normalize= False, 
                          title = 'Confusion matrix', 
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm=cm.astype('int')/ cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix' ),
    else:
            print('Confusion matrix, without normalization' )
    print(cm)
    
    thresh = cm.max() /2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],
                horizontalalignment= "center",
                 color = "white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[470]:


cm_plot_labels = ['had_never_terminated_Preg', 'Terminated_pregnancy'] 
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title = 'Confusion Matrix')


# Save the model

# In[433]:


import os.path


# In[434]:


os.chdir('C:\\Users\\joe62\OneDrive - Aberystwyth University\\Apps\Desktop')


# In[435]:


if os.path.isfile('C:\\Users\\joe62\OneDrive - Aberystwyth University\\Apps\Desktop\\Deep learning') is False: 
    model.save('Deep learning(Sequential sequential model')


# In[471]:


os.chdir('C:\\Users\\joe62')


# In[472]:


get_ipython().system('git init')


# In[477]:


get_ipython().system("git add 'C:\\\\Users\\\\joe62\\\\Deep_learning.ipynb'")


# In[456]:





# In[ ]:




