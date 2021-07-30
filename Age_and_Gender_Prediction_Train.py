#!/usr/bin/env python
# coding: utf-8

# #### Data Preprocessing

# In[84]:


fldr="sample"   #this is path to sample images containing 1024 images. Origianal dataset contained 23K images so it taking too long to train


# In[85]:


import os
files=os.listdir(fldr)   


# In[86]:


#reading images

import cv2
ages=[]
genders=[]
images=[]

for fle in files:
  age=int(fle.split('_')[0])
  gender=int(fle.split('_')[1])
  total=fldr+'/'+fle
  print(total)
  image=cv2.imread(total)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  image= cv2.resize(image,(48,48))
  images.append(image)

  


# In[87]:


for fle in files:
  age=int(fle.split('_')[0])
  gender=int(fle.split('_')[1])
  ages.append(age)
  genders.append(gender)


# In[88]:


from cv2_plt_imshow import cv2_plt_imshow
cv2_plt_imshow(images[24])


# In[89]:


print(ages[24])
print(genders[24])


# In[90]:


cv2_plt_imshow(images[53])


# In[91]:


print(ages[53])
print(genders[53])


# In[92]:


import numpy as np
images_f=np.array(images)
genders_f=np.array(genders)
ages_f=np.array(ages)


# In[93]:


np.save(fldr+'image.npy',images_f)
np.save(fldr+'gender.npy',genders_f)
np.save(fldr+'age.npy',ages_f)


# Male = 0
# Female= 1

# In[94]:



values, counts = np.unique(genders_f, return_counts=True)
print(counts)


# In[95]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
gender = ['Male', 'Female']
values=[4372,5047]
ax.bar(gender,values)
plt.show()


# In[96]:



values, counts = np.unique(ages_f, return_counts=True)
print(counts)


# In[97]:


val=values.tolist()
cnt=counts.tolist()


# In[98]:


plt.plot(counts)
plt.xlabel('ages')
plt.ylabel('distribution')
plt.show()


# In[99]:


labels=[]

i=0
while i<len(ages):
  label=[]
  label.append([ages[i]])
  label.append([genders[i]])
  labels.append(label)
  i+=1


# In[100]:


images_f_2=images_f/255


# In[101]:


labels_f=np.array(labels)


# In[102]:


images_f_2.shape


# In[103]:


import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[104]:


X_train, X_test, Y_train, Y_test= train_test_split(images_f_2, labels_f,test_size=0.25)


# In[105]:


Y_train[0:5]


# In[106]:


Y_train_2=[Y_train[:,1],Y_train[:,0]]
Y_test_2=[Y_test[:,1],Y_test[:,0]]


# In[107]:


Y_train_2[0][0:5]


# In[108]:


Y_train_2[1][0:5]


# ### Model

# In[109]:


from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten,BatchNormalization
from tensorflow.keras.layers import Dense, MaxPooling2D,Conv2D
from tensorflow.keras.layers import Input,Activation,Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def Convolution(input_tensor,filters):
    
    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.001))(input_tensor)
    x = Dropout(0.1)(x)
    x= Activation('relu')(x)

    return x
def model(input_shape):
  inputs = Input((input_shape))
  
  conv_1= Convolution(inputs,32)
  maxp_1 = MaxPooling2D(pool_size = (2,2)) (conv_1)
  conv_2 = Convolution(maxp_1,64)
  maxp_2 = MaxPooling2D(pool_size = (2, 2)) (conv_2)
  conv_3 = Convolution(maxp_2,128)
  maxp_3 = MaxPooling2D(pool_size = (2, 2)) (conv_3)
  conv_4 = Convolution(maxp_3,256)
  maxp_4 = MaxPooling2D(pool_size = (2, 2)) (conv_4)
  flatten= Flatten() (maxp_4)
  dense_1= Dense(64,activation='relu')(flatten)
  dense_2= Dense(64,activation='relu')(flatten)
  drop_1=Dropout(0.2)(dense_1)
  drop_2=Dropout(0.2)(dense_2)
  output_1= Dense(1,activation="sigmoid",name='sex_out')(drop_1)
  output_2= Dense(1,activation="relu",name='age_out')(drop_2)
  model = Model(inputs=[inputs], outputs=[output_1,output_2])
  model.compile(loss=["binary_crossentropy","mae"], optimizer="Adam",
	metrics=["accuracy"])
  
  return model


# In[110]:


Model=model((48,48,3))


# In[111]:


Model.summary()


# #### Training

# In[112]:


from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf


# In[113]:


fle_s='Age_sex_detection.h5'
checkpointer = ModelCheckpoint(fle_s, monitor='val_loss',verbose=1,save_best_only=True,save_weights_only=False, mode='auto',save_freq='epoch')
Early_stop=tf.keras.callbacks.EarlyStopping(patience=75, monitor='val_loss',restore_best_weights=True),
callback_list=[checkpointer,Early_stop]


# In[137]:


History=Model.fit(X_train,Y_train_2,batch_size=64,validation_data=(X_test,Y_test_2),epochs=500,callbacks=[callback_list])


# ### Evaluation

# In[138]:


Model.evaluate(X_test,Y_test_2)


# In[139]:


pred=Model.predict(X_test)


# In[140]:


pred[1]


# In[141]:


plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)


# ### For Gender

# In[142]:


plt.plot(History.history['sex_out_accuracy'])
plt.plot(History.history['val_sex_out_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.subplots_adjust(top=1.00, bottom=0.0, left=0.0, right=0.95, hspace=0.25,
                        wspace=0.35)


# ### For age

# In[143]:


fig, ax = plt.subplots()
ax.scatter(Y_test_2[1], pred[1])
ax.plot([Y_test_2[1].min(),Y_test_2[1].max()], [Y_test_2[1].min(), Y_test_2[1].max()], 'k--', lw=4)
ax.set_xlabel('Actual Age')
ax.set_ylabel('Predicted Age')
plt.show()


# #### For Gender

# In[144]:


i=0
Pred_l=[]
while(i<len(pred[0])):

  Pred_l.append(int(np.round(pred[0][i])))
  i+=1


# In[145]:


from sklearn.metrics import confusion_matrix 

from sklearn.metrics import classification_report 


# In[146]:



report=classification_report(Y_test_2[0], Pred_l)


# In[147]:


print(report)


# In[148]:



results = confusion_matrix(Y_test_2[0], Pred_l)


# In[149]:


import seaborn as sns

sns.heatmap(results, annot=True)


# In[150]:


def test_image(ind,images_f,images_f_2,Model):
  cv2_plt_imshow(images_f[ind])
  image_test=images_f_2[ind]
  pred_1=Model.predict(np.array([image_test]))
  #print(pred_1)
  sex_f=['Male','Female']
  age=int(np.round(pred_1[1][0]))
  sex=int(np.round(pred_1[0][0]))
  print("Predicted Age: "+ str(age))
  print("Predicted Sex: "+ sex_f[sex])


# ### Test 

# In[151]:


test_image(57,images_f,images_f_2,Model)


# In[152]:


test_image(137,images_f,images_f_2,Model)


# In[153]:


test_image(502,images_f,images_f_2,Model)


# In[154]:


test_image(24,images_f,images_f_2,Model)


# In[ ]:





# In[ ]:




