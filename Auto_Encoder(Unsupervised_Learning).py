
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist


# In[10]:


input_img=Input(shape=(784,)) #layer 1
encoded=Dense(32, activation='relu')
encoded=encoded(input_img) # layer 2
decoded=Dense(784, activation='sigmoid')
decoded=decoded(encoded) #layer 3
autoencoder=Model(input_img, decoded)


# In[11]:


autoencoder.summary()


# In[12]:


encoder=Model(input_img, encoded)
encoder.summary()


# In[14]:


encoded_input=Input(shape=(32,))
decoder_layer=autoencoder.layers[-1]
decoder=Model(encoded_input, decoder_layer(encoded_input))
decoder.summary()


# In[15]:


autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# In[16]:


(x_train,_), (x_test,_)=mnist.load_data()


# In[17]:


x_train=x_train/255
x_test=x_test/255
flattened_x_train=x_train.reshape(-1, 28 *28)
flattened_x_test=x_test.reshape(-1,28*28)
print(flattened_x_train.shape)
print(flattened_x_test.shape)


# In[19]:


fit_hist=autoencoder.fit(flattened_x_train, flattened_x_train,epochs=50,batch_size=256, validation_data=(flattened_x_test, flattened_x_test))


# In[22]:


encoded_img=encoder.predict(x_test[:10].reshape(-1,784))
decoded_img=decoder.predict(encoded_img)


# In[23]:


n=10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax=plt.subplot(2,10,i+1) #2행 10렬 이며 i인덱스가 0이기 때문에 i+1로 출발 ##원본 데이터
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax=plt.subplot(2,10,i+1+n) ##재구성된 데이터
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
##위-원본
##아래-재구성


# In[24]:


plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()


# In[32]:


n=10
plt.figure(figsize=(20,20))
for i in range(n):
    ax=plt.subplot(n,1,i+1)
    plt.imshow(encoded_img[i].reshape(1,32))
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


# In[33]:


input_img=Input(shape=(784,))
encoded=Dense(128, activation='relu')(input_img)
encoded=Dense(64, activation='relu')(encoded)
encoded=Dense(32, activation='relu')(encoded)

decoded=Dense(64,activation='relu')(encoded)
decoded=Dense(128,activation='relu')(encoded)
decoded=Dense(784,activation='sigmoid')(encoded)


# In[34]:


autoencoder=Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()


# In[35]:


autoencoder.fit(flattened_x_train, flattened_x_train, epochs=5, batch_size=256,
               validation_data=(flattened_x_test, flattened_x_test))


# In[36]:


decoded_img=autoencoder.predict(flattened_x_test)


# In[37]:


n=10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax=plt.subplot(2,10,i+1) # 원본 데이터
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax=plt.subplot(2,10,i+1+n) # 재구성된 데이터
    plt.imshow(decoded_img[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    


# In[38]:


input_img=Input(shape=(28,28,1)) # 'channels_firtst'이미지 데이터 형식을 사용하는 경우 이를 적용

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

## encoder의 shape = (samples, 4, 4, 8)

##복원
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x) #사이즈를 다시 키워주는 레이어
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()


# In[49]:


conv_x_train=np.reshape(x_train,(-1,28,28,1))
conv_x_test=np.reshape(x_test,(-1,28,28,1))

autoencoder.fit(conv_x_train,conv_x_train, batch_size=128, epochs=5,
validation_data=(conv_x_test,conv_x_test))


# In[52]:


decoded_imgs = decoder.predict(conv_x_test)
n = 10 # 이미지 갯수
plt.figure(figsize=(20, 4))
for i in range(n):
# 원본 데이터
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 재구성된 데이터
    ax = plt.subplot(2, n, i +1+ n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[54]:


noise_factor=0.5
x_train_noisy=conv_x_train+noise_factor*np.random.normal(
loc=0.0, scale=1.0, size=conv_x_train.shape)
x_test_noisy=conv_x_test+noise_factor*np.random.normal(
loc=0.0, scale=1.0, size=conv_x_test.shape)
x_train_noisy=np.clip(x_train_noisy,0.0, 1.0)
x_test_noisy=np.clip(x_train_noisy,0.0, 1.0)


# In[55]:


plt.gray()
n=10
plt.figure(figsize=(20,2))
for i in range(n):
    ax=plt.subplot(1,n,i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[56]:


fit_hist=autoencoder.fit(x_train_noisy, conv_x_train,
                        epochs=5, batch_size=128,
                        validation_data=(x_test_noisy, conv_x_test))


# In[ ]:


plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()


# In[ ]:


decoded_imgs = decoder.predict(x_test_noisy)
n = 10 # 이미지 갯수
plt.figure(figsize=(20, 4))
for i in range(n):
# 원본 데이터
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 재구성된 데이터
    ax = plt.subplot(2, n, i +1+ n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

