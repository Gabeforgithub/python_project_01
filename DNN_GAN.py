import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import*
from tensorflow.keras.models import Sequential

OUT_DIR='./OUT_img/'
img_shape=(28,28,1)
epoch=100000
batch_size=128
noise=100
sample_interval=100

(X_train,_),(_,_)=mnist.load_data()
print(X_train.shape)

X_train=X_train/127.5-1
X_train=np.expand_dims(X_train, axis=3)

print(X_train.shape)

generator_model=Sequential()
generator_model.add(Dense(128, input_dim=noise)) #이곳
generator_model.add(LeakyReLU(alpha=0.01))  #마이너스 값. 0~2값사이. 마이너스값에도 반응을 하는 렐루를 써줌. 알파가 민감도. 알파가 커질수록 마이너스값에 크게반응
generator_model.add(Dense(784, activation='tanh'))#이곳
generator_model.add(Reshape(img_shape)) #이미지를 그려봄
generator_model.summary()

lrelu=LeakyReLU(alpha=0.01)
discriminator_model=Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))
discriminator_model.add(Dense(128, activation=lrelu)) #이곳
discriminator_model.add(Dense(1,activation='sigmoid')) ##2, softmax를 써주면 np.ones, np.zeros후 원핫인코딩을 해줘야함
discriminator_model.summary()

discriminator_model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
discriminator_model.trainable=False #학습이 안되게 일단 막아둠

gan_model=Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
gan_model.summary()
gan_model.compile(loss='binary_crossentropy',optimizer='adam')

real=np.ones((batch_size,1))
print(real)
fake=np.zeros((batch_size,1))
print(fake)

for itr in range(epoch):
    idx=np.random.randint(0,X_train.shape[0], batch_size) #60000개(X_train.shape[0]번 인덱스), 이미지중 128개(배치사이즈) 추출
    real_imgs=X_train[idx] ##128개의 리얼 이미지

    z=np.random.normal(0,1,(batch_size,noise))
    fake_imgs=generator_model.predict(z)

    d_hist_real=discriminator_model.train_on_batch(real_imgs,real)
    d_hist_fake=discriminator_model.train_on_batch(fake_imgs,fake) #256번 학습

    d_loss, d_acc=0.5*np.add(d_hist_real, d_hist_fake)
    discriminator_model.trainable=False

    z=np.random.normal(0,1,(batch_size, noise))
    gan_hist=gan_model.train_on_batch(z, real) #128번 학습

    if itr%sample_interval==0:
        print('%d [D loss: %f, acc.: %.2f%%][G loss: %f]'%(itr,d_loss,d_acc*100,gan_hist))
        ##GAN모델의 로스는 결국 Generator의 로스
        row=col=4
        z=np.random.normal(0,1,(row*col, noise))
        fake_imgs=generator_model.predict((z))
        fake_imgs=0.5*fake_imgs+0.5
        _,axs=plt.subplots(row,col,figsize=(row,col),
                           sharey=True, sharex=True)
        cnt=0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt+=1
        path=os.path.join(OUT_DIR, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()