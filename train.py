import os
import numpy as np
from random import random, randint
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda,Activation, Input, GlobalAvgPool3D, Dense)
from keras.regularizers import l2 as l2l
from keras.models import Model
config = tf.ConfigProto()
sess = tf.Session(config=config)
size = 32
epoch = 10
xtrain = np.ones((465, 32, 32, 32))
xtest = np.ones((117, 32, 32, 32))
i = 0
path = "dataset/train_val"
path_list = os.listdir(path)
path_list.sort()
K2 = {}
def _conv_block(x, filters):
    bn,activation,kernel_initializer,decay,bottleneck = True,(lambda: Activation('relu')),'he_uniform',0.0001,4
    x = Conv3D(filters, kernel_size=(3, 3, 3),padding='same', use_bias=True,kernel_initializer=kernel_initializer, kernel_regularizer=l2l(decay))(activation()(BatchNormalization(scale=bn, axis=-1)(Conv3D(filters * bottleneck,kernel_size=(1, 1, 1), padding='same',use_bias=False,kernel_initializer=kernel_initializer,kernel_regularizer=l2l(decay))(activation()(BatchNormalization(scale=bn, axis=-1)(x))))))
    return x
def _dense_block(x, n):
    for _ in range(n):
        x = concatenate([_conv_block(x, 16), x], axis=-1)
    return x
def _transmit_block(x, is_last):
    bn = True
    activation = lambda: Activation('relu')
    kernel_initializer = 'he_uniform'
    decay = 0.0001
    compression = 2
    x = activation()(BatchNormalization(scale=bn, axis=-1)(x))
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = AveragePooling3D((2, 2, 2), padding='valid')(Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,kernel_initializer=kernel_initializer,kernel_regularizer=l2l(decay))(x))
    return x
def get_model(models=None, **kwargs):
    for k, v in kwargs.items():
        assert k in K2
        K2[k] = v
    d = [32, 32, 32]
    stscale, stlayer,kernel_initializer, decay,downstructure,output_size,shape= (lambda x: x / 128. - 1.),32,'he_uniform',0.0001,[4, 4, 4],2,d + [1]
    inputs = Input(shape=shape)
    if stscale is not None:
        scaled = Lambda(stscale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(stlayer, kernel_size=(3, 3, 3), padding='same', use_bias=True,kernel_initializer=kernel_initializer,kernel_regularizer=l2l(decay))(scaled)
    downsample_times = len(downstructure)
    for l, n in enumerate(downstructure):
        db = _dense_block(conv, n)
        conv = _transmit_block(db, l == downsample_times - 1)
    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'
    outputs = Dense(output_size, activation=last_activation,kernel_regularizer=l2l(decay),kernel_initializer=kernel_initializer)(conv)
    model = Model(inputs, outputs)
    model.summary()
    if models is not None:
        model.load_models(models, by_name=True)
    return model
def get_compiled(loss='categorical_crossentropy', optimizer='adam',metrics=["categorical_accuracy"],models=None, **kwargs):
    model = get_model(models=models, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,metrics=[loss] + metrics)
    return model
for filename in path_list:
    k3k = np.load(os.path.join(path, filename))
    voxel = k3k['voxel']
    seg = k3k['seg']
    xtrain[i] = (voxel * seg)[50 - 16:50 + 16, 50 - 16:50 + 16, 50 - 16:50 + 16]
    i = i + 1
i = 0
path = "dataset/test"
path_list = os.listdir(path)
path_list.sort()
for filename in path_list:
    k3k = np.load(os.path.join(path, filename))
    xtest[i] = (k3k['voxel'] * k3k['seg'])[50 - 16:50 + 16, 50 - 16:50 + 16, 50 - 16:50 + 16]
    i = i + 1
path = 'dataset/train_val.csv'
ytrain = np.loadtxt(path, int, delimiter=",", skiprows=1, usecols=1)
xval,yval,xk3k,yk3k,xtrain,ytrain   = xtrain.copy(),ytrain.copy(),xtrain.copy(),ytrain.copy(),np.ones((465 * 3, 32, 32, 32)),np.ones(465 * 3)
for i in range(0, 465):
    tarray1,tarray2,tarray3 = xk3k[i].copy(),xk3k[i].copy(),xk3k[i].copy()
    for j in range(0, 16):
        tarray1[j, :, :], tarray1[32 - 1 - j, :, :], tarray1[:, j, :], tarray1[:, 32 - 1 - j, :] = tarray1[32 - 1 - j,:, :], tarray1[j, :,:], tarray1[:,32 - 1 - j,:], tarray1[:, j,:]
    for j in range(0, 16):
        tarray2[:, j, :], tarray2[:, 32 - 1 - j, :],tarray2[:, :, j], tarray2[:, :, 32 - 1 - j] = tarray2[:, 32 - 1 - j, :], tarray2[:, j, :], tarray2[:, :,32 - 1 - j], tarray2[ :, :, j]
    for j in range(0, 16):
        tarray3[:, :, j], tarray3[:, :, 32 - 1 - j], tarray3[j, :, :], tarray3[32 - 1 - j, :, :] = tarray3[:, :,32 - 1 - j], tarray3[ :, :,j], tarray3[32 - 1 - j,:,:], tarray3[j, :,:]
    xtrain[i + 465 * 0], xtrain[i + 465 * 1], xtrain[i + 465 * 2] = tarray1.copy(), tarray2.copy(), tarray3.copy()
ytrain[465 * 0:465 * 1], ytrain[465 * 1:465 * 2], ytrain[465 * 2:465 * 3] = yk3k.copy(),yk3k.copy(),yk3k.copy()
xtrain, xval, xtest, ytrain, yval = xtrain.reshape(xtrain.shape[0], 32, 32, 32, 1), xval.reshape(xval.shape[0], 32, 32,32, 1), xtest.reshape(xtest.shape[0], 32, 32, 32, 1), to_categorical(ytrain, 2), to_categorical(yval, 2)
xval = xval.reshape(xval.shape[0], 32, 32, 32, 1), xtest.reshape(xtest.shape[0], 32, 32, 32, 1),to_categorical(ytrain, 2),to_categorical(yval, 2)
model = get_compiled()
class RocAuc2(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.xval,self.yval = validation_data
    def on_epoch_end(self, epoch, log={}):
        if epoch % self.interval == 0:
            ypred = self.model.predict(self.xval, verbose=0)
            score = roc_auc_score(self.yval, ypred)
            return score
RocAuc = RocAuc2(validation_data=(xval, yval), interval=1)
checkpointer = ModelCheckpoint(filepath='{epoch:02d}.h5', verbose=1, period=1, save_weights_only=True)
model.fit(xtrain,ytrain,epochs=epoch,validation_data=(xval, yval),shuffle=False,size=32,callbacks=[RocAuc, checkpointer])
trainb = model.predict(xval, 32, verbose=1)
testb = model.predict(xtest, 32, verbose=1)
print('finish')
