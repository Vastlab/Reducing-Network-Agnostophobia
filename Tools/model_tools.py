from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten,Input,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Activation,Concatenate
from keras.models import Model
from keras import backend as K
import numpy as np


def LeNet_plus_plus(perform_L2_norm=False,activation_type='softmax',ring_approach=False,background_class=False, knownsMinimumMag = None):
    
    mnist_image = Input(shape=(28, 28, 1), dtype='float32', name='mnist_image')

    # 28 X 28 --> 14 X 14
    conv1_1 = Conv2D(32, (5,5), strides=1, padding="same",name='conv1_1')(mnist_image)
    conv1_2 = Conv2D(32, (5,5), strides=1, padding="same",name='conv1_2')(conv1_1)
    conv1_2 = BatchNormalization(name='BatchNormalization_1')(conv1_2)
    pool1 = MaxPooling2D(pool_size=(2,2), strides=2,name='pool1')(conv1_2)
    # 14 X 14 --> 7 X 7
    conv2_1 = Conv2D(64, (5,5), strides=1, padding="same", name='conv2_1')(pool1)
    conv2_2 = Conv2D(64, (5,5), strides=1, padding="same", name='conv2_2')(conv2_1)
    conv2_2 = BatchNormalization(name='BatchNormalization_2')(conv2_2)
    pool2 = MaxPooling2D(pool_size=(2,2), strides=2, name='pool2')(conv2_2)
    # 7 X 7 --> 3 X 3
    conv3_1 = Conv2D(128, (5,5), strides=1, padding="same",name='conv3_1')(pool2)
    conv3_2 = Conv2D(128, (5,5), strides=1, padding="same",name='conv3_2')(conv3_1)
    conv3_2 = BatchNormalization(name='BatchNormalization_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2,2), strides=2, name='pool3')(conv3_2)
    flatten=Flatten(name='flatten')(pool3)
    fc = Dense(2,name='fc',use_bias=True)(flatten)

    if perform_L2_norm:
        alpha_multipliers = Input((1,), dtype='float32', name='alphas')
        act = Activation(lambda x: alpha_multipliers*(K.l2_normalize(x,axis=1)),name='act')(fc)
        pred = Dense(10, activation=activation_type,name='pred',use_bias=False)(act)
        model = Model(inputs=[mnist_image,alpha_multipliers], outputs=[pred])
    elif knownsMinimumMag is not None:
        knownUnknownsFlag = Input((1,), dtype='float32', name='knownUnknownsFlag')
        pred = Dense(10, name='pred',use_bias=False)(fc)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[mnist_image,knownsMinimumMag], outputs=[softmax,fc])
    elif background_class:
        pred = Dense(11, name='pred',use_bias=False)(fc)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[mnist_image], outputs=[softmax])
    else:
        """
        pred = Dense(10, name='pred',use_bias=False)(fc)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[mnist_image], outputs=[softmax])
        """
        pred = Dense(10, name='pred', use_bias=False)(fc)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[mnist_image], outputs=[softmax])
    return model






def LeNet(activation_type='softmax',ring_approach=False,background_class=False, knownsMinimumMag = None):
    
    mnist_image = Input(shape=(28, 28, 1), dtype='float32', name='mnist_image')
    
    conv1 = Conv2D(20, (5,5), padding="same",name='conv1')(mnist_image)
    act1 = Activation("relu",name="act1")(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2), strides=2,name='pool1')(act1)

    conv2 = Conv2D(50, (5,5), padding="same",name='conv2')(pool1)
    act2 = Activation("relu",name="act2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2,name='pool2')(act2)

    flatten=Flatten(name='flatten')(pool2)
    fc = Dense(500,name='fc',use_bias=True)(flatten)
#    act3 = Activation("relu",name="act3")(fc)
    act3 = fc
    
    if knownsMinimumMag is not None:
        knownUnknownsFlag = Input((1,), dtype='float32', name='knownUnknownsFlag')
        pred = Dense(10, name='pred',use_bias=False)(act3)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[mnist_image,knownsMinimumMag], outputs=[softmax,fc])
    elif background_class:
        pred = Dense(11, name='pred',use_bias=False)(act3)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[mnist_image], outputs=[softmax])
    else:
        pred = Dense(10, name='pred', use_bias=False)(act3)
        softmax = Activation(activation_type,name='softmax')(pred)
        model = Model(inputs=[mnist_image], outputs=[softmax])
    return model





def extract_features(model,data,layer_name = ['fc','softmax']):
    out=[]
    for l in layer_name:
        out.append(model.get_layer(l).output)
    intermediate_layer_model = Model(inputs=model.input,outputs=out)
    if len(model.input_shape)==4:
        intermediate_output = intermediate_layer_model.predict([data])
    elif len(model.input_shape)==3:
        intermediate_output = intermediate_layer_model.predict([data,np.ones(data.shape[0]),np.ones(data.shape[0])])        
    else:
        intermediate_output = intermediate_layer_model.predict([data,np.ones(data.shape[0])])
    return intermediate_output



def concatenate_training_data(obj,y,cross_entropy_probs,ring_loss=False):
    """
    Parameters:
        obj: is an object from a class in file data_prep.py
        y: are the images from the class that needs to be trained as negatives example cifar.images or letters.images
        cross_entropy_probs: Multiplier to the categorical labels
        ring_loss: Boolean value returns Y_pred_flags (Default:False)
    Returns:
        X_train_data: Numpy array containing training samples
        Y_train_data: Numpy array containing Label values
        sample_weights: 1D Numpy array containing weight of each sample
        Y_train_flags: Returned only when ring_loss=True. Numpy array containing flags indicating the sample is a known versus known unknown
    """
    X_train_data=np.concatenate((obj.X_train,y))
    Y_train_data=np.concatenate((obj.Y_train,np.ones((y.shape[0],10))*cross_entropy_probs))
    class_no=np.argmax(obj.Y_train,axis=1)
    sample_weights_knowns=np.zeros_like(class_no).astype(np.float32)
    for cls in range(obj.Y_train.shape[1]):
        sample_weights_knowns[class_no==cls]=100./len(class_no[class_no==cls])
    sample_weights=np.concatenate([sample_weights_knowns,np.ones((y.shape[0]))*(100./y.shape[0])])
    if ring_loss:
        Y_train_flags=np.zeros((X_train_data.shape[0],2))
        Y_train_flags[:obj.X_train.shape[0],0]=1
        Y_train_flags[obj.X_train.shape[0]:,1]=1
        return X_train_data,Y_train_data,sample_weights,Y_train_flags
    else:
        return X_train_data,Y_train_data,sample_weights
