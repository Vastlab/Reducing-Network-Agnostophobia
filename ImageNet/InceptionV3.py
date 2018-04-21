import numpy as np
import sys
import os
from time import time
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_bg_cls", dest="use_bg_cls", action="store_true",default=False)
parser.add_argument("--dont_include_known_unknowns", dest="include_known_unknowns", action="store_false", default=True)
parser.add_argument("--split_no", action="store", dest="split_no", type=int, default = 0)
parser.add_argument("--no_of_epochs", action="store", dest="no_of_epochs", type=int, default = 50)
parser.add_argument("--model", action="store", dest="use_model", default = None)
parser.add_argument('--gpus', nargs='+', help='GPU No or Numbers for Multiple GPUs', required=True, type=int)
parser.add_argument("--use_ring_loss", dest="use_ring_loss", action="store_true", default=False)
parser.add_argument('--cross_entropy_loss_weight', help='cross_entropy_loss_weight', type=float, default=1.)
parser.add_argument('--ring_loss_weight', help='ring_loss_weight', type=float, default=1.)
parser.add_argument("--snapshot_location", action="store", dest="snapshot_location", default ='')
parser.add_argument("--fine_tune", dest="fine_tune", action="store_true", default=False)

parser.add_argument('--no_of_known_unknown_classes', action="store", dest="no_of_known_unknown_classes", help='no_of_known_unknown_classes', type=int)

parser.add_argument("--solver", action="store", dest="solver", default = 'adam')
parser.add_argument("--lr", action="store", dest="lr", default = 0.1, type=float)
parser.add_argument("--decay", action="store", dest="decay", default = 0.9, type=float)

args = parser.parse_args()


model_file_path='/home/adhamija/faceness/ImageNet/{}'
if not args.include_known_unknowns:
    model_file_path = model_file_path.format('Pretrained_model/')
elif args.use_bg_cls:
    model_file_path = model_file_path.format('BG_Cls_Data/')
elif args.use_ring_loss:
    model_file_path = model_file_path.format('Ring_Loss_Data/')
else:
    model_file_path = model_file_path.format('Cross_Entr_Data/')

model_file_path = os.path.join(model_file_path,args.snapshot_location)
print model_file_path

if os.path.exists(model_file_path):
    print "Snapshot Location Exists ... Exiting!!!"
    exit()
    
os.makedirs(model_file_path)

logging.basicConfig(filename=model_file_path+'/log',level=logging.DEBUG)
for arg, value in sorted(vars(args).items()):
    logging.info("Argument %s: %r", arg, value)


#common_name='Model_Epoch_number_{epoch:02d}.hdf5'
common_name='Model_Epoch_number_{}.hdf5'
if args.use_ring_loss:
    common_name = 'Model_Epoch_number_'+str(args.cross_entropy_loss_weight)+'_' +str(args.ring_loss_weight)+'_{}.hdf5'

model_file = os.path.join(model_file_path,common_name)

parallel=False
Batch_Size = 64

if len(args.gpus)>1:
    Batch_Size=Batch_Size*len(args.gpus)
    
data_generator_params = dict(
                                batch_size=Batch_Size,
                                use_bg_cls=args.use_bg_cls,
                                split_no=args.split_no,
                                no_of_known_unknown_classes=args.no_of_known_unknown_classes,
                                debug=False #True
                            )

if args.use_ring_loss:
    data_generator_params['unknownsMaximumMag'] = 5.
    data_generator_params['knownsMinimumMag'] = 10.

if parallel:
    from imagenet_data_prep import parallelized_imagenet_data_prep    
    training_generator = parallelized_imagenet_data_prep(
                                                            db_type='train',
                                                            include_known_unknowns = args.include_known_unknowns,
                                                            shuffle=True,
                                                            **data_generator_params
                                                        )

    validation_generator = parallelized_imagenet_data_prep(
                                                            db_type='val',
                                                            training_data_obj=training_generator,
                                                            shuffle=False,
                                                            **data_generator_params
                                                        )

    fit_generator_params = dict(
                                workers=1,
                                steps_per_epoch=training_generator.get_no_of_batches,
                                validation_steps=validation_generator.get_no_of_batches,
                                use_multiprocessing=False
                                )
else:
    # This Implementation is slightly slower and uses the keras tutorial approach    
    from imagenet_data_prep import imagenet_data_prep    
    training_generator = imagenet_data_prep(
                                            db_type='train',
                                            include_known_unknowns = args.include_known_unknowns,
                                            shuffle=True,
                                            **data_generator_params
                                            )

    validation_generator = imagenet_data_prep(
                                                db_type='val',
                                                training_data_obj=training_generator,
                                                shuffle=False,
                                                **data_generator_params
                                            )
    fit_generator_params = dict(
                                workers=25,
                                use_multiprocessing=False # Dont set to true makes processing slow
                               )
    

import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Activation, Input
from keras import backend as K
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
import tensorflow as tf


def ring_loss(y_true,y_pred):
    pred=K.sqrt(K.sum(K.square(y_pred),axis=1))
    error=K.mean(K.square(
        # Loss for Knowns having magnitude greater than knownsMinimumMag
        y_true[:,0]*(K.maximum(knownsMinimumMag-pred,0.))
        # Add two losses
        +
        # Loss for unKnowns having magnitude greater than unknownsMaximumMag
        y_true[:,1]*(K.maximum(pred-unknownsMaximumMag,0.))
    ))
    return error


if len(args.gpus)==1:
#    os.environ["CUDA_VISIBLE_DEVICES"]='1'#str(args.gpus[0])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(args.gpus[0])
    set_session(tf.Session(config=config))


if args.use_model is None:
    #tf.reset_default_graph()
#    with tf.device('/gpu:0'):
#    with tf.device('/cpu:0'):
    if args.fine_tune:
        print "............. Initializing with ImageNet weights ..........."
        base_model = InceptionV3(weights='imagenet', include_top=False)
    else:
        print "............. Initializing with Random weights ..........."
        base_model = InceptionV3(weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024,name='fc',use_bias=True)(x)

    if args.use_bg_cls:
        pred = Dense(101,name='pred',use_bias=False)(x)
    else:
        pred = Dense(100,name='pred',use_bias=False)(x)

    softmax_output = Activation('softmax',name='softmax')(pred)
    model = Model(inputs=base_model.input, outputs=softmax_output)

else:
    
    model = keras.models.load_model(args.use_model)
    print "Loded Model",args.use_model
    
    if args.use_bg_cls and model.output_shape[1]!=101:
        # Adding additional node for the Background class to the fully connected layer
        weights = model.get_layer('pred').get_weights()
        # Adding the Bias Term
        if len(weights)>1:
            print "Adding Bias as Zero",weights[1]
            weights[1] = np.concatenate((weights[1],np.zeros(1)),axis=0)
        # Adding the Weights and initializing them
        weights[0] = np.concatenate((weights[0],-0.0001*np.random.random_sample((weights[0].shape[0],1)) + 0.0001),axis=1)
        
        model.layers.pop()
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
        output = model.get_layer('fc').output
        #output = Flatten()(output)
        pred = Dense(101,name='pred',use_bias=False)(output)
        softmax_output = Activation('softmax',name='softmax')(pred)
        model = Model(inputs=model.input, outputs=softmax_output)
        model.get_layer('pred').set_weights(weights)

        
if args.fine_tune:
    print "............. Training Last Two Inception Blocks Only ..........."
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
else:
    print "............. Training all Layers ..........."
    for layer in model.layers:
        layer.trainable = True


#optimizing_algo = RMSprop(lr=0.045, rho=0.9, epsilon=1., decay=0.9)
#optimizing_algo = SGD(lr=0.00001, decay=0.0009, momentum=0.8, nesterov=False)
#adam = RMSprop(lr=0.00018, rho=0.9, epsilon=1., decay=0.045, clipnorm=2.0)
#adam = SGD(lr=0.01, decay=0.0009, momentum=0.8, nesterov=False)


if args.solver == 'adam':
    optimizing_algo = Adam(lr=args.lr)
elif args.solver == 'rms':
    optimizing_algo = RMSprop(lr=args.lr, rho=0.9, epsilon=1., decay=args.decay)
elif args.solver == 'sgd':
    optimizing_algo = SGD(lr=args.lr, decay=args.decay, momentum=0.8, nesterov=False)

    

if args.use_ring_loss:
    unknownsMaximumMag = Input(shape=(1,), dtype='float32', name='unknownsMaximumMag')
    knownsMinimumMag = Input(shape=(1,), dtype='float32', name='knownsMinimumMag')
    softmax_output = model.get_layer('softmax').output
    fc = model.get_layer('fc').output
    model = Model(
                    inputs=[
                                model.input,
                                unknownsMaximumMag,
                                knownsMinimumMag
                            ], 
                    outputs=[
                                softmax_output, 
                                fc
                            ]
                 )



if len(args.gpus)>1:
    parallel_model = multi_gpu_model(model, gpus=args.gpus)
else:
    parallel_model = model

    
# compile the model (should be done *after* setting layers to non-trainable)
if args.use_ring_loss:
    parallel_model.compile(
                            optimizer=optimizing_algo,
                            loss={'softmax': 'categorical_crossentropy','fc':ring_loss},
                            loss_weights={'softmax': args.cross_entropy_loss_weight, 'fc': args.ring_loss_weight},
                            metrics=['acc']
                        )

else:
    parallel_model.compile(optimizer=optimizing_algo, loss={'softmax':'categorical_crossentropy'},metrics=['acc'])



tensorboard = TensorBoard(log_dir=os.path.join(model_file_path,"logs/{}".format(time())))



class model_saver_cb(keras.callbacks.Callback):
    def __init__(self, model):
        self.model_to_save = model
        
    def on_epoch_end(self, epoch, logs={}):
        print "Saving model at : ",model_file.format(epoch)
        self.model_to_save.save(model_file.format(epoch))

model_saver = model_saver_cb(model)

info = parallel_model.fit_generator(
                            generator=training_generator,
                            validation_data=validation_generator,
                            max_queue_size=50,
                            epochs=args.no_of_epochs,
                            callbacks=[model_saver,tensorboard],
#                            verbose=2,
                            **fit_generator_params
                            )

#parallel_model.save(model_file)