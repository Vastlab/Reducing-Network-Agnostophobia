import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_bg_cls", dest="use_bg_cls", action="store_true",default=False)
parser.add_argument("--dont_include_known_unknowns", dest="include_known_unknowns", action="store_false", default=True)
parser.add_argument("--split_no", action="store", dest="split_no", type=int, default = 0)
parser.add_argument("--no_of_epochs", action="store", dest="no_of_epochs", type=int, default = 25)
parser.add_argument("--model", action="store", dest="use_model", default = None)
parser.add_argument('--gpus', nargs='+', help='GPU No or Numbers for Multiple GPUs', required=True, type=int)
parser.add_argument("--use_ring_loss", dest="use_ring_loss", action="store_true", default=False)
parser.add_argument('--cross_entropy_loss_weight', help='cross_entropy_loss_weight', type=float, default=1.)
parser.add_argument('--ring_loss_weight', help='ring_loss_weight', type=float, default=1.)
parser.add_argument("--snapshot_location", action="store", dest="snapshot_location", default ='')
args = parser.parse_args()

parallel=False
Batch_Size = 32

if len(args.gpus)>1:
    Batch_Size=Batch_Size*len(args.gpus)
    
data_generator_params = dict(
                                batch_size=Batch_Size,
                                use_bg_cls=args.use_bg_cls,
                                split_no=args.split_no,
                                debug=True#False #True
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
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
import tensorflow as tf



#import os 
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
    if True:
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024,name='fc')(x)

        if args.use_bg_cls:
            pred = Dense(101,name='pred')(x)
        else:
            pred = Dense(100,name='pred')(x)

        softmax_output = Activation('softmax',name='softmax')(pred)
        model = Model(inputs=base_model.input, outputs=softmax_output)

else:
    
    model = keras.models.load_model(args.use_model)
    print "Loded Model",args.use_model
    
    if args.use_bg_cls and model.output_shape[1]!=101:
        # Adding additional node for the Background class to the fully connected layer
        weights = model.get_layer('pred').get_weights()
        # Adding the Bias Term
        weights[1] = np.concatenate((weights[1],np.zeros(1)),axis=0)
        # Adding the Weights and initializing them
        weights[0] = np.concatenate((weights[0],-0.0001*np.random.random_sample((weights[0].shape[0],1)) + 0.0001),axis=1)
        
        model.layers.pop()
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
        output = model.get_layer('fc').output
        #output = Flatten()(output)
        pred = Dense(101,name='pred')(output)
        softmax_output = Activation('softmax',name='softmax')(pred)
        model = Model(inputs=model.input, outputs=softmax_output)
        model.get_layer('pred').set_weights(weights)
    
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

print "RUNNING"
#for layer in model.layers[:1]:
#    layer.trainable = False
"""
for layer in model.layers:
    print layer
    layer.trainable = True
"""
"""
for layer in model.layers[:100]:
    layer.trainable = False
for layer in model.layers[100:]:
    layer.trainable = True
"""
# compile the model (should be done *after* setting layers to non-trainable)

global_loss = []
def ring_loss(y_true,y_pred):
    pred=K.sqrt(K.sum(K.square(y_pred),axis=1))
    error=K.mean(K.square(
#    error=(K.square(
        # Loss for Knowns having magnitude greater than knownsMinimumMag
        y_true[:,0]*(K.maximum(knownsMinimumMag-pred,0.))
        # Add two losses
        +
        # Loss for unKnowns having magnitude greater than unknownsMaximumMag
        y_true[:,1]*(K.maximum(pred-unknownsMaximumMag,0.))
    ))
#    global_loss.extend(error.tolist())
#    print "error",error.shape
    return error


def ard_mean_pred(y_true, y_pred):
    print "y_pred",y_pred.shape
    return y_pred


adam = Adam(lr=0.01)
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

    
if args.use_ring_loss:
    parallel_model.compile(
                            optimizer=adam,
                            loss={'softmax': 'categorical_crossentropy','fc':ring_loss},
                            loss_weights={'softmax': args.cross_entropy_loss_weight, 'fc': args.ring_loss_weight},
#                            metrics=[ard_mean_pred,'acc']
                            metrics=['acc']
                        )
    print "HERE"

else:
    parallel_model.compile(optimizer=adam, loss={'softmax':'categorical_crossentropy'},metrics=['acc'])



#common_name='Model_Epoch_number_{epoch:02d}.hdf5'
common_name='Model_Epoch_number_{}.hdf5'
model_file=''
if not args.include_known_unknowns:
    model_file='/home/adhamija/faceness/ImageNet/Pretrained_model/'+args.snapshot_location+common_name
elif args.use_bg_cls:
    model_file='/home/adhamija/faceness/ImageNet/BG_Cls_Data/'+args.snapshot_location+common_name
elif args.use_ring_loss:
#    model_file='Ring_Loss_Data/Model_Epoch_number_'+str(args.cross_entropy_loss_weight)+'_'+str(args.ring_loss_weight)+'_{epoch:02d}.hdf5'
#    model_file='Ring_Loss_Data/ALL_Layers_Model_Epoch_number_'+str(args.cross_entropy_loss_weight)+'_'+str(args.ring_loss_weight)+'_{epoch:02d}.hdf5'
    model_file='/home/adhamija/faceness/ImageNet/Ring_Loss_Data/'+args.snapshot_location+'Model_Epoch_number_'+str(args.cross_entropy_loss_weight)+'_'+str(args.ring_loss_weight)+'_{}.hdf5'
    
else:
    model_file='/home/adhamija/faceness/ImageNet/Cross_Entr_Data/'+args.snapshot_location+common_name


#model_file='JuNK_Model_Epoch_number_{epoch:02d}.hdf5'
    
    
class LossHistory(keras.callbacks.Callback):
    def __init__(self, model):
        self.model_to_save = model
        
    def on_train_begin(self, logs={}):
        self.losses = []
#        self.soft_loss = []

    def on_batch_end(self, batch, logs={}):
        pass
#        self.losses.append(logs.get('loss'))
#        self.soft_loss.extend([logs.get('softmax_ard_mean_pred').tolist()])

    def on_epoch_end(self, epoch, logs={}):
        self.model_to_save.save(model_file.format(epoch))

history = LossHistory(model)

checkpointer = ModelCheckpoint(
                                filepath=model_file, 
                                verbose=1, 
                                save_best_only=False , 
                                period=1, 
                                save_weights_only=False
                                )

info = parallel_model.fit_generator(
                            generator=training_generator,
                            validation_data=validation_generator,
                            max_queue_size=50,
                            epochs=args.no_of_epochs,
#                            callbacks=[checkpointer,history],
                            callbacks=[history],
#                            verbose=2,
                            **fit_generator_params
                            )

print len(history.losses)
print (history.losses)

print info
print info.history
print info.history.keys()

print history.soft_loss
print len(history.soft_loss)



print "global_loss",len(global_loss)
print global_loss
#parallel_model.save(model_file)