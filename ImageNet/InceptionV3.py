import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test_network", dest="test_network", action="store_true")
parser.add_argument("--use_bg_cls", dest="use_bg_cls", action="store_true")
parser.add_argument("--dont_include_known_unknowns", dest="include_known_unknowns", action="store_false", default=True)
parser.add_argument("--split_no", action="store", dest="split_no", type=int, default = 0)
parser.add_argument("--model", action="store", dest="use_model", default = None)
parser.add_argument('--gpus', nargs='+', help='GPU No or Numbers for Multiple GPUs', required=True, type=int)
args = parser.parse_args()

print "split_no",args.split_no,args.use_bg_cls,"args.gpus",args.gpus,len(args.gpus)


parallel=False

if parallel:
    from imagenet_data_prep import parallelized_imagenet_data_prep    
    training_generator = parallelized_imagenet_data_prep(
                                            db_type='train',
                                            batch_size=128,
                                            include_known_unknowns = args.include_known_unknowns,
                                            use_bg_cls=args.use_bg_cls,
                                            split_no=args.split_no,
                                            )

    validation_generator = parallelized_imagenet_data_prep(
                                            db_type='val',
                                            batch_size=128,
                                            training_data_obj=training_generator,
                                            use_bg_cls=args.use_bg_cls,
                                            split_no=args.split_no
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
                                            batch_size=64,
                                            include_known_unknowns = args.include_known_unknowns,
                                            use_bg_cls=args.use_bg_cls,
                                            split_no=args.split_no
                                            )

    validation_generator = imagenet_data_prep(
                                            db_type='val',
                                            batch_size=64,
                                            training_data_obj=training_generator,
                                            use_bg_cls=args.use_bg_cls,
                                            split_no=args.split_no
                                            )
    fit_generator_params = dict(
                                workers=100,
                                use_multiprocessing=True
                               )
    

import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import backend as K
from keras.optimizers import Adam


from keras.backend.tensorflow_backend import set_session
from keras.utils import multi_gpu_model
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


if args.use_model is None:
    #tf.reset_default_graph()
    with tf.device('/cpu:0'):
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024,name='fc')(x)

        if args.use_bg_cls:
            pred = Dense(101, activation='softmax',name='pred')(x)
        else:
            pred = Dense(100, activation='softmax',name='pred')(x)

        model = Model(inputs=base_model.input, outputs=pred)

else:
    model = keras.models.load_model(args.use_model)
    print ".....................DONE........................"
    if args.use_bg_cls and model.output_shape[1]!=101:
        model.layers.pop()
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
        output = model.get_layer('fc').output
        #output = Flatten()(output)
        #pred = Dense(101, activation='softmax',name='pred')(output)
        pred = Dense(101, activation='softmax',name='pred')(output)
        model = Model(inputs=model.input, outputs=pred)
        print ".....................DONE........................123"
    
if len(args.gpus)>1:
    parallel_model = multi_gpu_model(model, gpus=args.gpus)
else:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(args.gpus[0])
    set_session(tf.Session(config=config))
    parallel_model = model


for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

#for layer in base_model.layers:
#    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)

adam = Adam(lr=0.001)
parallel_model.compile(optimizer=adam, loss={'pred':'categorical_crossentropy'},metrics=['acc'])


# train the model on the new data for a few epochs
#model.fit_generator(...)
keras.backend.get_session().run(tf.global_variables_initializer())
parallel_model.fit_generator(
                            generator=training_generator,
                            validation_data=validation_generator,
                            max_queue_size=50,
                            epochs=10,
#                            verbose=2,
                            **fit_generator_params
                            )


if not args.include_known_unknowns:
    model.save('Pretrained_model/Finetuned_Model_Split_'+str(args.split_no))
else:
    if args.use_bg_cls:
        model.save('BG_Cls_Data/Finetuned_Model_Split_'+str(args.split_no))
    else:
        model.save('Cross_Entr_Data/Finetuned_Model_Split_'+str(args.split_no))
