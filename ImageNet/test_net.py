import numpy as np
import sys
import argparse
import pandas as pd
import tools

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="store", dest="use_model", required=True)
parser.add_argument('--gpus', nargs='+', help='GPU No or Numbers for Multiple GPUs', required=True, type=int)
parser.add_argument("--split_no", action="store", dest="split_no", type=int, default = 0)
#parser.add_argument("--dont_include_known_unknowns", dest="include_known_unknowns", action="store_false", default=True)
#parser.add_argument("--include_unknowns", dest="include_unknowns", action="store_true", default=False)

parser.add_argument("--include_known_unknowns", dest="include_known_unknowns", action="store_true", default=False)
parser.add_argument("--dont_include_unknowns", dest="include_unknowns", action="store_false", default=True)
parser.add_argument('--no_of_known_unknown_classes', action="store", dest="no_of_known_unknown_classes", help='no_of_known_unknown_classes', type=int)

parser.add_argument("--run_prediction", dest="run_prediction", action="store_true", default=False)
#parser.add_argument("--DIR_file_name", dest="dir_file_name", action="store", default=None)
#parser.add_argument("--results_file_name", dest="results_file_name", action="store", default=None)
parser.add_argument("--run_for_db_type", dest="db_type", action="store", default='test')
parser.add_argument("--use_bg_cls", dest="use_bg_cls", action="store_true",default=False)
args = parser.parse_args()


# This Implementation is slightly slower and uses the keras tutorial approach    
from imagenet_data_prep import imagenet_data_prep

model_path='/'.join(args.use_model.split('/')[:-1])

validation_generator = imagenet_data_prep(
                                        db_type=args.db_type,
#                                        db_type='train',
#                                        db_type='val',
                                        batch_size=128,
                                        include_known_unknowns = args.include_known_unknowns,
                                        include_unknowns = args.include_unknowns,
                                        split_no=args.split_no,
                                        shuffle = False,
                                        use_bg_cls = args.use_bg_cls,
                                        no_of_known_unknown_classes=args.no_of_known_unknown_classes
                                        )





import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input
from keras import backend as K
from keras.optimizers import Adam


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


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(args.gpus[0])
set_session(tf.Session(config=config))

if 'ring' in args.use_model:
    unknownsMaximumMag = Input(shape=(1,), dtype='float32', name='unknownsMaximumMag')
    knownsMinimumMag = Input(shape=(1,), dtype='float32', name='knownsMinimumMag')
    model = keras.models.load_model(args.use_model, custom_objects={'ring_loss': ring_loss})
else:
    model = keras.models.load_model(args.use_model)

#def extract_features(model,validation_generator):
if args.run_prediction:
#    layer_names = ['softmax','fc']
    layer_names = ['softmax','fc','pred']
else:
    layer_names = ['softmax']

output_layers = [model.get_layer(layer_name).output for layer_name in layer_names]
intermediate_layer_model = Model(inputs=model.input, outputs=output_layers)

if args.run_prediction:
    intermediate_output = intermediate_layer_model.predict_generator(
                                                                        generator=validation_generator,
                                                                        max_queue_size=50,
                                                                        workers=10,
                                                                        use_multiprocessing=False
                                                                    )
    print intermediate_output
    for i in intermediate_output:
        print i.shape
        
        
    Ground_truth_labels = []
    for sample_id in validation_generator.list_IDs:
        Ground_truth_labels.append(validation_generator.labels[sample_id])
    Ground_truth_labels=np.array(Ground_truth_labels)

    print np.sum(intermediate_output[0],axis=1)
    print np.sum(intermediate_output[0],axis=1).shape,Ground_truth_labels.shape
    print "Ground_truth_labels",Ground_truth_labels

    if True:
#    if args.dir_file_name is not None:
        tools.write_file_for_DIR(
                                    gt_y = Ground_truth_labels,     # Ground Truth Labels
                                    pred_y = intermediate_output[0], # Output of 'pred' layer
                                    file_name = model_path+'/DIR_with_UK.txt', # args.dir_file_name,
        #                            feature_vector = intermediate_output[1] # Output of 'fc' layer
                                    )
        
    if True:
#    if args.results_file_name is not None:

        print np.arange(Ground_truth_labels.shape[0]).shape,Ground_truth_labels.shape,intermediate_output[2].shape
        stacked_file_data=np.concatenate((
                                            np.arange(Ground_truth_labels.shape[0])[:,np.newaxis].astype(np.int32),  # SAMPLE_IDENTIFIER
                                            Ground_truth_labels[:,np.newaxis].astype(np.int32),    # GT_CLASS_ID
                                            intermediate_output[2]
                                            ),axis=1)
        df=pd.DataFrame(stacked_file_data)
        df.to_csv(model_path+'/'+args.db_type+'_pred.txt', sep=' ', index=False, header=False)

        #np.savetxt(args.results_file_name, stacked_file_data, delimiter=' ')#,fmt='%f')
        
        stacked_file_data=np.concatenate((
                                            np.arange(Ground_truth_labels.shape[0])[:,np.newaxis].astype(np.int32),  # SAMPLE_IDENTIFIER
                                            intermediate_output[1]
                                            ),axis=1)
#        np.savetxt(('/').join(args.results_file_name.split('/')[:-1])+"/features_"+args.results_file_name.split('/')[-1], stacked_file_data, delimiter=' ')#,fmt='%f')
        df=pd.DataFrame(stacked_file_data)
        df.to_csv(model_path+'/'+args.db_type+'_deep_features.txt', sep=' ', index=False, header=False)
        
        print intermediate_output[1][Ground_truth_labels!=-1].shape,intermediate_output[1][Ground_truth_labels==-1].shape
        tools.plot_histogram(
                        intermediate_output[1][Ground_truth_labels!=-1],
                        intermediate_output[1][Ground_truth_labels==-1],
                        file_name=model_path+"/hist.pdf"
#                        file_name=('/').join(args.results_file_name.split('/')[:-1])+"/hist.pdf"
                      )
        
else:
#    intermediate_layer_model.compile(optimizer='adam', loss={'softmax':'categorical_crossentropy'},metrics=['categorical_accuracy'])
    intermediate_layer_model.compile(optimizer='adam', loss={'softmax':'categorical_crossentropy'},metrics=['acc'])
    intermediate_output = intermediate_layer_model.evaluate_generator(
                                                                        generator=validation_generator,
                                                                        max_queue_size=50,
                                                                        workers=10,
                                                                        use_multiprocessing=False
                                                                    )
    print intermediate_layer_model.metrics_names
    print intermediate_output
