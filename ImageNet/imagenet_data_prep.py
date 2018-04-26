import cv2
import numpy as np
import pandas as pd
import keras
import random
import multiprocessing
from multiprocessing import Process, Queue, Pool
from functools import partial
random.seed(0)

class imagenet_data_prep(keras.utils.Sequence):
    
    def __init__(self,
                 db_type='train',
                 split_no=0,
                 batch_size=64,
                 shuffle = False,
                 dataset_path = '',
                 protocol_file_path = '',
                 use_bg_cls = False,
                 include_known_unknowns = False,
                 include_unknowns = False,
                 training_data_obj=None,
                 knownsMinimumMag=None,
                 debug = False,
                 hard_negative_sample_ids=None,
                 no_of_known_unknown_classes = 450,
                 resize_images=False
                ):

        self.db_type = db_type
        self.batch_size = batch_size
        self.use_bg_cls = use_bg_cls
        self.shuffle = shuffle
        self.knownsMinimumMag = knownsMinimumMag
        self.resize_images = resize_images
        
        if db_type=='val' or db_type=='train':
            # Reading and grouping file
            csv_content = pd.read_csv(
                                        protocol_file_path.format('train'),
                                        delimiter=' ',header=None, lineterminator='\n'
                                        )
            self.dataset_path = dataset_path.format('train')
        else:
            # Reading and grouping file
            csv_content = pd.read_csv(
                                        protocol_file_path.format('val'),
                                        delimiter=' ',header=None, lineterminator='\n'
                                        )
            self.dataset_path = dataset_path.format('val')
            
        data_frame_group = csv_content.groupby([1])
        self.data_frame_group=data_frame_group

        ids=[]
        labels=[]
        sample_weights=[]

        # Find all Classes to be identified as Knowns, Known Unknowns and Unknowns
        if db_type=='train' or ((db_type=='val' or db_type=='test') and training_data_obj is None):
            raw_labels=data_frame_group.groups.keys()
            random.shuffle(raw_labels)
            self.known_classes = raw_labels[split_no:split_no+(len(raw_labels)/10)]
            self.knowns_class_mapping = dict(zip(self.known_classes,range(len(self.known_classes))))
            not_known_labels = list(set(raw_labels)-set(self.known_classes))
            self.known_unknown_classes = not_known_labels[:int(0.5*len(not_known_labels))]
            
            self.known_unknown_classes = self.known_unknown_classes[:no_of_known_unknown_classes]
            self.unknown_classes = not_known_labels[int(0.5*len(not_known_labels)):]
            training_data_obj=self

        self.n_classes = len(training_data_obj.known_classes)

        if debug:
            training_data_obj.known_unknown_classes = training_data_obj.known_unknown_classes[:3]
            training_data_obj.known_classes = training_data_obj.known_classes[:3]
            training_data_obj.unknown_classes = training_data_obj.unknown_classes[:3]
            self.n_classes = 100

        # Adding Known Unknown Samples
        if include_known_unknowns:
            no_of_known_unknowns=0
            for key in training_data_obj.known_unknown_classes:
                id_list = data_frame_group.get_group(key)[0].values.tolist()
                id_list = self.select_ids(id_list)
                ids.extend(id_list)
                no_of_known_unknowns+=len(id_list)
            labels.extend((np.ones(no_of_known_unknowns)*-1).tolist())
            sample_weights.extend((np.ones(no_of_known_unknowns)*(100./len(id_list))).tolist())


        if include_unknowns:
            no_of_unknowns=0
            for key in training_data_obj.unknown_classes:
                id_list = data_frame_group.get_group(key)[0].values.tolist()
                id_list = self.select_ids(id_list)
                ids.extend(id_list)
                no_of_unknowns+=len(id_list)
            labels.extend((np.ones(no_of_unknowns)*-1).tolist())
            sample_weights.extend((np.ones(no_of_unknowns)*(100./len(id_list))).tolist())

        # Adding Known Samples
        for key in training_data_obj.known_classes:
            id_list = data_frame_group.get_group(key)[0].values.tolist()
            id_list = self.select_ids(id_list)
            ids.extend(id_list)
            labels.extend((np.ones(len(id_list))*training_data_obj.knowns_class_mapping[key]).tolist())
            sample_weights.extend((np.ones(len(id_list))*(100./len(id_list))).tolist())

        self.list_IDs = ids
        self.labels = dict(zip(ids,labels))
        self.sample_weights = dict(zip(ids,sample_weights))
        
        self.on_epoch_end()



    def select_ids(self,id_list):
        if self.db_type=='train':
            return id_list[:-50]
        elif self.db_type=='val':
            return id_list[-50:]
        else:
            return id_list            
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.list_IDs)) / self.batch_size))

    
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, sample_weight = self.__data_generation(list_IDs_temp)
        
        return X, y, sample_weight

    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        print len(self.list_IDs),"..................................... REINITIATING ..............................."
        if self.shuffle == True:
            print "..................................... SHUFFLING ..............................."
            np.random.shuffle(self.indexes)

            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        sample_weights = []
        
        # Generate data
        for ID in list_IDs_temp:
            # Store sample
            img = cv2.imread(self.dataset_path+ID)
            if img is None:
                print "Didn't Find Image at : ",img,self.dataset_path+ID
                exit()
            if self.resize_images:
                img = cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
            X.append(img)
            
            # Store class
            y.append(self.labels[ID])
            sample_weights.append(self.sample_weights[ID])

        X = np.array(X)
        X = X/256.
        y = np.array(y)
        sample_weights = np.array(sample_weights)

        if self.use_bg_cls:
            y[y==-1] = self.n_classes
            Y = keras.utils.to_categorical(y, num_classes=(self.n_classes+1))
            return X, Y, sample_weights
        else:
            Y = np.zeros((y.shape[0],self.n_classes))
            Y[y!=-1] = keras.utils.to_categorical(y[y!=-1], num_classes=self.n_classes)
            Y[y==-1] = np.ones_like(self.n_classes)*(1./self.n_classes)
            if self.knownsMinimumMag is not None:
                flag = np.zeros((y.shape[0],2))
                flag[y!=-1,0] = 1
                flag[y==-1,1] = 1
                return (
                        # Inputs
                        [
                            X,
                            np.ones((sample_weights.shape[0]))*self.knownsMinimumMag
                        ],
                        # Labels
                        [
                            Y,
                            flag
                        ],
                        # Sample Weights
                        [
                            sample_weights,
                            sample_weights
                        ])
            else:
                return X, Y, sample_weights
