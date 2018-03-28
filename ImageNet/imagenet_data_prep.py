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
                 dataset_path = '/net/kato/datasets/ImageNet/ILSVRC_2012/{}/',
                 protocol_file_path = '/net/ironman/scratch/adhamija/caffe/data/ilsvrc12/{}.txt',
                 use_bg_cls = False,
                 include_known_unknowns = False,
                 training_data_obj=None
                ):

        self.batch_size = batch_size
        self.dataset_path = dataset_path.format(db_type)
        self.use_bg_cls = use_bg_cls
        self.shuffle = shuffle,
        
        # Reading and grouping file
        csv_content = pd.read_csv(
                                    protocol_file_path.format(db_type),
                                    delimiter=' ',header=None, lineterminator='\n'
                                    )
        data_frame_group = csv_content.groupby([1])
        self.data_frame_group=data_frame_group

        ids=[]
        labels=[]
        sample_weights=[]

        if db_type=='train':
            raw_labels=data_frame_group.groups.keys()
            random.shuffle(raw_labels)
            self.known_classes = raw_labels[split_no:split_no+(len(raw_labels)/10)]
            self.knowns_class_mapping = dict(zip(self.known_classes,range(len(self.known_classes))))
            not_known_labels = list(set(raw_labels)-set(self.known_classes))
            self.known_unknown_classes = not_known_labels[:int(0.5*len(not_known_labels))]
            self.unknown_classes = not_known_labels[int(0.5*len(not_known_labels)):]
            training_data_obj=self
            self.shuffle = True

        self.n_classes = len(training_data_obj.known_classes)
        # Adding Known Unknowns
        if include_known_unknowns:
            no_of_known_unknowns=0
            for key in training_data_obj.known_unknown_classes:
                id_list = data_frame_group.get_group(key)[0].values.tolist()
                ids.extend(id_list)
                no_of_known_unknowns+=len(id_list)
            labels.extend((np.ones(no_of_known_unknowns)*-1).tolist())
            sample_weights.extend((np.ones(no_of_known_unknowns)*(100./len(id_list))).tolist())

        # Adding Known Samples
        for key in training_data_obj.known_classes:
            id_list = data_frame_group.get_group(key)[0].values.tolist()
            ids.extend(id_list)
            labels.extend((np.ones(len(id_list))*training_data_obj.knowns_class_mapping[key]).tolist())
            sample_weights.extend((np.ones(len(id_list))*(100./len(id_list))).tolist())

        self.list_IDs = ids
        self.labels = dict(zip(ids,labels))
        self.sample_weights = dict(zip(ids,sample_weights))
        
        self.on_epoch_end()


        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    
    
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
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)
        X = []
        y = []
        sample_weights = []
        
        # Generate data
        for ID in list_IDs_temp:
            # Store sample
            img = cv2.imread(self.dataset_path+ID)
            image = cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
            X.append(image)
            
            # Store class
            y.append(self.labels[ID])
            sample_weights.append(self.sample_weights[ID])

        X = np.array(X)
        y = np.array(y)
        sample_weights = np.array(sample_weights)

        if self.use_bg_cls:
            y[y==-1] = self.n_classes
            Y = keras.utils.to_categorical(y, num_classes=(self.n_classes+1))
        else:
            Y = np.zeros((y.shape[0],self.n_classes))
            Y[y!=-1] = keras.utils.to_categorical(y[y!=-1], num_classes=self.n_classes)
            Y[y==-1] = np.ones_like(self.n_classes)*(1./self.n_classes)
                                    
        return X, Y, sample_weights
    
    
    
    
    
prefetched_images = None
global_obj = None
class parallelized_imagenet_data_prep():
    
    def __init__(self,
                 db_type='train',
                 split_no=0,
                 batch_size=64,
                 dataset_path = '/net/kato/datasets/ImageNet/ILSVRC_2012/{}/',
                 protocol_file_path = '/net/ironman/scratch/adhamija/caffe/data/ilsvrc12/{}.txt',
                 use_bg_cls = False,
                 include_known_unknowns = False,
                 training_data_obj=None,
                 no_of_epochs=1
                ):

        self.batch_size = batch_size
        self.dataset_path = dataset_path.format(db_type)
        self.use_bg_cls = use_bg_cls
        
        # Reading and grouping file
        csv_content = pd.read_csv(
                                    protocol_file_path.format(db_type),
                                    delimiter=' ',header=None, lineterminator='\n'
                                    )
        data_frame_group = csv_content.groupby([1])
        self.data_frame_group=data_frame_group
        self.no_of_epochs = no_of_epochs

        ids=[]
        labels=[]
        sample_weights=[]

        if db_type=='train':
            raw_labels=data_frame_group.groups.keys()
            random.shuffle(raw_labels)
            self.known_classes = raw_labels[split_no:split_no+(len(raw_labels)/10)]
            self.knowns_class_mapping = dict(zip(self.known_classes,range(len(self.known_classes))))
            not_known_labels = list(set(raw_labels)-set(self.known_classes))
            self.known_unknown_classes = not_known_labels[:int(0.5*len(not_known_labels))]
            self.unknown_classes = not_known_labels[int(0.5*len(not_known_labels)):]
            training_data_obj=self

        self.n_classes = len(training_data_obj.known_classes)
        # Adding Known Unknowns
        if include_known_unknowns:
            no_of_known_unknowns=0
            for key in training_data_obj.known_unknown_classes:
                id_list = data_frame_group.get_group(key)[0].values.tolist()
                ids.extend(id_list)
                no_of_known_unknowns+=len(id_list)
            labels.extend((np.ones(no_of_known_unknowns)*-1).tolist())
            sample_weights.extend((np.ones(no_of_known_unknowns)*(100./len(id_list))).tolist())

        # Adding Known Samples
        for key in training_data_obj.known_classes:
            id_list = data_frame_group.get_group(key)[0].values.tolist()
            ids.extend(id_list)
            labels.extend((np.ones(len(id_list))*training_data_obj.knowns_class_mapping[key]).tolist())
            sample_weights.extend((np.ones(len(id_list))*(100./len(id_list))).tolist())

        self.list_IDs = ids
        self.labels = dict(zip(ids,labels))
        self.sample_weights = dict(zip(ids,sample_weights))
        
        global global_obj
        global_obj = self
        global prefetched_images
        prefetched_images = Queue(50*self.batch_size)
        
        
    def next(self):
        return get_batch()

    def get_no_of_batches(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

def get_batch():
#    main_prefetcher_process=Process(target=main_prefetcher)
#    main_prefetcher_process.start()
    i=0
#    if True:
#        for i in range(self.no_of_epochs):
    while True:
    #while data != 'END':
        batch=[]
        while len(batch)<global_obj.batch_size:
            #data = prefetched_images.get()
            data = process_each_sample(global_obj.list_IDs[i])
            print data
            i+=1
            batch.append(data)
#        else:
        X,y,sample_weights=zip(*batch)
        X=np.array(X)
        y=np.array(y)
        sample_weights=np.array(sample_weights)
        print X.shape,y.shape,sample_weights.shape
        yield (X,y,sample_weights)
        batch=[]
    main_prefetcher_process.join()

def main_prefetcher():
    random.shuffle(global_obj.list_IDs)
    p=Pool(multiprocessing.cpu_count()-3)
#    for i in range(global_obj.no_of_epochs):
    temp=p.map(partial(process_each_sample),global_obj.list_IDs)
#    prefetched_images.put('END')
    p.close()
    p.join()
    del p
    

def process_each_sample(ID):
    # Store sample
    img = cv2.imread(global_obj.dataset_path+ID)
    image = cv2.resize(img,(299,299),interpolation=cv2.INTER_CUBIC)
    y = global_obj.labels[ID]
    if global_obj.use_bg_cls:
        if y==-1:
            y = global_obj.n_classes
        Y = keras.utils.to_categorical(y, num_classes=(global_obj.n_classes+1))
    else:
        Y = np.zeros((global_obj.n_classes))
        if y == -1:
            Y = np.ones_like(global_obj.n_classes)*(1./global_obj.n_classes)
        else:
            Y = keras.utils.to_categorical(y, num_classes=global_obj.n_classes)

#    prefetched_images.put
    return (
                            image,
                            Y,
                            global_obj.sample_weights[ID]
                        )