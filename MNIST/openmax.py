import sys
sys.path.insert(0, '../Tools')
import evaluation_tools as tools
import pandas as pd
import EVM
import libmr
import scipy
import functools
import cPickle
import numpy as np
from multiprocessing import Process,Pool,pool

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rerun", dest="rerun", action="store_true", default=False)
parser.add_argument("--run_on_logits", dest="run_on_logits", action="store_true", default=False)
parser.add_argument("--use_euclidean", dest="use_euclidean", action="store_true", default=False)
parser.add_argument("--DIR_file_name", dest="dir_file_name", action="store", default=None)#, required=True)
parser.add_argument("--tail_size", type=int, dest="tail_size", action="store", default=15)
args = parser.parse_args()

alpha=5

def cls_mav_dist(cls):
    data={}
    same_cls = feature_vector_file.iloc[results_file.index[results_file[1]==cls],1:]
    print "same_cls",same_cls.shape
    MAV=same_cls.mean()
    if args.use_euclidean:
        distances=same_cls.apply(lambda row:scipy.spatial.distance.euclidean(MAV, row),axis=1)
    else:
        distances=same_cls.apply(lambda row:scipy.spatial.distance.cosine(MAV, row),axis=1)
    distances=distances.values.tolist()
    
    print "len(distances)",len(distances)
    mr = libmr.MR()
    # Fitting an EVT on the distances
    mr.fit_high(distances,args.tail_size)
    data['model']=mr
    data['MAV']=MAV
    data['distances']=distances
    del distances,same_cls
    return data

#if args.rerun:
if True:
    
    feature_vector_file = pd.read_csv('Train_MNIST_openMax_deep_features.txt',
                                      delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)
    results_file = pd.read_csv('Train_MNIST_openMax_pred.txt',
                               delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)

    """
    feature_vector_file = pd.read_csv('OpenMax/train_deep_features.txt',
                                      delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)
    results_file = pd.read_csv('OpenMax/train_pred.txt',
                               delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)
    """
    print feature_vector_file.shape,results_file.shape
    print feature_vector_file,results_file
    
#    feature_vector_file = pd.read_csv('/home/adhamija/faceness/ImageNet/Pretrained_model/features_train_only_knowns.txt',
#                                      delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)
#    results_file = pd.read_csv('/home/adhamija/faceness/ImageNet/Pretrained_model/train_only_knowns.txt',
#                               delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)

    if args.run_on_logits:
#        results_file=np.array(results_file)
        print results_file.shape,results_file[:,1:].shape
        feature_vector_file=results_file[:,1:]
    """
    for a in range(100):
        t=cls_mav_dist(a)
    """
    p = Pool(48)
    MAV_models = p.map(cls_mav_dist,range(10))
    p.close()
    p.join()

    data = dict(zip(range(100),MAV_models))
#    cPickle.dump(data,open('/home/adhamija/faceness/ImageNet/Pretrained_model/Models_data','wb'))
    #print MAV_models
    
    del feature_vector_file
    del results_file

#else:
#    data = cPickle.load(open('/home/adhamija/faceness/ImageNet/Pretrained_model/Models_data','rb'))
    
    
def openmax_processing(row,ind=0):
    feature_to_process = row[1:].tolist()
#    print "feature_to_process",len(feature_to_process)
#    print row.name,row[0],results_file.iloc[row.name,0]
#    print row.name,results_file.shape
    raw_decision = results_file.iloc[row.name,2:].tolist()
#    raw_decision = results_file[results_file[0]==row[0]][2:]
#    unknown_score=0
#    print "raw_decision[0]",len(raw_decision),raw_decision[0]
    unknown_score=max(raw_decision[0],0)
    ranks = np.argsort(raw_decision)[::-1]
    weights = np.zeros(len(raw_decision))

    wscores=[]
    for i, r in enumerate(ranks):
        weights[r] = max((alpha-1.-i) / (alpha-1.), 0.)

        MAV = data[i]['MAV'].tolist()
        #print "MAV",MAV
        model = data[i]['model']
        #print "model",model
        if args.use_euclidean:
            distance = scipy.spatial.distance.euclidean(MAV, feature_to_process)
        else:
            distance = scipy.spatial.distance.cosine(MAV, feature_to_process)
        
        wscores.append(model.w_score(distance))
        
    weights = 1. - wscores * weights

    revised_decision = raw_decision * weights
    unknown_score += np.sum(raw_decision * (1.-weights))
    pred=[unknown_score]+revised_decision.tolist()

    softmax_scores=np.exp(pred)/np.sum(np.exp(pred))
    
    return softmax_scores.tolist()

def process_each_split(ind):
    return feature_vector_file.iloc[ind:ind+rows_per_proc].apply(openmax_processing,ind=ind,axis=1)
    

feature_vector_file = pd.read_csv('Test_MNIST_openMax_deep_features.txt',
                                  delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)
results_file = pd.read_csv('Test_MNIST_openMax_pred.txt',
                           delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)

"""
feature_vector_file = pd.read_csv('/home/adhamija/faceness/ImageNet/Pretrained_model/features_val_knowns_and_unknowns.txt',
                                  delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)
results_file = pd.read_csv('/home/adhamija/faceness/ImageNet/Pretrained_model/val_knowns_and_unknowns.txt',
                           delimiter=' ',header=None)#,memory_map=True)#, nrows=10000)
"""
if args.run_on_logits:
    feature_vector_file=results_file[:,1:]

print "feature_vector_file",feature_vector_file.shape
no_of_rows=feature_vector_file.shape[0]
rows_per_proc = no_of_rows/48
print range(0,no_of_rows,rows_per_proc)

p = Pool(48)
MAV_models = p.map(process_each_split,range(0,no_of_rows,rows_per_proc))
p.close()
p.join()

print "len(MAV_models)",len(MAV_models)

data_to_write=[]#pd.DataFrame()
for m in MAV_models:
    print m.shape
    data_to_write.extend(m.tolist())
data_to_write=np.array(data_to_write)
print data_to_write.shape

if args.use_euclidean:
#    tools.write_file_for_DIR(results_file.iloc[:,1],data_to_write[:,1:],args.dir_file_name+'/euclidean_'+str(args.tail_size)+'_openmax.txt')
    tools.write_file_for_DIR(
                                results_file.iloc[:,1],
                                data_to_write[:,1:],
                                ('OpenMax/{}_euclidean_'+str(args.tail_size)+'_openmax.txt').format('DIR_with_UK'),#args.dir_file_name),
                                feature_vector=None,num_of_known_classes=10
                            )
else:
    print results_file.iloc[:,1].shape,data_to_write[:,1:].shape
    tools.write_file_for_DIR(
                                results_file.iloc[:,1],
                                data_to_write[:,1:],
#                                args.dir_file_name+'/cosine_'+str(args.tail_size)+'_openmax.txt'
                                ('OpenMax/{}_cosine_'+str(args.tail_size)+'_openmax.txt').format('DIR_with_UK'),#(args.dir_file_name),
                                feature_vector=None,num_of_known_classes=10
                            )