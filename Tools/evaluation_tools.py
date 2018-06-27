import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool

def write_file_for_DIR(gt_y,pred_y,file_name,feature_vector=None,num_of_known_classes=100):
    """
    Writes a file for use by Bob's DIR Script.
    The format is:
    ############    'PREDICTED_CLASS_ID','PREDICTED_TEMPLATE_ID/Same as Predicted Class ID','GT_CLASS_ID','SAMPLE_IDENTIFIER','SIMILARITY_SCORE'
    'GT_CLASS_ID','GT_TEMPLATE_ID/Same as GT Class ID','PREDICTED_CLASS_ID','SAMPLE_IDENTIFIER','SIMILARITY_SCORE'
    
    The SIMILARITY SCORE is calculated as:
    When using the Unknown Class: The softmax score of the Class
    When using Cross Entropy: The softmax score of a Class Multiplied by the magnitude of the in lying feature vector.
    """
    
    if pred_y.shape[1]==num_of_known_classes+1:
        pred_y=pred_y[:,:-1]

    sample_identifier=np.tile(np.arange(gt_y.shape[0]),num_of_known_classes)
    gt_y=np.tile(gt_y,num_of_known_classes)
    predicted_class_id = np.repeat(np.arange(num_of_known_classes),pred_y.shape[0])

    similarity_score = pred_y.flatten('F')
    if feature_vector is not None:
        file_name = ('/').join(file_name.split('/')[:-1])+"/Multiplying_with_mag_"+file_name.split('/')[-1]
        print "file_name",file_name
        similarity_score=similarity_score*np.tile(np.sqrt(np.sum(np.square(feature_vector),axis=1)),num_of_known_classes)
        
    """
    stacked_file_data=np.stack((
                                gt_y,               # GT_CLASS_ID
                                gt_y,               # GT_CLASS_ID
                                predicted_class_id, # PREDICTED_CLASS_ID
                                sample_identifier,  # SAMPLE_IDENTIFIER
                                similarity_score    # SIMILARITY_SCORE
                                ),axis=1)

    """
    stacked_file_data=np.stack((
                                predicted_class_id, # PREDICTED_CLASS_ID
                                predicted_class_id, # PREDICTED_TEMPLATE_ID
                                gt_y,               # GT_CLASS_ID
                                sample_identifier,  # SAMPLE_IDENTIFIER
                                similarity_score    # SIMILARITY_SCORE
                                ),axis=1)
    np.savetxt(file_name, stacked_file_data, delimiter=' ',fmt=['%d','%d','%d','%d','%f'])


    
def pandas_process_each_file(file_name):
    csv_content = pd.read_csv(file_name,delimiter=' ',header=None, lineterminator='\n')
    data=[]
    for k,g in csv_content.groupby(3):
        data.append(g.loc[g[4].idxmax(),:].tolist())
    df = pd.DataFrame(data)
    df = df.sort_values(by=[4],ascending=False)
    positives=len(df[df[2]!=list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns=len(df[df[2]==list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns_label=list(set(df[2].tolist())-set(df[1].tolist()))[0]
    FP=[0]
    TP=[0]
    for ind,row in df.iterrows():
        # If Sample is Unknown
        if row[2]==unknowns_label:
            FP.append(FP[-1]+1)
            TP.append(df.loc[(df[2]==df[1]) & (df[4]>row[4])].shape[0])

    TP=np.array(TP[1:]).astype(np.float32)
    FP=np.array(FP[1:]).astype(np.float32)
    print file_name,TP[0],FP[0],TP[1],FP[1]
    return TP,FP,positives,unknowns


def process_each_file(file_name):
    csv_content = pd.read_csv(file_name,delimiter=' ',header=None, lineterminator='\n')
    data=[]
    for k,g in csv_content.groupby(3):
        data.append(g.loc[g[4].idxmax(),:].tolist())
    df = pd.DataFrame(data)
    df = df.sort_values(by=[4],ascending=False)
    positives=len(df[df[2]!=list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns=len(df[df[2]==list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns_label=list(set(df[2].tolist())-set(df[1].tolist()))[0]
    FP=[0]
    TP=[0]
    N=0
    N_above_UK=0
    UK_prob=1.
    for ind,row in df.iterrows():
        # If Sample is Unknown
        if row[2]==unknowns_label:
            UK_prob=row[4]
            FP.append(FP[-1]+1)
            TP.append(N)
        # If Sample is Known and Classified Correctly
        else:
            if row[1]==row[2]:
                N_above_UK+=1
                if row[4] < UK_prob:
                    N=N_above_UK
                    
    TP=np.array(TP[1:]).astype(np.float32)
    FP=np.array(FP[1:]).astype(np.float32)
#    print file_name,TP[0],FP[0],TP[1],FP[1]
    return TP,FP,positives,unknowns


def _process_each_file(file_name):
    csv_content = pd.read_csv(file_name,delimiter=' ',header=None, lineterminator='\n')
    data=[]
    for k,g in csv_content.groupby(3):
        data.append(g.loc[g[4].idxmax(),:].tolist())
    df = pd.DataFrame(data)
    df = df.sort_values(by=[4],ascending=False)
    positives=len(df[df[2]!=list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns=len(df[df[2]==list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns_label=list(set(df[2].tolist())-set(df[1].tolist()))[0]
    knowns_prob=1.
    unknowns_prob=1.
    FP=[]
    TP=[]
    FP_=0
    TP_=0
    for ind,row in df.iterrows():
        if knowns_prob<unknowns_prob:
            if len(FP)==1:
                FP.append(FP_)
                TP.append(0)
            FP.append(FP_)
            TP.append(TP_)
        # If Sample is Unknown
        if row[2]==unknowns_label:
            FP_+=1
            unknowns_prob=row[4]
        # If Sample is Known and Classified Correctly
        elif row[1]==row[2]:
            TP_+=1
            knowns_prob=row[4]

    FP.append(FP_)
    TP.append(TP_)
    TP=np.array(TP).astype(np.float32)
    FP=np.array(FP).astype(np.float32)
    return TP,FP,positives,unknowns

def old_process_each_file(file_name):
    csv_content = pd.read_csv(file_name,delimiter=' ',header=None, lineterminator='\n')
    data=[]
    for k,g in csv_content.groupby(3):
        data.append(g.loc[g[4].idxmax(),:].tolist())
    df = pd.DataFrame(data)
    df = df.sort_values(by=[4],ascending=False)
    positives=len(df[df[2]!=list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns=len(df[df[2]==list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns_label=list(set(df[2].tolist())-set(df[1].tolist()))[0]
    FP=[0]
    TP=[0]
    for ind,row in df.iterrows():
#        if row[2]==-1:
        if row[2]==unknowns_label:
            FP.append(FP[-1]+1)
            TP.append(TP[-1])
        elif row[1]==row[2]:
            TP[-1]+=1
    TP=np.array(TP).astype(np.float32)
    FP=np.array(FP).astype(np.float32)
    return TP,FP,positives,unknowns


def create_deviation_plot(files_to_process,labels,DIR_filename=None):
    p=Pool(len(files_to_process)*len(files_to_process[0]))
    f=[]
    [f.extend(files) for files in files_to_process]
    to_plot=p.map(process_each_file,f)
    p.close()
    p.join()
    
    consolidated_mean=[]
    consolidated_std=[]
    temp_TP=[]
    for i,(TP,FP,positives,unknowns) in enumerate(to_plot):
        temp_TP.append(TP/positives)
        print f[i],len(files_to_process),(i+1)%len(files_to_process)
        if (i+1)%(len(f)/len(files_to_process)) == 0:
            temp_TP=np.array(temp_TP)
            consolidated_mean.append(np.mean(temp_TP, axis=0))
            print consolidated_mean[-1]
            consolidated_std.append(np.std(temp_TP, axis=0))
            temp_TP=[]
    FP=FP/len(FP)
#    temp_TP=np.array(temp_TP)
#    consolidated_mean.append(np.mean(temp_TP, axis=0))
#    consolidated_std.append(np.std(temp_TP, axis=0))
    
    print "Plotting"
    fig, ax = plt.subplots()
    for i,TP in enumerate(consolidated_mean):
        print i, len(FP), len(TP), labels[i]
#        ax.errorbar(FP, TP, fmt='-o',label=labels[i])
        ax.errorbar(FP, TP, yerr=consolidated_std[i], fmt='-o',label=labels[i])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xscale('log')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([0,1])
    ax.set_ylabel('Correct Classification Rate', fontsize=18, labelpad=10)
    ax.set_xlabel('False Positive Rate : Total Unknowns '+str(len(FP)), fontsize=18, labelpad=10)
#    fig.suptitle('OpenSet Recognition', fontsize=12)
#    ax.set_title('Knowns:MNIST KnownUnknowns:Letters Unknowns:Devanagari', fontsize=10)
    ax.legend(loc="upper right")
    #ax.legend(loc="best")
    
#    ax.legend(loc='lower center',bbox_to_anchor=(0.5, -0.75),ncol=2,fontsize=18,frameon=False)
    if DIR_filename is not None:
        fig.savefig(DIR_filename+'.png', bbox_inches="tight",dpi=80)
    plt.show()


def process_files(files_to_process,labels,DIR_filename=None,out_of_plot=False):
    p=Pool(len(files_to_process))
    to_plot=p.map(process_each_file,files_to_process)
    p.close()
    p.join()
    """
    to_plot=[]
    for f in files_to_process:
        to_plot.append(process_each_file(f))
    """
    print "Plotting"
#    matplotlib.rcParams.update({'font.size': 16})
    u = []
    fig, ax = plt.subplots()
    for i,(TP,FP,positives,unknowns) in enumerate(to_plot):
        ax.plot(FP/unknowns,TP/positives,label=labels[i])
        print labels[i],
        for i in np.arange(1,5,1)[::-1]:
#            print np.around(FP/unknowns,decimals=i)==1./(10**(i)),TP/positives
            print np.round(np.mean((TP/positives)[np.around(FP/unknowns,decimals=i)==1./(10**(i))]),decimals=4),'&',
        print ''
        u.append(unknowns)
#    ax.plot(FP/unknowns,np.zeros(FP.shape[0]),'--',label='Squared MLP with Ensemble')
    ax.set_xscale('log')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([0,1])
    ax.set_ylabel('Correct Classification Rate', fontsize=18, labelpad=10)
    ax.set_xlabel('False Positive Rate : Total Unknowns '+str(list(set(u))[0]), fontsize=18, labelpad=10)
#    fig.suptitle('OpenSet Recognition', fontsize=12)
#    ax.set_title('Knowns:MNIST KnownUnknowns:Letters Unknowns:Devanagari', fontsize=10)

    if out_of_plot:
#    matplotlib.rcParams.update({'font.size': 16})
        ax.legend(loc='lower center',bbox_to_anchor=(-0.75, 0.),ncol=1,fontsize=18,frameon=False)
#    ax.legend(loc='center',bbox_to_anchor=(1.5, 0.5),ncol=2,fontsize=18,frameon=False)
    #ax.legend(loc="best")
    else:
        ax.legend(loc="upper left")
        
    if DIR_filename is not None:
        print DIR_filename+'.pdf'
        fig.savefig(DIR_filename+'.pdf', bbox_inches="tight")
    plt.show()


    
    
    
    
def ensemble_process_each_file(file_name):
    csv_content = pd.read_csv(file_name,delimiter=' ',header=None, lineterminator='\n')
    data=[]
    for k,g in csv_content.groupby(3):
        data.append(g.loc[g[4].idxmax(),:].tolist())
    df = pd.DataFrame(data)
    df = df.sort_values(by=[4],ascending=False)
    positives=len(df[df[2]!=list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns=len(df[df[2]==list(set(df[2].tolist())-set(df[1].tolist()))[0]])
    unknowns_label=list(set(df[2].tolist())-set(df[1].tolist()))[0]
    FP=[0]
    TP=[0]
    Thresholds=[]
    N=0
    N_above_UK=0
    UK_prob=1.
    for ind,row in df.iterrows():
        # If Sample is Unknown
        if row[2]==unknowns_label:
            UK_prob=row[4]
            FP.append(FP[-1]+1)
            TP.append(N)
            Thresholds.append(row[4])
        # If Sample is Known and Classified Correctly
        else:
            if row[1]==row[2]:
                N_above_UK+=1
                if row[4] < UK_prob:
                    N=N_above_UK
                    
    TP=np.array(TP[1:]).astype(np.float32)
    FP=np.array(FP[1:]).astype(np.float32)
    Thresholds=np.array(Thresholds).astype(np.float32)
#    print file_name,TP[0],FP[0],TP[1],FP[1]
    return TP,FP,Thresholds,positives,unknowns,df.shape[0]

    
def ensemble_process_files(files_to_process,labels,DIR_filename=None):
    p=Pool(len(files_to_process))
    to_plot=p.map(ensemble_process_each_file,files_to_process)
    p.close()
    p.join()
    """
    to_plot=[]
    for f in files_to_process:
        to_plot.append(process_each_file(f))
    """

    print "Plotting"
#    matplotlib.rcParams.update({'font.size': 16})
    min_y=1.
    u = []
    fig, ax = plt.subplots()
    for i,(TP,FP,Thresholds,positives,unknowns,size) in enumerate(to_plot):
        X=np.divide(TP,(TP+FP))[::-1]
        ax.plot(Thresholds[::-1],X,label=labels[i])
        u.append(unknowns)
        min_y=min(min_y,np.min(X[X>0]))

    ax.plot(np.square(Thresholds[::-1]),X,'--',label='Squared MLP with Ensemble')
    u.append(unknowns)
    min_y=min(min_y,np.min(X[X>0]))
    
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([min_y,1.])
#    ax.set_xlim([0.,0.9])
    ax.set_ylabel('Precision', fontsize=18, labelpad=10)
    ax.set_xlabel('Confidence Threshold', fontsize=18, labelpad=10)
    #ax.legend(loc="lower right")
#    ax.legend(loc="upper right")
    #ax.legend(loc="best")
    if DIR_filename is not None:
        fig.savefig(DIR_filename+'.pdf', bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--files_to_process', nargs='+', help='DIR file Names', required=True)
    parser.add_argument("--labels", nargs='+', help='DIR file Labels', required=True)
    parser.add_argument("--DIR_filename", help='Name of DIR filename to create', required=True)
    args = parser.parse_args()
    
    if len(args.files_to_process) != len(args.labels):
        print "Please provide a label for each file name ... Exiting!!!"
        exit()
        
    process_files(args.files_to_process,args.labels,args.DIR_filename)
    """
    To_process=[
    #                    '/home/adhamija/faceness/ImageNet/Pretrained_model/Pre_Adam_1e-3/DIR_with_UK.txt'
                        ('/home/adhamija/faceness/MNIST/Hindi/Vanilla.txt','SoftMax'),
                        ('/home/adhamija/faceness/MNIST/Hindi/BG.txt','BG'),
                        ('/home/adhamija/faceness/MNIST/Hindi/Cross.txt','Entropic Open-set'),
                        ('/home/adhamija/faceness/MNIST/Hindi/Multiplying_with_mag_Cross.txt','Mag Entropic Open-set'),
                        ('/home/adhamija/faceness/MNIST/Hindi/Ring.txt','Objectosphere'),
                        ('/home/adhamija/faceness/MNIST/Hindi/Multiplying_with_mag_Ring.txt','Mag Objectosphere'),
                        ('/home/adhamija/faceness/MNIST/Hindi/OnlyRing.txt','Only_Ring'),
                        ('/home/adhamija/faceness/MNIST/Hindi/Multiplying_with_mag_OnlyRing.txt','Mag_Only_Ring')
                    ]

    
    files_to_process,labels=zip(*To_process)

    """
