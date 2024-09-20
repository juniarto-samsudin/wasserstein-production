import numpy as np
import ot
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
from otdd.pytorch.moments import *
from otdd.pytorch.utils import *

import random
import time
import pickle
import zipfile
import requests
import argparse
import logging
import importlib

np.random.seed(42)

# Get the script's filename without the extension
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Generate the log filename
log_filename = '{}.log'.format(script_name)

logging.basicConfig(filename=("./logs/{}".format(log_filename)), 
                    level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')


parser = argparse.ArgumentParser(description="Measure the Wasserstein distance between two datasets.")
parser.add_argument('--datasetpath', type=str, help='The path to dataset to measure', default='/media/samsudinj/myextssd/WORK/Wasserstein/OCT/OCT1')
parser.add_argument('--referencepath', type=str, help='The path to reference dataset', default='/media/samsudinj/myextssd/WORK/Wasserstein/OCT/test')
parser.add_argument('--dataset', type=str, help='OCT, etc', default='OCT')
args = parser.parse_args()

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
    logging.info("Using GPU")
else:
    device = torch.device("cpu")  # Use CPU
    logging.info("Using CPU")

def euclidean_dist_torch(x1, x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1, x2.t())
    distance = x1p.expand_as(prod_x1x2) + \
        x2p.t().expand_as(prod_x1x2) - 2*prod_x1x2
    return torch.sqrt(distance)  # /x1.size(0)/x2.size(0)


def get_interp_measure(xs,xt,G0,t,thresh=1e-5):
    """ Get an exact interpolating measure between xs, xt 
    given the transport plan G0 and $t$.

    Args:
        xs (array): _description_
        xt (array): _description_
        G0 (_type_): _description_
        t (scalar real): _description_

    Returns:
        _type_: _description_
    """
    n_s, dim = xs.shape
    n_t = xt.shape[0]
    xsp = np.zeros((n_s+n_t+1, dim))
    xtp = np.zeros((n_s+n_t+1, dim))
    weights = np.zeros((n_s+n_t+1,))
    k = 0
    for i in range(xs.shape[0]):
        ind = np.where(G0[i, :]>thresh)[0]
        for j in range(len(ind)):
            xsp[k,:] = xs[i, :]
            xtp[k,:] = xt[ind[j], :]
            weights[k] = G0[i,ind[j]]
            k += 1
    # if k > n_s:
    #     print(k, n_s)
    #     pass
    xsp = xsp[:k, :]
    xtp = xtp[:k, :]
    xz = (1-t)*xsp + t*xtp
    weights = weights[:k]/np.sum(weights[:k])
    #print(xz.shape, weights.shape)
    
    return xz, weights

def interp_meas(X,Y,t_val=None,metric='sqeuclidean',approx_interp=True,
                a = None,b = None):
    """
    compute an the OT plan, cost and an interpolating measure
    works for squared euclidean distance
    everything is done on numpy
    
    return 
        * the interpolating measure
        * the OT cost between X and Y
        * the transport plan
    """
    nx, ny  = X.shape[0], Y.shape[0]
    p = 2 if metric=='sqeuclidean' else 1  
    if a is None:  
        a = np.ones((nx,),dtype=np.float64) / nx
    if b is None:
        b = np.ones((ny,),dtype=np.float64) / ny  
    # loss matrix
    M = ot.dist(X,Y,metric=metric) # squared euclidean distance 'default'
    # compute EMD
    norm = np.max(M) if np.max(M)>1 else 1
    G0 = ot.emd(a, b, M/norm)
    
    
    t = np.random.rand(1) if t_val==None else t_val
    #print('t',t)
    if approx_interp:
        Z = (1-t)*X + t*(G0*nx)@Y
        weights =  np.ones((nx,),dtype=np.float64) / nx
    else:
        Z, weights = get_interp_measure(X,Y,G0,t)
    cost = np.sum(G0*M)**(1/p)
    return Z, weights, cost, G0



    
def learn_interp_meas_support(xs,xt,n_supp=100,n_epoch=100,
                                t_val = None, lr= 0.01,p=2,
                                z_init=None, verbose=False,
                                a = None, b = None):
    """
    xs and xt are supposed to be numpy arrays
    
    p = 2 squared euclidean distance
    p = 1 euclidean distance
    
    output are numpy arrays 
    """
    if t_val is None:
        t_val = np.random.rand(1)[0] if t_val==None else t_val
    # TODO: add numpy transformation of xs and xt
    
    dim = xs.shape[1]
    c = np.ones(n_supp)/n_supp
    z = nn.Embedding(n_supp, dim)
    if z_init is not None:
        z.weight.data = torch.from_numpy(z_init)
    else:
        z.weight.data = torch.ones(n_supp, dim)
    z_init = z.weight.detach().clone()
    ns = xs.shape[0]
    nt = xt.shape[0]
    if a is None:  
        a = np.ones((ns,),dtype=np.float64) / ns
    if b is None:
        b = np.ones((nt,),dtype=np.float64) / nt 
    optimizer = optim.Adam(z.parameters(), lr=lr)
    s_list = []
    #print('learn',t_val)
    for i in range(n_epoch):
        # computing distance matrices 
        # between samples and interpolating measure

        Ms = euclidean_dist_torch(torch.from_numpy(xs).double(), z.weight.double()).pow(p)
        Mt = euclidean_dist_torch( z.weight.double(), torch.from_numpy(xt).double()).pow(p)
        with torch.no_grad():
            Ms_aux =  Ms.detach().data.numpy()
            Mt_aux =  Mt.detach().data.numpy()
            normMs = np.max(Ms_aux) if np.max(Ms_aux)>1 else 1
            normMt = np.max(Mt_aux) if np.max(Mt_aux)>1 else 1

            gamma_s = ot.emd(a, c, Ms_aux/normMs)
            gamma_s = torch.from_numpy(gamma_s)
            gamma_t = ot.emd(c,b, Mt_aux/normMt)
            gamma_t = torch.from_numpy(gamma_t)
        S = (1-t_val)*(torch.sum(Ms*gamma_s)).pow(1/p) + t_val*(torch.sum(Mt*gamma_t)).pow(1/p)
        z.zero_grad()
        S.backward()
        s_list.append(S.item())
        optimizer.step()
    cost = (torch.sum(Ms*gamma_s)).pow(1/p) + (torch.sum(Mt*gamma_t)).pow(1/p)
    z = z.weight.detach().numpy()
    # TODO: change plan to the full plan from X to Y
    return z, cost.detach().item(), [gamma_s,gamma_t], s_list



class InterpMeas:
    def __init__(self,metric='sqeuclidean',t_val=None,approx_interp=True,
                 learn_support=False):
        self.metric = metric
        self.t_val = t_val
        self.n_supp = 100
        self.approx_interp = approx_interp

        #-- useful for learning support
        self.lr = 0.01
        self.n_epoch = 100
        self.int_init = None
        self.learn_support = learn_support
    def fit(self,X,Y, a=None, b=None):
        """_summary_

        Args:
            X (np_array): size nx x dim
            Y (np_array): _description_
            a (np_array, optional): _weights of the empirical distribution X . Defaults to None with equal weights.
            b (np_array, optional): _weights of the empirical distribution X . Defaults to None with equal weights.
            
        Returns:
            An InterpMeas object with the following attributes:
            int_m (np_array): size n_supp x dim
            weights (np_array): size n_supp x 1
            plan (np_array): size nx x nt
            loss_learn (list): list of the loss function during the learning of the support
            cost (float): cost of the optimal transport plan
        """
        t = np.random.rand(1)[0] if self.t_val==None else self.t_val
        #logging.info('inside InterpMeas Fit t= {}'.format(t)) # 0.5
        if not self.learn_support: # if we don't learn the support, use this
            #logging.info('inside InterpMeas Fit() not learn_support')
            Z, weights, cost, G0 = interp_meas(X,Y,t_val=t,metric=self.metric,
                                               a=a,b=b,approx_interp=self.approx_interp)
            self.t = t
            self.int_m = Z 
            self.cost = cost
            self.plan = G0
            self.weights = weights
        elif self.learn_support:
            t = np.random.rand(1)[0] if self.t_val==None else self.t_val
            p = 2 if self.metric=='sqeuclidean' else 1    
            Z, cost, gamma, s_list = learn_interp_meas_support(X,Y,n_supp=self.n_supp,n_epoch=self.n_epoch,
                                t_val = t, lr= self.lr, p=p,
                                z_init= self.int_init,
                                a=a, b = b)
            self.int_m = Z
            self.weights = np.ones((Z.shape[0],),dtype=np.float64) / Z.shape[0]
            self.cost =  cost
            # TODO: change plan to the full plan from X to Y
            self.plan = gamma
            self.loss_learn = s_list
        
        
        
        return self




class FedOT:
    def __init__(self, n_supp,n_epoch, t_val=None,verbose=False,
                 get_int_list=False,
                 metric = 'sqeuclidean'):
        self.n_supp = n_supp  # n_supp of the interpolating measure
        self.n_epoch = n_epoch
        self.t_val = t_val
        self.verbose = verbose
        self.get_int_list = get_int_list
        self.metric = metric
        self.random_val_init = 1
        if self.metric == 'sqeuclidean':
            self.p=2
        elif self.metric == 'euclidean':
            self.p=1 

    def fit(self,xs, xt, ws = None, wt = None,approx_interp=True,
            learn_support=False):   
        self.approx_interp = approx_interp
        self.learn_support = learn_support
        dim = xs.shape[1]
        cost_diff = 0
        istensor = False

        # xs A ,  xt B , server C
        if type(xs) == torch.Tensor:
            xs_ = torch.clone(xs) 
            xs= xs.detach().numpy() # (100,2000)
            logging.info('FedOT fit xs shape: {}'.format(xs.shape)) # (100, 9216)
            xt_ = torch.clone(xt) 
            xt= xt.detach().numpy() #(300,2000)
            logging.info('FedOT fit xt shape: {}'.format(xt.shape)) # (100, 9216)
            istensor = True
            if ws is not None:
                logging.info('FedOT ws is not NONE')
                ws = ws.numpy().astype(np.float64)
            else :
                ws = np.ones((xs.shape[0],),dtype=np.float64) / xs.shape[0]
                logging.info('FedOT ws is NONE')
                logging.info('FedOT fit ws shape: {}'.format(ws.shape)) # (100,)
                #logging.info('ws: {}'.format(ws))
            if wt is not None:
                logging.info('FedOT wt is not NONE')
                wt = wt.numpy().astype(np.float64)
            else :
                wt = np.ones((xt.shape[0],),dtype=np.float64) / xt.shape[0]
                logging.info('FedOT fit wt shape: {}'.format(wt.shape)) # (100,)
                #logging.info('FedOT fit wt: {}'.format(wt))
        # creating object for interpolation
        interp_G = InterpMeas(metric=self.metric,t_val=self.t_val,approx_interp=approx_interp,
                              learn_support=self.learn_support)
        interp_H = InterpMeas(metric=self.metric,t_val=self.t_val,approx_interp=approx_interp,
                              learn_support=self.learn_support)
        interp_m = InterpMeas(metric=self.metric,t_val=self.t_val,approx_interp=approx_interp,
                              learn_support=self.learn_support)

        int_m = np.random.randn(self.n_supp,dim)*self.random_val_init  # (100,9216)
        logging.info('FedOT fit int_m shape: {}'.format(int_m.shape)) # (100, 9216)
        #logging.info('int_m: {}'.format(int_m)) # random value of int_m
        weight_int_m = np.ones(self.n_supp)/self.n_supp
        logging.info('FedOT weight_int_m shape: {}'.format(weight_int_m.shape)) # (100,)
        logging.info('weight_int_m: {}'.format(weight_int_m))# [0.01, 0.01,..]

        list_cost = []
        list_int_m = []
        list_int_G = []
        list_int_H = []
        
        for i in range(self.n_epoch):
            if self.verbose:
                print(i)
            if self.get_int_list:
                list_int_m.append(int_m)

            # xs-- G.int_m --int_m  -- H.int_m -- xt

            # on client S
            logging.info('FedOT fit for dataset i: {}'.format(i))
            interp_G.fit(int_m,xs,a=weight_int_m, b=ws)
            G, weight_G, cost_g= interp_G.int_m, interp_G.weights, interp_G.cost
            interp_G.int_init = G
            # on client T
            logging.info('FedOT fit for reference i: {}'.format(i))
            interp_H.fit(int_m,xt, a=weight_int_m, b=wt)
            H, weight_H, cost_h = interp_H.int_m, interp_H.weights, interp_H.cost
            interp_H.int_init = H
            # send costs, G and H to the server
            # on server
            list_cost.append( cost_g+ cost_h)
            logging.info('FedOT fit for server i: {}'.format(i))
            interp_m = interp_m.fit(H, G,a=weight_H,b=weight_G)
            int_m, weight_int_m = interp_m.int_m, interp_m.weights
            interp_m.int_init = int_m.copy()
            if self.get_int_list:
                list_int_G.append(G)
                list_int_H.append(H)
        # preparing output for differentiable cost
        if istensor:
            eps = 1e-6

            Ms = euclidean_dist_torch(xs_.double(), torch.from_numpy(int_m).double()).pow(self.p)
            Mt = euclidean_dist_torch(torch.from_numpy(int_m).double(), xt_.double()).pow(self.p)
         
            
            with torch.no_grad():
                ns, nt = xs_.shape[0], xt_.shape[0]
                nm = int_m.shape[0]
                c = weight_int_m
                Ms_aux =  Ms.detach().data.numpy()
                Mt_aux =  Mt.detach().data.numpy()
                normMs = np.max(Ms_aux) if np.max(Ms_aux)>1 else 1
                normMt = np.max(Mt_aux) if np.max(Mt_aux)>1 else 1
                #print(np.sum(a),np.sum(b),np.sum(c),)
                gamma_s = ot.emd(ws, c, Ms_aux/normMs)
                planS = torch.from_numpy(gamma_s)
                gamma_t = ot.emd(c,wt, Mt_aux/normMt)
                planT = torch.from_numpy(gamma_t)
            cost = (torch.sum(Ms*planS)+ eps)**(1/self.p) + \
                         (torch.sum(Mt*planT)+ eps)**(1/self.p) 
        else:
            nt = xt.shape[0]
            interp_G.fit(xs,int_m)
            G, weight_G, cost_g, planS = interp_G.int_m, interp_G.weights, interp_G.cost, interp_G.plan
            interp_H.fit(int_m,xt)
            H, weight_G, cost_h, planT = interp_H.int_m, interp_H.weights, interp_H.cost, interp_H.plan
            cost = cost_g + cost_h

        
        self.int_meas = int_m
        self.weights = weight_int_m
        self.list_cost = list_cost
        self.cost = cost 
        self.cost_g = cost_g
        self.cost_h = cost_h
        self.planS, self.planT = planS, planT
        self.plan = planS@planT*nt
        self.list_int_meas = list_int_m
        self.list_int_G = list_int_G
        self.list_int_H = list_int_H

        return self

class direct_learn:
    def __init__(self,metric='sqeuclidean',t_val=None,approx_interp=True,
                 learn_support=False):
        self.metric = metric
        self.t_val = t_val
        self.n_supp = 100
        self.approx_interp = approx_interp

        #-- useful for learning support
        self.lr = 0.01
        self.n_epoch = 100
        self.int_init = None
        self.learn_support = learn_support
        if self.metric == 'sqeuclidean':
            self.p=2
        elif self.metric == 'euclidean':
            self.p=1 

    def fit_direct(self,xs, xt, ws = None,wt = None,approx_interp=True,
                learn_support=False):   
            self.approx_interp = approx_interp
            self.learn_support = learn_support


            istensor = False
    

            if type(xs) == torch.Tensor:
                xs_ = torch.clone(xs) 
                xs= xs.detach().numpy() # (100,2000)
                xt_ = torch.clone(xt) 
                xt= xt.detach().numpy() #(300,2000)
                istensor = True
                if ws is not None:
                    ws = ws.numpy().astype(np.float64)
                else :
                    ws = np.ones((xs.shape[0],),dtype=np.float64) / xs.shape[0]
                
                if wt is not None:
                    wt = wt.numpy().astype(np.float64)
                else :
                    wt = np.ones((xt.shape[0],),dtype=np.float64) / xs.shape[0]
                
            # creating object for interpolation
            interp_G = InterpMeas(metric=self.metric,t_val=self.t_val,approx_interp=approx_interp,
                                  learn_support=self.learn_support)
    
            
            interp_G.fit(xs,xt,a=wt, b=ws)
            G, weight_G, cost_g= interp_G.int_m, interp_G.weights, interp_G.cost
            interp_G.int_init = G
            self.int_m = interp_G.int_m
            self.cost_g = cost_g
            # preparing output for differentiable cost
            if istensor:
                eps = 0
    
                Ms = euclidean_dist_torch(xs_.double(), xt_.double()).pow(self.p)
             
                with torch.no_grad():

                    Ms_aux =  Ms.detach().data.numpy()
                    normMs = np.max(Ms_aux) if np.max(Ms_aux)>1 else 1
    
                    #print(np.sum(a),np.sum(b),np.sum(c),)
                    gamma_s = ot.emd(ws, wt, Ms_aux/normMs)
                    planS = torch.from_numpy(gamma_s)
    
                cost = (torch.sum(Ms*planS)+ eps)**(1/self.p)
            
                return cost




def dataloader_to_df(dataloader):
    data = []
    labels = []
    
    for inputs, target in dataloader:
        # Convert inputs to numpy array and flatten the images
        inputs = inputs.numpy().reshape(inputs.shape[0], -1)  # Flatten each image
        target = target.numpy()
        
        # Append to lists
        data.append(inputs)
        labels.append(target)
    
    # Concatenate all batches
    data = np.vstack(data)
    labels = np.concatenate(labels)
    
    # Check shapes for debugging
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Ensure the length of labels matches the number of rows in DataFrame
    if len(labels) != df.shape[0]:
        raise ValueError("The number of labels does not match the number of data samples.")
    
    df['label'] = labels
    
    return df


def process_data(x,y):
    # index_list = torch.argmax(torch.Tensor(y), dim=1)
    #device = 'cpu'

    numpy_data = np.array(x)
    numpy_labels = np.array(y)

    dim = x.shape[1]
    assert len(numpy_data) == len(numpy_labels)

    data_with_labels = [(data, label) for data, label in zip(numpy_data, numpy_labels)]

    batch_size = 32  
    dataset = DatasetSplit(data_with_labels,numpy_data,numpy_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    targets1 = dataset.targets
    vals1, cts1 = torch.unique(targets1, return_counts=True)
    min_labelcount = 2
    V1 = torch.sort(vals1[cts1 >= min_labelcount])[0]
    idxs1 = np.array([i for i in range(len(targets1))])
    classes1 = vals1
    
    M1, C1 = compute_label_stats(data_loader, targets1, idxs1, classes1, diagonal_cov=True)
    # print(M1.shape)
    # print(C1.shape)
    DA = (dataset.dataset.view(-1,dim).to(device), dataset.targets.to(device))
    print("DA: {}".format(DA[0].shape))
    XA = augmented_dataset(DA, means=M1, covs=C1, maxn=10000)
    
    return XA

class DatasetSplit(Dataset):

    def __init__(self, data_with_labels,numpy_data,label):
        self.data_with_labels = data_with_labels
        self.dataset = torch.Tensor(numpy_data)
        self.targets = torch.LongTensor(label)
        self.idxs = np.array([i for i in range(len(label))])
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.data_with_labels[index]
        return image, label

def cal_distance(xs_t,xt_t,support=200,n_epoch=50,metric='euclidean',t_val=0.5):

    
    xs = xs_t.numpy()
    xt = xt_t.numpy()


    k,dim = xs.shape[0],xs.shape[1]
    logging.info('sample size in cal_distance:  {}'.format(k))
    logging.info('xs.shape in cal_distance: {}'.format(xs.shape))
    logging.info('xt.shape in cal_distance: {}'.format(xt.shape))
    logging.info('Support size in call_distance: {}'.format(support))
    
    fedot_st = FedOT(n_supp=support,n_epoch=n_epoch,metric=metric,t_val=t_val)
    
    fedot_start = time.time()
    fedot_st.fit(xs_t,xt_t)
    st_distance = fedot_st.cost
    fedot_end = time.time()
    logging.info('time needed: {}'.format(fedot_end - fedot_start))
    logging.info('list_cost: {}'.format([float(cost) for cost in fedot_st.list_cost]))

    return st_distance,fedot_st.int_meas

# MAIN PROGRAM

if args.dataset == 'OCT':
    oct_datasetreader = importlib.import_module('DatasetReader.Oct')
    trainset = oct_datasetreader.OCTDataset(args.datasetpath)#use default transform defined in OCTDataset
    testset = oct_datasetreader.OCTDataset(args.referencepath)#use default transform defined in OCTDataset


trainloader = DataLoader(trainset, batch_size=32, shuffle=False)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

df_train = dataloader_to_df(trainloader)[:1000]
df_test = dataloader_to_df(testloader)[:1000]
logging.info('df_train shape: {}'.format(df_train.shape))
logging.info('df_test shape: {}'.format(df_test.shape))

#device = 'cpu'
sample_num = 100
train_x, train_y = df_train.iloc[:sample_num,:-1],  df_train.iloc[:sample_num,-1]
test_x, test_y = df_test.iloc[:sample_num,:-1],  df_test.iloc[:sample_num,-1]

XA = process_data(train_x,train_y) #tensor
XT = process_data(test_x,test_y)#tensor

logging.info('XA shape: {}'.format(XA.shape))
logging.info('XT shape: {}'.format(XT.shape))

t_val = 0.5
candidate_ls = [XA]
candidate_item = ['XA']
results = {}

for i in range(len(candidate_ls)):
    test_data_name = str(candidate_item[i])
    logging.info('--------------- evaluate {} --------------- '.format(test_data_name))
    XS = candidate_ls[i]
    distance, int_meas= cal_distance(XS.cpu(),XT.cpu(),support=XS.shape[0],n_epoch=50,metric='euclidean',t_val=0.5)
    results[test_data_name] = [distance, int_meas]

for i in range(len(candidate_item)):
    logging.info('Distance of {} is {}'.format(candidate_item[i],results[candidate_item[i]][0]))

'''
time 89.83394384384155
Distance of XA is 345.0982863481417
'''
