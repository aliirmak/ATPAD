# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 06:43:13 2019

@author: Ali
"""
#%% libs
# importing required libraries
import numpy as np
import pandas as pd

# import support vector classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier # multi-layer perceptron (MLP)
from sklearn.model_selection import train_test_split

#from timeit import default_timer as timer
import multiprocessing as mp

# optimization tool
import cvxpy as cp

#%% attack and defender
def attack_finder(jj, no_neuron, num_alphas, budget,
                  states_neuron, w, b,
                  x_lp, ax_lp, obj_pre_lp, alphas_lp, y_lp,
                  w1_lp, b1_lp, w2_lp, b2_lp,
                  x_data, y_data,
                  clf_nn):
    obj_lp = cp.Minimize(obj_pre_lp)

    output_vals = np.zeros((2**no_neuron,1))
    x_vals = np.zeros((2**no_neuron,num_alphas))

    # we attack only the faulty data
    if y_data==1:
        x_lp.value = x_data
    else:
        return x_data

    #shuffle(states_neuron)
    #start_p = timer()
    for idx, states in enumerate(states_neuron):
        # print(states)

        # when there is no activation we can skip this
        # make sure bias is bigger than 0
        if (sum(states) == 0) and (b[1]>0):
            output_vals[idx] = 1
            x_vals[idx] = x_lp.value
            continue

		# budget constraints
        cons_lp = []
        cons_lp = [cp.sum(alphas_lp) >= budget]

		# SAT constraints
        for idy, state in enumerate(states):
            if state == 0:
                cons_lp += [y_lp[idy] == 0]
                cons_lp += [ax_lp@w1_lp[:,idy] + b1_lp[idy] <= 0]
            else:
                # cons_lp += [y_lp[idx] > 0]
                cons_lp += [y_lp[idy] == ax_lp@w1_lp[:,idy] + b1_lp[idy]]

        prob_lp = 0
        prob_lp = cp.Problem(obj_lp,cons_lp)

        prob_lp.solve(solver=cp.GUROBI)

        if alphas_lp.value is None:
            alphas_lp.value = np.ones(num_alphas)
        x_vals[idx] = x_lp.value * alphas_lp.value

        # check if the solution is good enough
        # move to the next iteration
        # idx:idx+1 is necessary walkaround
        output_vals[idx] = clf_nn.predict(x_vals[idx:idx+1])
        if output_vals[idx] == -1:
            break

    if sum(output_vals) == 2**no_neuron:
        return x_data
    else:
        return x_vals[idx]

#%% main
if __name__ == '__main__':
    #%% data prep
    scaler = StandardScaler()

    # reading csv file and extracting class column to y.
    df1 = pd.read_csv("data1.csv")

    # infinity check
    #df1.replace(np.inf, np.finfo(np.float64).max-100, inplace=True)
    df1.replace(np.inf, 65500, inplace=True)

    # linear clasify svm
    df1['marker'] = df1['marker'].apply(
            lambda x: -1 if x == "Natural" else 1)
    x = df1.drop("marker",1)   #Feature Matrix
    x[x.columns] = scaler.fit_transform(x[x.columns])
    y = df1["marker"]          #Target Variable

    x1_np = pd.DataFrame.to_numpy(x)
    y1_np = pd.DataFrame.to_numpy(y)
    
    # split data
    x1_np, x2_np, y1_np, y2_np = train_test_split(x1_np, y1_np, test_size=0.2, shuffle=True, stratify=y1_np)

    feature_size = len(x1_np[0])
    batch_size = len(x1_np)
    
    #%% mlp classifier
    _max_iter = 300
    
    hidden_layer_sizes = (5,)
    clf_nn = MLPClassifier(
                solver='lbfgs',    # solver used when training
                alpha=1e-5,        # L2 penalty (regularization term) parameter
                hidden_layer_sizes=hidden_layer_sizes,
                random_state=1,    # If int, it's the seed of the random number gen.
                max_iter=_max_iter,      # max no of iter
                activation="relu") # activation function

    # Train the neural network
    clf_nn.fit(x1_np, y1_np)
    # print(clf_nn)
    # print("Layers", clf_nn.n_layers_)
    # print("Coefs", clf_nn.coefs_)
    # print("Intercepts", clf_nn.intercepts_)
    # print("n inter", clf_nn.n_iter_)
    # print("loss", clf_nn.loss_)

    #%% MLP classifier prediction
    # Now predict the value of the digit on the second half:
    y1_pred_nn = clf_nn.predict(x1_np)
    y2_pred_nn = clf_nn.predict(x2_np)

    #print(sum(y1_pred==y)/len(y), sum(y2_pred==y2)/len(y2))
    #print(accuracy_score(y, y1_pred_nn), accuracy_score(y2, y2_pred_nn))
    #print(confusion_matrix(y,y1_pred_nn))
    #print(confusion_matrix(y2,y2_pred_nn))
    #print(classification_report(y,y1_pred_nn))
    #print(classification_report(y2,y2_pred_nn))

    #%% extract coeff
    w = clf_nn.coefs_
    b = clf_nn.intercepts_

    #%% linear search problem for finding attacks
    # problem variabels
    num_alphas = 128
    budget = 125
    no_neuron = hidden_layer_sizes[0]

    sample_size_testing = x2_np.shape[0]
    x2_np_attack = x2_np.copy()
    y2_pred_attack = y2_np.copy()

    alphas_lp = cp.Variable(num_alphas, boolean = True)
    y_lp = cp.Variable(no_neuron, nonneg = True)

    x_lp = cp.Parameter(num_alphas)
    ax_lp = cp.multiply(x_lp, alphas_lp)

    w1_lp = cp.Parameter( (128,no_neuron), value=w[0])
    b1_lp = cp.Parameter( (no_neuron,),    value=b[0])
    w2_lp = cp.Parameter( (no_neuron,),    value=w[1].squeeze())
    b2_lp = cp.Parameter( (),              value=b[1].squeeze())

    obj_pre_lp = y_lp @ w2_lp + b2_lp

    # mil problem
    states_neuron = []
    from itertools import product
    for i in product([0,1], repeat=no_neuron):
        states_neuron.append(i)

    states_neuron.reverse()
    pool = mp.Pool(processes=8)
    print("pool starting for finding attacks")
    results = [pool.apply_async(attack_finder,
                                args=(jj, no_neuron, num_alphas, budget,
                                      states_neuron, w, b,
                                      x_lp, ax_lp, obj_pre_lp, alphas_lp, y_lp,
                                      w1_lp, b1_lp, w2_lp, b2_lp,
                                      x2_np_attack[jj].copy(),
                                      y2_pred_attack[jj].copy(),
                                      clf_nn
                                      ))
               for jj in range(sample_size_testing)]
    x2_np_attack = np.asarray([p.get() for p in results])
    print("pool ending for finding attacks")

    y2_pred_attack = clf_nn.predict(x2_np_attack)
    
    attack_flip_2 = np.ones(x2_np.shape[0])
    for jxx, xx in enumerate(x2_np_attack):
        if np.array_equal(xx,x2_np[jxx]):
            attack_flip_2[jxx] = -1
            
    #%% linear search problem for defending against attacks
    # first attack the training model
    # use the original model
    # problem variabels
    sample_size_training = x1_np.shape[0]
    x1_np_attack = x1_np.copy()
    y1_pred_attack = y1_np.copy()

    print("pool starting for finding attacks for defense purpose")
    results = [pool.apply_async(attack_finder,
                                args=(jj, no_neuron, num_alphas, budget,
                                      states_neuron, w, b,
                                      x_lp, ax_lp, obj_pre_lp, alphas_lp, y_lp,
                                      w1_lp, b1_lp, w2_lp, b2_lp,
                                      x1_np_attack[jj].copy(),
                                      y1_pred_attack[jj].copy(),
                                      clf_nn
                                      ))
               for jj in range(sample_size_training)]
    x1_np_attack = np.asarray([p.get() for p in results])
    print("pool ending for finding attacks for defense purpose")

    y1_pred_attack = clf_nn.predict(x1_np_attack)
    
    attack_flip_1 = np.ones(x1_np.shape[0])
    for jxx, xx in enumerate(x1_np_attack):
        if np.array_equal(xx,x1_np[jxx]):
            attack_flip_1[jxx] = -1
            
    #%% train + adverserial train
    # split size
    split_perc = 0.25
    
    # first pick arbitrary 
    nono = np.arange(x1_np.shape[0])
    nono_1, nono_2, _, _ = train_test_split(nono, y1_np, test_size=split_perc, shuffle=True, stratify=y1_np)        
    
    nono_unsorted = np.hstack((nono_1,nono_2)) 
    ind_non = np.argsort(nono_unsorted)
    x1_np_attack = np.vstack((x1_np[nono_1],x1_np_attack[nono_2])) 
    x1_np_attack = x1_np_attack[ind_non]
        
    #%% mlp defender classifier
    clf_nn_defend = MLPClassifier(
                                  solver='lbfgs',    # solver used when training
                                  alpha=1e-5,        # L2 penalty (regularization term) parameter
                                  hidden_layer_sizes=hidden_layer_sizes,
                                  random_state=1,    # If int, it's the seed of the random number gen.
                                  max_iter=_max_iter,      # max no of iter
                                  activation="relu") # activation function

    # Train the neural network
    clf_nn_defend.fit(x1_np_attack, y1_np)

    #%% extract coeff
    w2 = clf_nn_defend.coefs_
    b2 = clf_nn_defend.intercepts_

    #%% attack the defender
    # problem variabels belonging to the defender
    x2_np_attack_defend = x2_np.copy()
    y2_pred_attack_defend = y2_np.copy()

    w1_lp = cp.Parameter( (128,no_neuron), value=w2[0])
    b1_lp = cp.Parameter( (no_neuron,),    value=b2[0])
    w2_lp = cp.Parameter( (no_neuron,),    value=w2[1].squeeze())
    b2_lp = cp.Parameter( (),              value=b2[1].squeeze())

    obj_pre_lp = y_lp @ w2_lp + b2_lp

    print("pool starting for attacking defended model")
    results = [pool.apply_async(attack_finder,
                                args=(jj, no_neuron, num_alphas, budget,
                                      states_neuron, w2, b2,
                                      x_lp, ax_lp, obj_pre_lp, alphas_lp, y_lp,
                                      w1_lp, b1_lp, w2_lp, b2_lp,
                                      x2_np_attack_defend[jj].copy(),
                                      y2_pred_attack_defend[jj].copy(),
                                      clf_nn_defend
                                      ))
               for jj in range(sample_size_testing)]
    x2_np_attack_defend = np.asarray([p.get() for p in results])
    print("pool ending for attacking defended model")
    
    attack_flip_3 = np.ones(x2_np.shape[0])
    for jxx, xx in enumerate(x2_np_attack_defend):
        if np.array_equal(xx,x2_np[jxx]):
            attack_flip_3[jxx] = -1

    y2_pred_attack_defend = clf_nn_defend.predict(x2_np_attack_defend)
    #y2_pred_attack_defend = clf_nn_defend.predict(x2_np_attack)
    y1_pred_attack_defend = clf_nn_defend.predict(x1_np_attack)

    #print(sum(y1_pred==y)/len(y), sum(y2_pred==y2)/len(y2))

    # original model
    print(accuracy_score(y1_np, y1_pred_nn),
          accuracy_score(y2_np, y2_pred_nn))
    print(confusion_matrix(y1_np,y1_pred_nn))
    print(confusion_matrix(y2_np,y2_pred_nn))

    # original model getting attacked
    print(accuracy_score(y2_np, y2_pred_attack))
    print(confusion_matrix(y2_np,y2_pred_attack))
    
    # resilient model original data
    y1_pred_nn_defend = clf_nn_defend.predict(x1_np)
    y2_pred_nn_defend = clf_nn_defend.predict(x2_np)
        
    print(accuracy_score(y1_np, y1_pred_nn_defend),
          accuracy_score(y2_np, y2_pred_nn_defend))
    print(confusion_matrix(y1_np,y1_pred_nn_defend))
    print(confusion_matrix(y2_np,y2_pred_nn_defend))
    
    # resilient model getting defended
    print(accuracy_score(y1_np, y1_pred_attack_defend),
          accuracy_score(y2_np, y2_pred_attack_defend))
    print(confusion_matrix(y1_np,y1_pred_attack_defend))
    print(confusion_matrix(y2_np,y2_pred_attack_defend))
    