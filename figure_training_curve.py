#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 22:01:06 2020

@author: yingma
"""
import numpy as np
import matplotlib.pyplot as plt
saved_path="training_result4"
index=1
Num_seq=[5,10,0]
x_lim=300000
N=10
for i in range(3):
    num_seq=Num_seq[i]
    Num_interact1 = np.load(saved_path+"/Num_interact1_{}".format(index)+"nums_"+str(num_seq)+".npy")
    Cum_reward = np.load(saved_path+"/Cum_reward_{}".format(index)+"nums_"+str(num_seq)+".npy")
    X = np.load(saved_path+"/X_{}".format(index)+"nums_"+str(num_seq)+".npy")
    Episode1 = np.load(saved_path+"/Episode1_{}".format(index)+"nums_"+str(num_seq)+".npy")
    
    Num_interact = np.load(saved_path+"/Num_interact_{}".format(index)+"nums_"+str(num_seq)+".npy")
    X1 = np.load(saved_path+"/X1_{}".format(index)+"nums_"+str(num_seq)+".npy")
    X2 = np.load(saved_path+"/X2_{}".format(index)+"nums_"+str(num_seq)+".npy")
    X3 = np.load(saved_path+"/X3_{}".format(index)+"nums_"+str(num_seq)+".npy")
    
    plt.figure(0)
    plt.plot(Num_interact1[0:x_lim], X[0:x_lim])
    plt.title('distance')
    plt.xlabel('interactive number')
    plt.ylabel('X')
    plt.legend(Num_seq)
    
    
    plt.figure(1)
    plt.plot(Episode1, X)
    plt.title('distance')
    plt.xlabel('Episode')
    plt.ylabel('X')
    
    plt.figure(2)
    plt.plot(Num_interact1, Cum_reward)
    plt.title('reward')
    plt.xlabel('interactive number')
    plt.ylabel('Cum_reward')

    x_limit = sum(k<x_lim for k in Num_interact)
    plt.figure(3)
    plt.plot(Num_interact[0:x_limit], X2[0:x_limit])
    plt.title('X2')
    plt.xlabel('interactive number')
    plt.ylabel('X2')
    
    X2 = np.convolve(X2, np.ones((N,))/N, mode='valid')
    plt.figure(4)
    plt.plot(Num_interact[0:x_limit-N], X2[0:x_limit-N])
#    plt.title('X2')
    plt.xlabel('interactive number')
    plt.ylabel('X2')









