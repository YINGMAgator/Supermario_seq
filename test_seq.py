#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 11:36:55 2019

@author: yingma
"""

import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env,create_train_env_atari
from src.model import ActorCritic, ActorCritic_seq
import torch.nn.functional as F
import time
from torch.distributions import Categorical
import numpy as np


def str2bool(value):

    if value.lower()=='true':
        return True
    return False
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--saved_path", type=str, default="training_result")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--num_sequence", type=int, default=5)
    parser.add_argument("--final_step", type=int, default=5000)
    parser.add_argument("--start_initial", type=str, default="random",help="inital method, can be random, noop or reset")
    parser.add_argument("--start_interval", type=str, default=20)
    parser.add_argument("--game", type=str, default="Supermario", help= "game select, can be Supermario, MsPacman-v0")

    parser.add_argument("--use_gpu", type=str2bool, default=False)
    args = parser.parse_args()
    return args


def test(opt):
    gate_max=True
    action_max=True
    torch.manual_seed(123)
    if opt.game == "Supermario":
        env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        env, num_states, num_actions = create_train_env_atari(opt.game,opt.saved_path,output_path=None)
#    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,
#                                                    "{}/video_{}_{}.mp4".format(opt.output_path, opt.world, opt.stage))
    model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if torch.cuda.is_available() and opt.use_gpu:
        model.load_state_dict(torch.load(opt.saved_path+"/trained_model"))
        model.cuda()
    else:
        model.load_state_dict(torch.load(opt.saved_path+"/trained_model",
                                         map_location=lambda storage, loc: storage))        

    done=True
    while True:
        if done:
            
            curr_step_test = 0
            cum_r=0    
            with torch.no_grad():
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
                g_0_ini = torch.ones((1))
                
                g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
               
                env.reset()
                if opt.start_initial =='random':
                    for i in range(opt.start_interval):
                        if opt.game =='Supermario':
                            state, reward, _,done, info = env.step(env.action_space.sample())
                        else:
                            state, reward, _,done, info = env.step(env.action_space.sample(),0,video_save=False)
                        if done:
                            env.reset()       
                    state=torch.from_numpy(state)
                else:
                    state = torch.from_numpy(env.reset())
            if opt.use_gpu:
                state = state.cuda()        
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()
                g_0_ini = g_0_ini.cuda() 
                g_0 = g_0.cuda()         
                
            num_interaction=1
            score=0           
            
            
        curr_step_test += 1
        with torch.no_grad():
            h_0 = h_0.detach()
            c_0 = c_0.detach()
    
        if gate_max:
            logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = model(state, h_0, c_0,g_0,g_0_ini,certain=True)
        else:
            logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = model(state, h_0, c_0,g_0,g_0_ini)
        g_0_ini = torch.zeros((1))
        if opt.use_gpu:
            g_0_ini = g_0_ini.cuda()            
        policy = F.softmax(logits, dim=1)
        if action_max:
            action = torch.argmax(policy).item()
        else:
            m = Categorical(policy)
            action = m.sample().item()  
            
            
        if opt.game =='Supermario':
            state, reward, raw_reward,done, info = env.step(action)
        else:
            state, reward, raw_reward,done, info = env.step(action,g_0_cnt,video_save=False)
 
#        if save:
#            print(reward,raw_reward)
        env.render()
#            time.sleep(0.5)
        score += raw_reward
        state = torch.from_numpy(state)
        if opt.use_gpu:
            state = state.cuda()
        cum_r = cum_r+reward
#        actions.append(action)
        if g_0_cnt==0:
            time.sleep(1)
            num_interaction+=1

        else:
            print(g_0_cnt,num_interaction)
        if done:
            if opt.game=="Supermario":
                x=info['x_pos']
                print(x,num_interaction)

if __name__ == "__main__":
    opt = get_args()
    test(opt)
