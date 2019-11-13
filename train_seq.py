#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:22:55 2019

@author: yingma
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.env import create_train_env,create_train_env_atari
from src.model import ActorCritic,ActorCritic_seq
from src.optimizer import GlobalAdam
from src.process_seq import local_train, local_test, local_test_certain
import torch.multiprocessing as _mp
import shutil

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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--num_sequence", type=int, default=5)
    parser.add_argument("--final_step", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--log_interval", type=int, default=10, help="Number of steps between log")
    parser.add_argument("--max_test_steps", type=int, default=10000, help="Max Number of steps for test")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--start_initial", type=str, default="random",help="inital method, can be random, noop or reset")
    parser.add_argument("--start_interval", type=str, default=20)
    parser.add_argument("--internal_reward",type=float,default=0.1)
    parser.add_argument("--saved_path", type=str, default="training_result")
    
    ##
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--value_loss_coef", type=float, default=None)
    
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--load_from_previous_stage", type=str2bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=str2bool, default=False)
    # atari arguement 
    parser.add_argument("--game", type=str, default="Supermario", help= "game select, can be Supermario, MsPacman-v0")    
    args = parser.parse_args()
    return args


def train(opt):
#    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
#    if not os.path.isdir(opt.saved_path):
#        os.makedirs(opt.saved_path)
    if not opt.saved_path:
        if opt.game == "Supermario":
            saved_path="{}_{}_{}_{}".format(opt.game,opt.num_sequence,opt.internal_reward,opt.world,opt.stage)
        else:
            saved_path="{}_{}".format(opt.game,opt.num_sequence)
    else:
        saved_path=opt.saved_path
    if not os.path.isdir(saved_path):
        os.makedirs(saved_path)
    mp = _mp.get_context("spawn")
    if opt.game == "Supermario":
        env, num_states, num_actions = create_train_env(opt.world, opt.stage,opt.action_type, opt.final_step)
    else:
        env, num_states, num_actions = create_train_env_atari(opt.game,saved_path,output_path=None)
    print(opt.num_sequence)
    global_model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if opt.use_gpu:
        global_model.cuda()
    global_model.share_memory()


    if opt.load_model_path:
        file_ = opt.load_model_path+"/trained_model"
        global_model.load_state_dict(torch.load(file_))
            
            
    if opt.load_from_previous_stage:
        if opt.stage == 1:
            previous_world = opt.world - 1
            previous_stage = 4
        else:
            previous_world = opt.world
            previous_stage = opt.stage - 1
        file_ = "{}/a3c_seq_super_mario_bros_{}_{}".format(opt.game,saved_path, previous_world, previous_stage)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_))
#    global_model.load_state_dict(torch.load("{}_{}_{}_{}/trained_model_{}_{}".format(opt.game,opt.num_sequence,opt.internal_reward,opt.lr, opt.world, opt.stage),
#                                     map_location=lambda storage, loc: storage))
    
    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    processes = []
#    local_train(0, opt, global_model, optimizer, save=True)
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
#    process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
#    process.start()
#    processes.append(process)
#    process = mp.Process(target=local_test_certain, args=(opt.num_processes, opt, global_model))
#    process.start()
#    processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
