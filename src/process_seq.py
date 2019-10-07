#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:46:03 2019

@author: yingma
"""

"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
from src.env import create_train_env
from src.model import ActorCritic_seq
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import numpy as np

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,opt.final_step)
    local_model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    state = torch.from_numpy(env.reset())
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    debug_test_i=0 
    loss_matrix = np.zeros((100000))
    Cum_reward = []
    X = []
    Episode = []    
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_seq_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            print("Process {}. Episode {}".format(index, curr_episode),done,debug_test_i)
        curr_episode += 1

        local_model.load_state_dict(global_model.state_dict())
        g_0_cnt = 0 
        g_0_ini = torch.ones((1))
        g_0 = torch.zeros((1, opt.num_sequence), dtype=torch.float)
        if done:
            
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            
            cum_r=0           
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
            g_0 = g_0.detach()
        
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            g_0_ini = g_0_ini.cuda()
            g_0 = g_0.cuda()

        log_policies = []
        log_gates = []
        values = []
        rewards = []
        reward_internals = []
        entropies = []

        for aaaaa in range(opt.num_local_steps):
            curr_step += 1
            g_pre = g_0
            g_pre_cnt = g_0_cnt

            logits, value, h_0, c_0, g_0,g_0_cnt,gate_flag1, gate_flag2= local_model(state, h_0, c_0,g_0_ini)

            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()
            state, reward, done, info = env.step(action)
            reward_internal = reward
            
            if g_0_ini==1:

                log_gate = torch.zeros((), dtype=torch.float)
                if opt.use_gpu:
                    log_gate = log_gate.cuda()
            elif gate_flag1:

#                log_gate = log_gate
                log_gate = torch.zeros((), dtype=torch.float)
            elif gate_flag2:
                  
#                log_gate = log_gate + torch.log(1-g_pre[0,g_pre_cnt]) 
                log_gate = torch.log(1-g_pre[0,g_pre_cnt]) 
            else:
#                log_gate = log_gate+torch.log(g_0[0,g_0_cnt-1])
                log_gate = torch.log(g_0[0,g_0_cnt-1])
                if reward>0:
                    reward_internal = reward+0.01
            g_0_ini = torch.zeros((1))
            if opt.use_gpu:
                g_0_ini = g_0_ini.cuda()
#            if save:
#                env.render()
#                print(reward)
#                time.sleep(1)  
            state = torch.from_numpy(state)
            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True
                print('max glabal step achieve')

            if done:
                print(info['x_pos'])   
                curr_step = 0
                state = torch.from_numpy(env.reset())
                if opt.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            log_gates.append(log_gate)
            rewards.append(reward)
            reward_internals.append(reward_internal)
            entropies.append(entropy)
            cum_r+=reward
            if done:
                Episode.append(curr_episode)
                X.append(info['x_pos'])
                Cum_reward.append(cum_r)
                
                break
#        print(log_policies,log_gates)
        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R, _, _ ,_,_,gate_flag1, gate_flag2= local_model(state, h_0, c_0,g_0_ini)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        
#        next_value = R
#        for value, log_policy, log_gate, reward, reward_internal, entropy in list(zip(values, log_policies, log_gates, rewards,reward_internals, entropies))[::-1]:
#            gae = gae * opt.gamma * opt.tau
#            gae = gae + reward_internal + opt.gamma * next_value.detach() - value.detach()
#            next_value = value
#            actor_loss = actor_loss + (log_policy+log_gate) * gae
#            R = R * opt.gamma + reward
#            critic_loss = critic_loss + (R - value) ** 2 / 2
#            entropy_loss = entropy_loss + entropy
        
# estimate internal reward directly      
        if not (gate_flag1 or gate_flag2):
            if R >0:    
                R=R+0.01
        next_value = R  
        for value, log_policy, log_gate, reward, reward_internal, entropy in list(zip(values, log_policies, log_gates, rewards,reward_internals, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward_internal + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + (log_policy+log_gate) * gae
            R = R * opt.gamma + reward_internal
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
            
# estimate external reward      

#        next_value = R  
#        for value, log_policy, log_gate, reward, reward_internal, entropy in list(zip(values, log_policies, log_gates, rewards,reward_internals, entropies))[::-1]:
#            gae = gae * opt.gamma * opt.tau
#            gae = gae + reward_internal-0.01* + opt.gamma * next_value.detach() - value.detach()
#            next_value = value
#            actor_loss = actor_loss + (log_policy+log_gate) * gae
#            R = R * opt.gamma + reward
#            critic_loss = critic_loss + (R - value) ** 2 / 2
#            entropy_loss = entropy_loss + entropy
            
            
        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        loss_matrix[curr_episode]=total_loss.detach().cpu().numpy()
        
        if not np.isfinite(total_loss.detach().cpu().numpy()):
            debug_test_i+=1
        if curr_episode%1000==0:
#            print('aaaaaaaaaaa',X,Cum_reward)
            np.save("{}/loss{}".format(opt.saved_path,index), loss_matrix)
            np.save("{}/Cum_reward{}".format(opt.saved_path,index), Cum_reward)
            np.save("{}/X{}".format(opt.saved_path,index), X)
            np.save("{}/Episode{}".format(opt.saved_path,index), Episode)

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type,opt.final_step)
    local_model = ActorCritic_seq(num_states, num_actions,opt.num_sequence)
    local_model.eval()
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    Cum_reward = []
    X = []
    i=0
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
                g_0_ini = torch.ones((1))
                state = torch.from_numpy(env.reset())
                cum_r=0
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0,g_0,g_0_cnt,gate_flag,_ = local_model(state, h_0, c_0,g_0_ini)
        #print(g_0,g_0_cnt)
        g_0_ini = torch.zeros((1))
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, info = env.step(action)
#        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        cum_r = cum_r+reward
        if done:
            print(i,'test',info['x_pos'])
            i=i+1
            curr_step = 0
            actions.clear()
            state = env.reset()
            
            X.append(info['x_pos'])
            Cum_reward.append(cum_r)
                
                
        state = torch.from_numpy(state)
        
        if i%100==0:

            np.save("{}/Cum_reward_test".format(opt.saved_path), Cum_reward)
            np.save("{}/X_test".format(opt.saved_path), X)
