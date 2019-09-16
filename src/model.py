"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx



class ActorCritic_seq(nn.Module):
    def __init__(self, num_inputs, num_actions,num_sequence):
        super(ActorCritic_seq, self).__init__()
        self.num_sequence = num_sequence
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        
        
        self.conv1_gate = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2_gate = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3_gate = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4_gate = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gate_linear = nn.Linear(32 * 6 * 6, self.num_sequence)
        self.counter = 0
        self.seq_ini_flag=False
        self.bnl = Bernoulli (0.5)
        self.g = torch.zeros((1, self.num_sequence), dtype=torch.float)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx,g_ini):
        if self.counter==self.num_sequence or g_ini==1:
            self.seq_ini_flag = True
                     

        if self.seq_ini_flag:
            
            self.counter = 0
            g = F.relu(self.conv1(x))
            g = F.relu(self.conv2(g))
            g = F.relu(self.conv3(g))
            g = F.relu(self.conv4(g))   
            self.g = torch.sigmoid(self.gate_linear(g.view(g.size(0),-1)))
            
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))  
            self.x_pre = x
            self.seq_ini_flag = False
         #   ggg=self.bnl.sample()
          #  if ggg==1 and g_ini!=1:
           #     self.counter +=1
        else:
            self.counter += 1
        hx, cx = self.lstm(self.x_pre.view(self.x_pre.size(0), -1), (hx, cx))   
         
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx, self.g,self.counter
