import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import torch.nn.utils as utils

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from mlagents.envs import UnityEnvironment

env = UnityEnvironment(file_name="UnityEnvironments/3DBall/3DBall_Headless.exe")

def Model():
    model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size[0])),
    ('relu',nn.ReLU()),
    ('fc2',nn.Linear(hidden_size[0], hidden_size[1])),
    ('relu',nn.ReLU()),
    ('fc3',nn.Linear(hidden_size[1], output_size)),
    ('Tanh',nn.Tanh())]))
    return model

def f(w):
    try_model = Model()
    try_model.parameters = utils.vector_to_parameters(w, try_model.parameters())

    num_epochs = 1
    #changed num epochs from +1
    for e in range(num_epochs):
        env_info = env.reset(train_mode=True)[brain_name]
        scores = np.zeros(len(env_info.rewards))
        states = env_info.vector_observations
        for t in range(10000):
            states_tensor = torch.from_numpy(states).float()
            actions = try_model.forward(states_tensor).cpu().detach().numpy()
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            states = next_states
            scores = np.add(scores, rewards) 
            if True in dones:
                break
    return np.mean(scores)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

seed = 0
use_seed = True
if use_seed:
    np.random.seed(seed)
    torch.manual_seed(seed)

input_size = len(env_info.vector_observations[0])
hidden_size = [64,32]
output_size = brain.vector_action_space_size[0]

# hyperparameters
npop = 10 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.02 # learning rate

model = Model()
ES_accuracies = []

w = utils.parameters_to_vector(Model().parameters())
for i in range(61):
    ES_accuracy = f(w)
    ES_accuracies.append(ES_accuracy.item())
    if i % 20 == 0:
        print('iter %d, reward: %f' % (i, ES_accuracy))
    R = np.zeros(npop)
    N = torch.randn(npop, w.size()[0])
    for j in range(npop):
        w_try = w + sigma*N[j]
        R[j] = f(w_try)

    A = (R - np.mean(R)) / (np.std(R)+0.00001)
    w = w + alpha/(npop*sigma) * torch.mm(torch.transpose(N, 0, 1), torch.from_numpy(A).view(npop,1).float()).view(w.size()[0])
    
plt.plot(ES_accuracies,label='EvoStrat')
plt.ylabel('accuracy')
plt.legend()
plt.show()


model.parameters = utils.vector_to_parameters(w, model.parameters())
torch.save(model.state_dict(), '3Dball_checkpoint.pth')
env.close()
