import torch
from torch import nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
import torch.nn.utils as utils
from collections import OrderedDict
import matplotlib.pyplot as plt
import threadlib as tl
import numpy as np
from mlagents.envs import UnityEnvironment
from random import randrange
from time import sleep

def Model():
    model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size[0])),
    ('relu',nn.ReLU()),
    ('fc2',nn.Linear(hidden_size[0], hidden_size[1])),
    ('relu',nn.ReLU()),
    ('fc3',nn.Linear(hidden_size[1], output_size)),
    ('Tanh',nn.Tanh())]))
    return model

def f(args):
    w = args[0]
    index = args[1]
    N = args[2]
    R = args[3]

    w_try = w + sigma*N

    model = Model()
    model.parameters = utils.vector_to_parameters(w_try, model.parameters())
    model.train()
    
    num_epochs = 1
    for e in range(num_epochs):
        env_info = envs[index].reset(train_mode=True)[brain_name]
        scores = np.zeros(len(env_info.rewards))
        states = env_info.vector_observations
        for t in range(10000):
            states_tensor = torch.from_numpy(states).float()
            actions = model.forward(states_tensor).cpu().detach().numpy()
            env_info = envs[index].step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            states = next_states
            scores = np.add(scores, rewards) 
            if True in dones:
                break
    R[index] = np.mean(scores)
    
def wait_delay(d):
    print('sleeping for (%d)sec' % d)
    sleep(d)
    
def f_test(w):
    model = Model()
    model.parameters = utils.vector_to_parameters(w, model.parameters())
    model.eval()
    
    env_info = envs[0].reset(train_mode=training_mode)[brain_name]
    scores = np.zeros(len(env_info.rewards))
    states = env_info.vector_observations
    for t in range(10000):
        states_tensor = torch.from_numpy(states).float()
        actions = model.forward(states_tensor).cpu().detach().numpy()
        env_info = envs[0].step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        states = next_states
        scores = np.add(scores, rewards) 
        if True in dones:
            break
    return np.mean(scores)

training_mode = True

seed = 0
use_seed = True
if use_seed:
    np.random.seed(seed)
    torch.manual_seed(seed)

# hyperparameters
npop = 30 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.02 # learning rate

envs = []
for i in range(npop):
    env = UnityEnvironment(file_name="UnityEnvironments/3DBall/3DBall_Headless.exe", worker_id=i)
    envs.append(env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=training_mode)[brain_name]

input_size = len(env_info.vector_observations[0])
hidden_size = [64,32]
output_size = brain.vector_action_space_size[0] 
    
w = utils.parameters_to_vector(Model().parameters())
pool = tl.ThreadPool(npop)
IDs = [i for i in range(npop)]
ES_accuracies = []

for i in range(501):
    if i % 5 == 0:
        ES_accuracy = f_test(w)
        ES_accuracies.append(ES_accuracy)
        print('iter %d, reward: %f' % (i, ES_accuracy))
    R = np.zeros(npop)
    N = torch.randn(npop, w.size()[0])
    for j, d in enumerate(IDs):
        pool.add_task(f, [w,d,N[d],R])
    pool.wait_completion()
    
    A = (R - np.mean(R)) / (np.std(R)+0.00001)
    w = w + alpha/(npop*sigma) * torch.mm(torch.transpose(N, 0, 1), torch.from_numpy(A).view(npop,1).float()).view(w.size()[0])    

print("Done!")

model = Model()
model.parameters = utils.vector_to_parameters(w, model.parameters())
torch.save(model.state_dict(), 'Test_Parallel_checkpoint.pth')
for env in envs:
    env.close()