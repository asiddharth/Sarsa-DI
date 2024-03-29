"""
Translated from https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp
Algorithm described at https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node89.html

Some minor adjustments to constants were made to make the program work on environments
besides Mountain Car.
"""

import random
import math
import numpy as np
import gym
import sys
from gym import wrappers

np.random.seed(0)

env = gym.make(sys.argv[1])
num_epi = int(sys.argv[2])

initial_epsilon = 0.1 # probability of choosing a random action (changed from original value of 0.0)
alpha = 0.1 # learning rate
lambda_ = float(sys.argv[3]) # trace decay rate
gamma = 1.0 # discount rate
N = 30000 # memory for storing parameters (changed from original value of 3000)

M = env.action_space.n
NUM_TILINGS = 10
NUM_TILES = 8

def main():
    epsilon = initial_epsilon
    theta = np.zeros(N) # parameters (memory)

    #Train with sys.argv[4] step size
    for episode_num in xrange(num_epi):
        print episode_num,episode(epsilon, theta, env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
        epsilon = epsilon * 0.999 # added epsilon decay
    return theta

def episode(epsilon, theta, max_steps):
    Q = np.zeros(M) # action values
    e = np.zeros(N) # eligibility traces
    F = np.zeros((M, NUM_TILINGS), dtype=np.int32) # features for each action

    def load_F(observation):
        state_vars = []
        for i, var in enumerate(observation):
            range_ = (env.observation_space.high[i] - env.observation_space.low[i])
            # in CartPole, there is no range on the velocities, so default to 1
            if range_ == float('inf'):
                range_ = 1
            state_vars.append(var / range_ * NUM_TILES)

        for a in xrange(M):
            F[a] = get_tiles(NUM_TILINGS, state_vars, N, a)

    def load_Q():
        for a in xrange(M):
            Q[a] = 0
            for j in xrange(NUM_TILINGS):
                Q[a] += theta[F[a,j]]

    def getEUpdate(action):
        eUpdate = 0
        for j in xrange(NUM_TILINGS):
            eUpdate += e[F[action,j]]
        return eUpdate


    observation = env.reset()
    load_F(observation)
    load_Q()
    action = np.argmax(Q) # numpy argmax chooses first in a tie, not random like original implementation
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    step = 0
    lastStateActionValue = Q[action]
    lastState = observation
    while True:
        step +=1
	observation, reward, done, info = env.step(action)
        delta = reward - lastStateActionValue
        temp = 0
        eUpdate = getEUpdate(action)
        load_F(observation)
        load_Q()
        next_action = np.argmax(Q)
        if np.random.random() < epsilon:
            next_action = env.action_space.sample()
        if not done:
            delta += gamma * Q[next_action]
            temp = Q[next_action]
        load_F(lastState)
        load_Q()

        #print "eUpdate", eUpdate, "lastStateActionValue", lastStateActionValue, "temp", temp 
        #eligibility update
        e *= gamma * lambda_ 
        for a in xrange(M):
            v = 0.0
            if a == action:
                v = 1.0
            for j in xrange(NUM_TILINGS):
                e[F[a,j]] += v*alpha*(1-gamma*lambda_*eUpdate) #/ float(NUM_TILINGS)
	
	#theta Update
        theta +=  delta * e #* (float(1.0) / float(NUM_TILINGS))
        for a in xrange(M):
            v = 0.0
            if a == action:
                v = 1.0
            for j in xrange(NUM_TILINGS):
                theta[F[a,j]] += v*float(alpha * (lastStateActionValue-Q[action])) #/ float(NUM_TILINGS)

        load_F(observation)
        load_Q()

        if done or step > max_steps:
            break
        action = next_action
        lastState = observation
        lastStateActionValue = temp
    return step

# translated from https://web.archive.org/web/20030618225322/http://envy.cs.umass.edu/~rich/tiles.html
def get_tiles(num_tilings, variables, memory_size, hash_value):
    num_coordinates = len(variables) + 2
    coordinates = [0 for i in xrange(num_coordinates)]
    coordinates[-1] = hash_value

    qstate = [0 for i in xrange(len(variables))]
    base = [0 for i in xrange(len(variables))]
    tiles = [0 for i in xrange(num_tilings)]

    for i, variable in enumerate(variables):
        qstate[i] = int(math.floor(variable * num_tilings))
        base[i] = 0

    for j in xrange(num_tilings):
        for i in xrange(len(variables)):
            if (qstate[i] >= base[i]):
                coordinates[i] = qstate[i] - ((qstate[i] - base[i]) % num_tilings)
            else:
                coordinates[i] = qstate[i] + 1 + ((base[i] - qstate[i] - 1) % num_tilings) - num_tilings

            base[i] += 1 + (2 * i)
        coordinates[len(variables)] = j
        tiles[j] = hash_coordinates(coordinates, memory_size)

    return tiles

rndseq = np.random.randint(0, 2**32-1, 2048)

def hash_coordinates(coordinates, memory_size):
    total = 0
    for i, coordinate in enumerate(coordinates):
        index = coordinate
        index += (449 * i)
        index %= 2048
        while index < 0:
            index += 2048
        total += rndseq[index]
    index = total % memory_size
    while index < 0:
        index += memory_size
    return index

theta = main()
'''
totalSteps = 0
print "starting"
for episode_num in xrange(1000):
    pers = int(sys.argv[5])
    epsilon = 0
    alpha = 0
    thisstep = episode(epsilon, theta, env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    totalSteps+=thisstep
    #print episode_num,thisstep 
print sys.argv[4] + " " + sys.argv[5] + " " + str(float(totalSteps)/float(1000))
'''
