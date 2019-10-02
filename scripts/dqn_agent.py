#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:06:35 2019

@author: racss
"""

import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,LSTM

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.lstm_training =[]
        self.lstm_out =[]

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done,index):
        self.memory.append((state, action, reward, next_state, done,index))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    def lstmmodelpredict(self,model,index,lookback = 4):
        if(index <= 3):
            values=np.asarray(self.lstm_training)
            inval =values[(lookback-index):(lookback)]  
            aux=np.zeros(((lookback-index),11))
            newval=np.reshape(np.append((aux.flatten()),inval.flatten()),(1,4,11))
            return(model.predict(newval))
        else:
            values=np.asarray(self.lstm_training)
            inval =values[(index-lookback):(index)]
#            print(inval)
#            print(index)
#            print(len(values))
            newval=inval.reshape((1,4,11))
            return(model.predict(newval))
        
        

    def replay(self,batch_size,coach):
        """
        performs normarl replay mem when lstm model is missing
        when model is available uses the lstm to modify the target
        dqn agent based on previus states
        """
        if(coach):
            print("in coach")
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done , index in minibatch:
                target = reward
                if not done:
#                    print(self.lstmmodelpredict(coach,index))
                    target = self.lstmmodelpredict(coach,index)*(reward + self.gamma *
                              np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            print("epsilon decrease")
            if self.epsilon > self.epsilon_min:
                print("decreasing epsilon")
                self.epsilon *= self.epsilon_decay
        
            
        else:
            print("not in coach")
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done,index in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma *
                              np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
    
    def lstm_data(self,state, action, reward, next_state, done,success):
        states=np.concatenate((state,next_state))
        a=np.insert(states,len(states),[action,reward,done])
        self.lstm_training.append(a)
        self.lstm_out.append(success)