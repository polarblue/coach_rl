
"""
modifes an agents target function 
based on a policy of an already pretrained agent
pretrained agents policy is shared through
an lstm to understand 
"""
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,LSTM
from dqn_agent import DQNAgent

import matplotlib.pyplot as plt
EPISODES = 1000



def buildlstm():
    """
    builds keras lstm model
    """
    look_back = 4
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 11)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['acc'])
    return model

def trainlstm(model,train_x,train_y):
    """
    trains the model based on already made data
    """
    model.fit(train_x, train_y, epochs=4, batch_size=32, verbose=2)
    return model

def create_dataset(dataset,success,look_back=4):
    """
    sets up the lstm data in correct shape based on look
    back
    """
    dataset=np.asarray(dataset)
    success = np.asarray(success)
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(success[i + look_back])
    return np.array(dataX), np.array(dataY)


def lstmpredict(randomelem,totaldata,lookback,totalresults,model):
    """
    allows to train with random pull 
    from the agents memory 
    checks a the sequence of agents 
    memory
    """
    a=randomelem
    inval =totaldata[(a-lookback):(a)]
    return(model.predict(inval))
        
        
def determine_sucess(done,score):
    """
    creates a value that determines what a prefered sequence in
    the game causes an upperbound on sum of reward for coached agent
    """
    if(done):
        if (score <199):
            success = (.01)
            return success
        else:
            success =10
            return success
    else:
        success = 1
        return success

def train_expert():
    """
    craetes an agent that is trained to an optimal policy
    and captures all values required to train lstm
    """
    agent_score_keep=[]
    agent_episode_keep=[]
    score = 0
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(len(env.reset()),env.action_space.n)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    index = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
#                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            a_reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            index = index + 0
            agent.remember(state, action, a_reward, next_state, done,index)
            score = score + reward 
            success=determine_sucess(done,score)
            agent.lstm_data(state, action, reward, next_state, done,success) 
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, score, agent.epsilon))
                agent_score_keep.append(score)
                agent_episode_keep.append(e)
                score=0
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size,None)
    return agent,agent_score_keep,agent_episode_keep

def train_coach(agent):
    """
    properly trains coach
    """
    dataset_in=agent.lstm_training
    data_test=agent.lstm_out
    train_x,train_y=create_dataset(dataset_in,data_test,look_back=4)
    coach=buildlstm()
    trained_coach=trainlstm(coach,train_x,train_y)
    return trained_coach



def train_speed_agent(coach):
    """
    takes caoch(lstm) to modify the
    target reward function for the 
    agent
    
    """
    score = 0
    coaching_score_keep=[]
    coaching_episode_keep=[]
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    coaching = DQNAgent(len(env.reset()),env.action_space.n)
    done = False
    batch_size = 32
    index = 0
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            #env.render()
            action = coaching.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            index = index + 1
            coaching.remember(state, action, reward, next_state, done,index)
            score = score + reward 
            success=determine_sucess(done,score)
            coaching.lstm_data(state, action, reward, next_state, done,success) 
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, coaching.epsilon))
                coaching_score_keep.append(score)
                coaching_episode_keep.append(e)
                score=0
                break
        if len(coaching.memory) > batch_size:
            coaching.replay(batch_size,coach) 
    return agent,coaching_score_keep,coaching_episode_keep
    
                
if __name__ == "__main__":
    agent,agent_score_keep,agent_episode_keep=train_expert()
    caoch=train_coach(agent)
    coached,coaching_score_keep,coaching_episode_keep=train_speed_agent(caoch)
    plt.plot(agent_episode_keep,agent_score_keep)
    plt.plot(coaching_episode_keep,coaching_score_keep)
    plt.show()

#    def coach_predict(time):
#        if(time<4):
         
