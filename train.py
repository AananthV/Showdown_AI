import asyncio

import math
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import constants
import config
from showdown.battle_modifier import update_battle

from env_manager import PokemonEnvManager

Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super().__init__()

        self.l1 = nn.Linear(in_features=inputs, out_features=64)
        self.l2 = nn.Linear(in_features=64, out_features=32)
        self.l3 = nn.Linear(in_features=32, out_features=16)
        self.out = nn.Linear(in_features=16, out_features=outputs)

    def forward(self, t):
        t = F.relu(self.l1(t))
        t = F.relu(self.l2(t))
        t = F.relu(self.l3(t))
        t = self.out(t)

        return t

class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)

class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = np.random.permutation(self.num_actions) # explore
            return torch.tensor(action).to(self.device) # explore
        else:
            with torch.no_grad():
                return policy_net(state).argsort(dim = -1, descending=True).to(self.device)[0] # exploit

class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        # print(actions.unsqueeze(-1).size(), policy_net(states).size())
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1)).to(device)

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state, 0)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.next_state, 0)
    t4 = torch.cat(batch.reward)

    return (t1,t2,t3,t4)

# Hyperparameters
batch_size = 32
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.000001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 1000

last100 = []

inputs = 313
outputs = 9

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pem = PokemonEnvManager(device)
asyncio.get_event_loop().run_until_complete(pem.initbots())

strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

agent = Agent(strategy, outputs, device)
memory = ReplayMemory(memory_size)

policy_net = DQN(inputs, outputs).to(device)
target_net = DQN(inputs, outputs).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

plt.ion()
plt.xlabel('Episode #')
plt.ylabel('Win % in last 100 battles')

def battle_is_finished(msg):
    return constants.WIN_STRING in msg and constants.CHAT_STRING not in msg

async def runEpisode(bot):
    bot_state = pem.get_bot_state(bot)
    action_required = True

    while True:
        bot_original_state = pem.get_state(bot)

        action_taken = False

        if bot_state.size()[1] != 313:
            print(bot_original_state)

        if action_required and not pem.bots[bot]['battle'].wait:
            bot_action_list = agent.select_action(bot_state, policy_net)
            # print("Action List: ", bot, bot_action_list)
            bot_move, bot_action = await pem.make_move(bot_original_state, bot_action_list, bot)
            pem.numMoves += 1
            action_taken = True

        msg = await pem.bots[bot]['ps_websocket_client'].receive_message()

        action_required = await update_battle(pem.bots[bot]['battle'], msg)

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, next_states, rewards = extract_tensors(experiences)

            current_q_values = QValues.get_current(policy_net, states, actions)
            # print("Current Q Values: ", current_q_values)
            next_q_values = QValues.get_next(target_net, next_states)
            # print("Next Q Values: ", next_q_values)
            # print("Rewards: ", rewards.unsqueeze(1))
            target_q_values = (next_q_values * gamma) + rewards.unsqueeze(1)
            # print("Target Q Values: ", target_q_values)

            loss = F.mse_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if battle_is_finished(msg):
            winner = msg.split(constants.WIN_STRING)[-1].split('\n')[0].strip()
            print(winner, "won")
            if winner == config.bot_username_1:
                bot_reward = 10
            else:
                bot_reward = -10

            bot_reward = torch.cuda.FloatTensor([bot_reward])

            bot_next_state = torch.cuda.FloatTensor([0 for i in range(inputs)]).unsqueeze(0)

            memory.push(bot_state, bot_action, bot_next_state, bot_reward)

            break

        if action_taken:
            bot_reward = pem.calculate_reward(bot_original_state, bot)

            bot_next_state = pem.get_bot_state(bot)

            memory.push(bot_state, bot_action, bot_next_state, bot_reward)

            bot_state = bot_next_state

async def runRandomEpisode(bot):
    action_required = True

    while True:
        if action_required and not pem.bots[bot]['battle'].wait:
            await pem.make_random_move(bot)

        msg = await pem.bots[bot]['ps_websocket_client'].receive_message()
        action_required = await update_battle(pem.bots[bot]['battle'], msg)

        if battle_is_finished(msg) or pem.numMoves > 500:
            break

async def mainLoop():
    for episode in range(num_episodes):
        print("Starting Battle {}".format(episode + 1))

        await pem.start()

        await asyncio.gather(
            runEpisode(0),
            runRandomEpisode(1)
        )

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(target_net.state_dict(), './state_random_lakh')

        if pem.winner == 0:
            last100.append(1)
            if len(last100) == 2:
                last100.pop()
        else:
            last100.append(0)

        print(last100)

        if len(last100) == 1:
            plt.scatter(episode, sum(last100) / 1)
            plt.show()

        await pem.stop()

async def testLoop():
    for episode in range(5):
        print("Starting Battle {}".format(episode + 1))

        await pem.start()

        await asyncio.gather(
            runEpisode(0),
            runTestEpisode(1)
        )

        await pem.stop()

asyncio.get_event_loop().run_until_complete(mainLoop())
