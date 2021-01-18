############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import numpy as np
import random
import torch
import collections

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 500 
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Buffer
        self.buffer = collections.deque(maxlen=5000) 
        self.dqn = DQN()
        self.distance_until_goal = None
        self.number_of_episodes = 0
        self.epsilon = 0.99 
        self.epsilon_minimum = 0.1 
        self.epsilon_decay_rate = 0.999 

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.num_steps_taken = 0
            self.number_of_episodes+=1 
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        if self.state is None:
            discrete_action =  np.random.choice((0,1,2,3))
                # Convert the discrete action into a continuous action.
            action = self._discrete_action_to_continuous(discrete_action)
        else:
            self.epsilon = max(self.epsilon_minimum, self.epsilon*self.epsilon_decay_rate)
            discrete_action = self._choose_next_action_epsilon_greedy(self.epsilon)
#               Convert the discrete action into a continuous action.
            action = self._discrete_action_to_continuous(discrete_action)
#          Update the number of steps which the agent has taken
        self.num_steps_taken += 1
#         Store the state; this will be used later, when storing the transition
        self.state = state
#         Store the action; this will be used later, when storing the transition
        self.action = action
        return action
    
#         Function for the agent to choose its next action
    def _choose_next_action_epsilon_greedy(self, epsilon):
        q_val = self.dqn.q_values_single_state(self.state) 
        # set A => array of 4 lots of epsilon/4
        A= np.ones(4)*(epsilon/4)
        A_set = np.zeros([4,1])
        
#         this is the greedy choice that chooses the action as to maximise Q
        best_action = np.argmax(q_val)
#              set the whole state indexed row of A_set to A = epsilon/4
        A_set = A
#              set the best action index to epsilon/4 + 1 - epsilon
        A_set[best_action] += (1.0 - epsilon)
#             Now when choosing the policy index, randomly choose based on the epsilon greedy policy function
        discrete_action = np.random.choice((0,1,2,3),p=tuple(A_set))
        return discrete_action

#      Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move right
            continuous_action = np.array([0.02, 0], dtype=np.float32)
        elif discrete_action == 1: 
            # Move up
            continuous_action = np.array([0, 0.02], dtype=np.float32)
        elif discrete_action == 2: # left
            # Move left
            continuous_action = np.array([-0.02, 0], dtype=np.float32)
        elif discrete_action == 3:
            # move down
            continuous_action = np.array([0, -0.02], dtype=np.float32)
        return continuous_action
    
#      Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _continuous_action_to_discrete(self, cont_action):
        if (cont_action == np.array([0.02, 0], dtype=np.float32)).all():
            # Move right
            discrete_action = 0
        elif (cont_action == np.array([0, 0.02], dtype=np.float32)).all(): 
            # Move up
            discrete_action = 1
        elif (cont_action == np.array([-0.02, 0], dtype=np.float32)).all(): 
            # Move left
            discrete_action = 2
        elif (cont_action == np.array([0, -0.02], dtype=np.float32)).all():
            # move down
            discrete_action = 3
        return discrete_action

#      Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
#          Convert the distance to a reward
        self.distance_until_goal = distance_to_goal
        reward = self._compute_reward(distance_to_goal)
        discrete_action = self._continuous_action_to_discrete(self.action)
#          Create a transition
        transition = (self.state, discrete_action, reward, next_state)
#         add a transition to the replay buffer
        self.buffer.append(transition)
        if len(self.buffer) > 400: 
            #Index some transitions from the buffer
            buffer_indices = np.random.choice(len(self.buffer),400,replace = False)
            transitions_list = [self.buffer[index] for index in buffer_indices]
            loss = self.dqn.train_q_target_net(transitions_list)
        if self.num_steps_taken % 500 == 0: 
        ### update the q_network to the q_target network
            self.dqn.update_Q_network()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        ## This line below may not work - need to find out how to reference a function in a different class with this class
        q_val = self.dqn.q_values_single_state(state) 
        # this is the greedy choice that chooses the action as to maximise Q
        discrete_action = np.argmax(q_val)
        action = self._discrete_action_to_continuous(discrete_action)
        return action
    

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))  
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.q_target_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss_BE(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()
    
    def q_values_single_state(self,state):
        
        state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
        q_value_tensor = self.q_network.forward(state_tensor).detach()
        
        return q_value_tensor
    
    def q_values_single_state_tar(self,state):
        
        state_tensor = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
        q_value_tensor = self.q_target_network.forward(state_tensor).detach()
        
        return q_value_tensor_tar
    
    def update_Q_network(self):
           # q_target is only updated when it goes through this function 
        q_net = self.q_network
        q_tar_net = self.q_target_network
        q_tar_net.load_state_dict(q_net.state_dict())
                    
        # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_target_net(self, minibatch):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss_BE_target_network(minibatch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()
    
    
    # Function to calculate the loss for a particular transition.
    def _calculate_loss_BE_target_network(self, minibatch):
#         pass
        # TODO
        # NOTE: when just training on a single example on each iteration, the NumPy array (and Torch tensor) still needs to have two dimensions: the mini-batch dimension, and the data dimension. And in this case, the mini-batch dimension would be 1, instead of 5. This can be done by using the torch.unsqueeze() function.
        minibatch_states,minibatch_actions,minibatch_rewards,minibatch_next_state = zip(*minibatch)
        minibatch_states_tensor = torch.tensor(minibatch_states,dtype=torch.float32)
        minibatch_actions_tensor = torch.tensor(minibatch_actions, dtype = torch.long)
        minibatch_rewards_tensor = torch.tensor(minibatch_rewards,dtype=torch.float32)
        minibatch_next_state_tensor = torch.tensor(minibatch_next_state,dtype=torch.float32)
        # always assumes youre passing a mini batch into a NN using torch so we need to 
        #package the data up as if it were a mini-batch, in this case it is size 1 so we
        # have to create that with [0,...]- index first element in mini-batch
        predicted_q_value_tensor = self.q_network.forward(minibatch_states_tensor).gather(dim=1,index= minibatch_actions_tensor.unsqueeze(-1)).squeeze(-1)
#         ## Bellman Equation
        # with double Q-learning - 1
        state_q_values = self.q_target_network.forward(minibatch_next_state_tensor).detach()
        argmax_action_tensor = torch.argmax(state_q_values, dim=1)
        state_action_double_q_values = self.q_network.forward(minibatch_next_state_tensor).gather(dim=1,index= argmax_action_tensor.unsqueeze(-1)).squeeze(-1)
        
        discounted_sum_fut_rewards_double = minibatch_rewards_tensor + (0.95*state_action_double_q_values)
        
# Calculate the loss
        loss= torch.nn.MSELoss()(predicted_q_value_tensor,discounted_sum_fut_rewards_double)
        return loss
      
    
    