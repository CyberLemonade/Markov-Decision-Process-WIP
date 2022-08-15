# Markov Decision Process (WIP)
Implementation of MDP in Java using Deep Q-Learning

## Classes:
1. Matrix - supplementary class for matrix operations over float types
2. Vector - subclass of Matrix class for easier handling of vectors
3. NeuralNetwork - implementation of a basic neural network for the MDP agent
4. Memory - supplementary class for easier storing of episode data
5. Agent - the MDP agent

## How it Works:
- The class to be used is Agent.
- Constructor for Agent expects the state_size and action_size as input, to initialize the neural network for predicting q-values.
- Agent.get_action(...) accepts a current state and outputs the action index taken. The action can either be random, or determined by the q-table.
- Agent.update_EXPLORATION_PROB() updates the probability with which the agent takes deterministic actions.
- Agent.add_replay(...) accepts and stores the tuple (s_t, a_t, r_{t+1}, s_{t+1}) in the memory buffer
- Agent.train() samples a random batch of BATCH_SIZE episodes and runs the neural network training on them
