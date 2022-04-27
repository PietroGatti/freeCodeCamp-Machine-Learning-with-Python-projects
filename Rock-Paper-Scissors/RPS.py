# We use the Q-learning algorithm to build the 'player' function.
# The actions are the possible plays 'R'ock, 'P'aper, 'S'cissors.
# The states are triples (player last two plays, opponennt last play).
# The rewards are respectively 1, 0, -1 if the player won, tied or lost after
# his last action.
# We use a variable EPSILON to regulate exploration and exlpoitation. It is
# reduced after each game according to DECAY_EPSILON.

# The 'player' function rebuilds the matrix Q each time it is called in the
# function 'play' during the number of games specified. This must be big enough.

import numpy as np
import random

ACTIONS = 'RPS'
REWARD = {'RR': 0, 'RP': -1, 'RS': 1, 'PR': 1, 'PP': 0, 'PS': -1, 'SR': -1,     
          'SP': 1, 'SS': 0}
STATES = list(REWARD.keys())
STATES = [state + action for state in STATES for action in ACTIONS]

Q = np.zeros((len(STATES), len(ACTIONS)))

# the following choice of parameters yielded the best results during testing.

EPSILON_initial = 0.95
EPSILON = EPSILON_initial
DECAY_EPSILON = 0.002

ALPHA = 0.5
GAMMA = 0.005

def player(prev_play, opponent_history=[], my_history=[],
           Q=np.zeros((len(STATES), len(ACTIONS))), with_log=True):

  global EPSILON

  if prev_play:
    opponent_history.append(prev_play)
  
  # If there is no previous play by the opponent it means we are facing a new
  # opponent, so we reinitialize Q and EPSILON and reset the history.
  else:
    Q[:] = np.zeros((len(STATES), len(ACTIONS)))
    EPSILON = EPSILON_initial
    opponent_history[:] = []
    my_history[:] = []

  # For the first 4 games there are no states to consider, so the player 
  # chooses a move at random.
  if len(opponent_history) < 4:
    action = random.choice(ACTIONS)
    
  # The following is the main code of the player function. We update the matrix
  # Q according to the Q-learning formula.
  else:
    current_state = my_history[-2] + my_history[-1] + opponent_history[-1]
    prev_state = my_history[-3] + my_history[-2] + opponent_history[-2]
    my_last_action = my_history[-1]
    opponent_last_action = opponent_history[-1]

    s = STATES.index(prev_state)
    ns = STATES.index(current_state)
    a = ACTIONS.index(my_last_action)
    r = REWARD[my_last_action + opponent_last_action]

    Q[s, a] = Q[s, a] + ALPHA * (r + GAMMA * np.max(Q[ns, :]) - Q[s, a])
   
    # Exploration vs exlpoitation: if the random number is smaller than EPSILON
    # the player explores a random action, else it chooses the best action
    # according to Q.
    if np.random.uniform(0, 1) < EPSILON:
      action = random.choice(ACTIONS)
  
    else:
      action_index = np.argmax(Q[ns, :])
      action = ACTIONS[action_index]

  # EPSILON gets reduced after each game
  if EPSILON > 0:
      EPSILON -= DECAY_EPSILON

  # player records the action it is going to perform before returning it
  my_history.append(action)

  return action
    
    

