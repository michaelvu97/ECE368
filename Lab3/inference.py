import numpy as np
import graphics
import rover
from rover import Distribution

def observe(observation_model_fixed):
    return lambda state, observation: 1 if observation is None else observation_model_fixed(state)[observation]

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps 

    observation_model = observe(observation_model)
    
    # TODO: Compute the forward messages
    print("forwards")
    for z in forward_messages[0]:
        # Mask using the 0th observation
        prob = forward_messages[0][z] * observation_model(z, observations[0])
        forward_messages[0][z] = prob

    # Remove 0-probability keys
    keys_to_remove = [x for x in forward_messages[0].keys() if not x in forward_messages[0].get_probable_keys()]
    for key in keys_to_remove:
        del forward_messages[0][key]
    forward_messages[0].renormalize()

    for i in range(num_time_steps):
        if i == 0:
            continue

        forward_messages[i] = Distribution()
        for z_curr in all_possible_hidden_states:
            observation_prob = observation_model(z_curr, observations[i])
            sum_prob = 0.0

            for z_prev in forward_messages[i - 1].get_probable_keys():
                sum_prob += forward_messages[i - 1][z_prev] * transition_model(z_prev)[z_curr]
            forward_messages[i][z_curr] = sum_prob * observation_prob
        forward_messages[i].renormalize()

    # TODO: Compute the backward messages
    backward_messages[num_time_steps - 1] = Distribution()
    for z in all_possible_hidden_states:
        backward_messages[num_time_steps - 1][z] = 1.0
    backward_messages[num_time_steps - 1].renormalize()

    print("backwards")
    for i in reversed(range(num_time_steps - 1)):
        backward_messages[i] = Distribution()

        for z_curr in all_possible_hidden_states:
            prob = 0.0
            for z_next in backward_messages[i + 1].keys():
                obs_prob = observation_model(z_next, observations[i + 1])
                prob += backward_messages[i + 1][z_next] * obs_prob * transition_model(z_curr)[z_next]
            if prob > 0:
                backward_messages[i][z_curr] = prob
        backward_messages[i].renormalize()
    
    # TODO: Compute the marginals 
    print("marginals")
    for i in range(num_time_steps):
        marginals[i] = Distribution()
        for z in forward_messages[i].get_probable_keys():
            prob = forward_messages[i][z] * backward_messages[i][z]
            if prob > 0:
                marginals[i][z] = prob
        marginals[i].renormalize()

    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    observation_model = observe(observation_model)
    N = len(observations)
    w = [None] * N
    phi = [None] * N

    # Initialize w
    w[0] = prior_distribution
    for z in prior_distribution:
        w[0][z] = np.log(w[0][z] * observation_model(z, observations[0]))

    # Recursion
    for i in range(1, N):
        w[i] = Distribution()
        phi[i - 1] = Distribution()
        for z_curr in all_possible_hidden_states:
            log_obs_prob = np.log(observation_model(z_curr, observations[i]))

            maxval = -1 * np.inf
            maxarg = None

            for z_prev in w[i - 1]:
                val = np.log(transition_model(z_prev)[z_curr]) + w[i - 1][z_prev]
                if val > maxval:
                    maxval = val
                    maxarg = z_prev

            w[i][z_curr] = maxval + log_obs_prob
            phi[i - 1][z_curr] = maxarg

    # Backtrack to find the most probable path
    print("backtracking")
    estimated_hidden_states = [None] * N
    maxval = -1 * np.inf
    maxarg = None

    for z, prob in w[N - 1].items():
        if prob > maxval:
            maxval = prob
            maxarg = z
    
    estimated_hidden_states[N - 1] = maxarg

    for i in reversed(range(0, N - 1)):
        estimated_hidden_states[i] = phi[i][estimated_hidden_states[i + 1]]

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')

    prior_distribution = rover.initial_distribution()

   
    timestep = 30
    print("Most likely parts of marginal at time %d:" % (timestep))
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])
  
    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    # estimated_states = [None]*num_time_steps
    # marginals = [None]*num_time_steps
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()
        
