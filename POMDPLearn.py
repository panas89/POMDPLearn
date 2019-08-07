import sys
from tqdm import tqdm #progress bar
sys.path
sys.path.append('./scripts/')
############################ MDP class ####################################

class MDP:
    """
    Markov Decision Process class, used to represent the components of an MDP.
    
    states: numpy array of unique states
    actions: numpy array of unique actions,
    rewards: numpy array of rewards
    T: numpy array of transition probabilities
    discount: scalar discount factor
    U: numpy array of utilities
    policy: numpy array of optimal policy
    horizon: scalar of horizon, number of epochs
    epsilon: float of error for V.I.
    learning_rate: float of learning rate of GD of IRL algorithm
    num_iter: number of iterations in EM algorithm (DBNs)
    stochastic: boolean for policy of IRL is stochastic
    adaptive: boolean to use an adaptive step size
    output: boolean to print output
    action_invariant: boolean to depict if transition matrix will be invariant of actions
    max_iter: number of iterations of IRL algorithm
    norm_option: integer to depict different normalization options
    isMDP: boolean to depict if object is MDP or POMDP
    """

    def __init__(self,states,actions,rewards=None,T=None,U=None,policy=None,discount=0.9,horizon=None,epsilon=0.01,learning_rate=0.01,num_iter=100,stochastic=True,
                 adaptive=True,output=False,action_invariant=False, max_iter=5,norm_option=-1):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.T = T
        self.discount = discount
        self.U = U
        self.policy = policy
        self.horizon = horizon
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.stochastic = stochastic
        self.adaptive = adaptive
        self.output = output
        self.action_invariant = action_invariant
        self.max_iter = max_iter
        self.norm_option = norm_option
        self.isMDP = True
        
        
    def trainMDP(self,MDPDataset):
        """Method to train and learn the componenets of an MDP."""
        
        print('Learning the transition matrix ...')
        ########################## Transition matrix
        self.T = self.LearnMDP_T(MDPDataset) # assigning T    
        print('Learning the rewards ...')
        ########################## reward function
        self.rewards = self.LearnMDP_rewards(MDPDataset)
        return;
    
    
    def LearnMDP_rewards(self,MDPDataset):
        """Method to learn the rewards of an MDP."""
        
        rs = maxEntrIRL(total_states=len(self.states),total_actions=len(self.actions),T=self.T,gamma=self.discount,epsilon=self.epsilon,trajectories = MDPDataset.stateTrajectories,learning_rate=self.learning_rate,N=self.num_iter,stochastic=self.stochastic,adaptive=self.adaptive,output=self.output)
        
        rs = mormalizeRewards(rs,option=self.norm_option)
        
        rewards = np.array([rs]*len(self.actions))
        
        return rewards
    
    def checkIfOctave(self,Dataset):
        """Method to check if enumeration begins at 1 as data will be input to octave"""
        
        if (0 in self.states or 0 in self.actions) and self.isMDP:
            Dataset.states_data = Dataset.states_data + 1
            Dataset.actions_data = Dataset.actions_data + 1#data input added 1, octave enumeration
        elif (0 in self.states or 0 in self.actions or 0 in self.observations) and not self.isMDP:
            Dataset.states_data = Dataset.states_data + 1
            Dataset.actions_data = Dataset.actions_data + 1
            Dataset.observations_data = Dataset.observations_data + 1
            
        return;
    
    
    def LearnMDP_T(self,MDPDataset):
        """Method to learn the transition matrix of an MDP.
        
        if not action invariant learning a transition matrix for each action otherwise learn one transition matrix used for all actions
        
        """
                
        #################### transition matrix
        if not self.action_invariant:
            T = []
            for i in self.actions:
                actions_data = (MDPDataset.actions_data.copy() + 1).astype(float)
                states_data = (MDPDataset.states_data.copy() + 1).astype(float)
                actions_data[actions_data!=i+1]=np.nan
                states_data[np.isnan(actions_data)]=np.nan
                components = getMDPComponents(intraLength=1, interLength=1, ns=[len(self.states)], horizon=self.horizon, data=states_data,max_iter=self.max_iter) 
                T.append(components[0][1])
        else:
            actions_data = MDPDataset.actions_data + 1
            states_data = MDPDataset.states_data + 1
            components = getMDPComponents(intraLength=1, interLength=1, ns=[len(self.states)], horizon=self.horizon, data=states_data, max_iter=self.max_iter) #data input does not start from zero do not add 1
            T = [components[0][1]]*len(self.actions)
            
        return np.array(T)
    
    
    def MDPSolve(self):
        """Method to solve an MDP using value iteration and return the utility of each state and a policy (action to state mapping). """
        U, p = valueIteration(total_states=len(self.states),total_actions=len(self.actions),
                              T=self.T,rs=self.rewards,gamma=self.discount,epsilon=self.epsilon,Output=self.output)  
        
        self.U, self.policy = U, p
        return;
    
    
    def policyExecution(self,initial_states):
        """Method to get state action pairs of optimal actions. """
        optimal_state_action_pairs = []
        
        for state in initial_states:
            episode_state_action_pairs = []
            episode_state = state
            for t in range(self.horizon):
                curr_action = int(self.policy[episode_state])
                episode_state_action_pairs.append((episode_state,curr_action))
                next_state = np.argmax(self.T[curr_action,episode_state,:]*self.U) # by multiplying T[action,state,:] only allowed transitions since rest are zero
                episode_state = next_state #we are at the new state and trying to find the optimal action
            optimal_state_action_pairs.append(episode_state_action_pairs)
            
        return np.array(optimal_state_action_pairs)
          
                                                 
        

class POMDP(MDP):
    """pomdpTest.action_rewards
    Partially-observable MDP class, used to represent the components of a POMDP.
    observations: numpy array of observations
    O: numpy array of observation matrix (states,observations)
    alpha_vectors: alpha vector objects
    state_rewards: numpy array of rewards of states
    action_rewards: numpy array of rewards of actions
    T_actionMDP: numpy array of transition matrix of action MDP
    isMDP: boolean of isMDP set to False
    solver: string of type solver to use for POMDP
    """

    def __init__(self,states,actions,observations,rewards=None,T=None,O=None,U=None,policy=None,alpha_vectors=None,discount=0.9,horizon=None,epsilon=0.01,learning_rate=0.01,num_iter=100,stochastic=True,adaptive=True,output=False,action_invariant=False,max_iter=5,norm_option=-1,solver = 'QMDP'):
        MDP.__init__(self,states,actions,rewards,T,U,policy,discount,horizon,epsilon,learning_rate,num_iter,stochastic,adaptive,output,action_invariant,max_iter,norm_option)
        self.observations = observations
        self.O = O
        self.alpha_vectors = alpha_vectors
        self.state_rewards = None
        self.action_rewards = None
        self.T_actionMDP = None
        self.isMDP = False
        self.solver = solver
        
   
    def trainPOMDP(self,POMDPDataset):
        """Method to train and learn the componenets of a POMDP."""
        
        ########################## Transition and observation matrix
        print('Learning the transition and observation matrix ...')
        self.T,self.O = self.LearnPOMDP_TO(POMDPDataset) # assigning T and O  
        
        ########################## reward function
        print('Learning the state rewards ...')
        #state MDP
        self.state_rewards = self.LearnMDP_rewards(POMDPDataset)
        
        ########## learn action MDP T and then rewards of actions
        print('Learning the transition matrix of the action MDP ... \n')
        self.T_actionMDP = self.LearnActionMDP_T(POMDPDataset)
        print('Learning the action rewards ... \n')
        self.action_rewards = self.LearnActionMDP_rewards(POMDPDataset)
        ###################### multiplicative model
        print('Using the multiplicative model to learn state action pair rewards ...')
        self.rewards = self.state_rewards*self.action_rewards
        
        return;
    
    def LearnPOMDP_TO(self,POMDPDataset):
        """Method to learn the transition matrix of an MDP.
         T and O learn simultaneously using a DBN network with the following structure.
         
         O  O           O
         |  |           |
         T--T   ....  --T
        """
        
        #self.checkIfOctave(POMDPDataset)
        
        #################### transition matrix
        if not self.action_invariant:
            T,O = [],[]
            for i in self.actions:
                actions_data = (POMDPDataset.actions_data + 1).astype(float)
                states_data = (POMDPDataset.states_data + 1).astype(float)
                observations_data = (POMDPDataset.observations_data + 1).astype(float)
                actions_data[actions_data!=i+1]=np.nan
                states_data[np.isnan(actions_data)]=np.nan
                observations_data[np.isnan(actions_data)]=np.nan
                states_obs_data = self.getStateObs(states_data,observations_data)
                components = getPOMDPComponents(intraLength=2, interLength=2, ns=[len(self.states),len(self.observations)], horizon=self.horizon, data=states_obs_data, max_iter=self.max_iter) 
                T.append(components[0][2])
                O.append(components[0][1])
        else:
            actions_data = POMDPDataset.actions_data + 1
            states_data = POMDPDataset.states_data + 1
            observations_data = POMDPDataset.observations_data + 1
            states_obs_data = self.getStateObs(states_data,observations_data)
            components = getPOMDPComponents(intraLength=2, interLength=2, ns=[len(self.states),len(self.observations)], horizon=self.horizon, data=states_obs_data, max_iter=self.max_iter)
            T = [components[0][2]]*len(self.actions)
            O = [components[0][1]]*len(self.actions)
            
        return np.array(T),np.array(O)
    
    def LearnActionMDP_T(self,POMDPDataset):
        """Method to learn the transition matrix of an MDP."""
        
        #self.checkIfOctave(POMDPDataset)
        
        #################### transition matrix
        if not self.action_invariant:
            POMDPDataset.actions_data.astype(float)
            T = []
            for i in self.actions:
                actions_data = (POMDPDataset.actions_data + 1).astype(float)
                states_data = (POMDPDataset.states_data + 1).astype(float)
                actions_data[actions_data!=i]=np.nan
                states_data[np.isnan(actions_data)]=np.nan
                components = getMDPComponents(intraLength=1, interLength=1, ns=[len(self.actions)], horizon=self.horizon, data=states_data, max_iter=self.max_iter) 
                T.append(components[0][1])
        else:
            actions_data = POMDPDataset.actions_data + 1
            states_data = POMDPDataset.actions_data + 1
            components = getMDPComponents(intraLength=1, interLength=1, ns=[len(self.actions)], horizon=self.horizon, data=states_data, max_iter=self.max_iter) #data input does not start from zero do not add 1
            T = [components[0][1]]*len(self.actions)
            
        return np.array(T)
    
    def LearnActionMDP_rewards(self,POMDPDataset):
        """Method to learn the rewards of an MDP."""
        
        ra = maxEntrIRL(total_states=len(self.actions),total_actions=len(self.actions),T=self.T_actionMDP,gamma=self.discount,epsilon=self.epsilon,trajectories = POMDPDataset.actionTrajectories,learning_rate=self.learning_rate,N=self.num_iter,stochastic=self.stochastic,adaptive=self.adaptive,output=self.output)
        
        ra = mormalizeRewards(ra,option=self.norm_option)
        
        rewards = np.array([ra]*len(self.states)) #reshaping to match element wise multiplication with state rewards
        
        return rewards.T #multiplicative model
    
    
    def POMDPSolve(self):
        """Method to solve the pomdp and obtain alpha vectors"""
        if self.solver == 'VI':
            self.alpha_vectors = pomdpVI(total_states=len(self.states),
                                         total_actions=len(self.actions),
                    total_obs=len(self.observations),discount=self.discount,
                    horizon=self.horizon,T=self.T,O=self.O,R=self.rewards,
                    epsilon=self.epsilon,pairwise=True)
        elif self.solver == 'QMDP':
            self.alpha_vectors = QMDP(total_states=len(self.states),
                                         total_actions=len(self.actions),
                                      discount=self.discount,T=self.T,
                          R=self.rewards,epsilon=self.epsilon,
                          stochastic=self.stochastic, Output=self.output)
            
        print('POMDP solved!')
        return;
    
    def getStateObs(self,states,observations):
        """Method to get numpy array where each column is state_0,obs_0,state_1,obs_1,...."""
        assert(states.shape==observations.shape)
        states_obs = np.zeros((states.shape[0],states.shape[1]*2))#shape states+obs
        counter = 0
        for i in range(0,states.shape[1]):
            states_obs[:,counter] = states[:,i]
            states_obs[:,counter+1] = observations[:,i]
            counter = counter+2
        return states_obs
    
    def getRecActions(self,POMDPDataset):
        """Method to get the recommended actions and updated beliefs"""
        
        return pomdpTest(init_beliefs=POMDPDataset.initial_beliefs,
                                      observations=POMDPDataset.observations_data,
                                      T=self.T,O=self.O,gamma=self.alpha_vectors)

###### SOS do not forget to reverse the effect of check if octave
        
class MDPDataset:
    """
    Class that stores as a datastructure the required data to learn an MDP.
    states_data: numpy array of states over horizon
    unique_states: numpy array of unique states
    action_data: numpy array of action data
    unique_actions: numpy array of unique actions
    horizon: number of epochs of finite horizon MDP
    stateTrajectories: list of state,action tuples 
    actionTrajectories: list of action,action tuples
    """
    
    def __init__(self,df):
        self.states_data = self.getColumnsData(df,keyword='state')
        self.unique_states = self.getUniqueVals(self.states_data)
        self.actions_data = self.getColumnsData(df,keyword='action')
        self.unique_actions = self.getUniqueVals(self.actions_data)
        self.horizon = self.states_data.shape[1] #number of states over tim is horizon
        self.stateTrajectories = self.getTrajectories(self.states_data,self.actions_data) #state_MDP--used for learn state rewards
        self.actionTrajectories = self.getTrajectories(self.actions_data,self.actions_data) #action_MDP--used for learn action rewards
        
    @staticmethod
    def getUniqueVals(vals):
        """Method that returns the unique states over time as a numpy array."""
        unique_vals = np.unique(vals)
        return unique_vals[~np.isnan(unique_vals)] #dropping nulls
    
    @staticmethod
    def getColumnsData(df,keyword):
        """Method that returns the dataframe data over time as a numpy array."""
        cols = [i for i in df.columns if keyword in i]
        return df.loc[:,cols].to_numpy()
    
    @staticmethod
    def getTrajectories(states,actions):
        """Method that computes state-action pairs and returns a list of episodes"""
        
        trajectories = []

        for state_seq,action_seq in tqdm(zip(states,actions)):
            episode = []
            for state,action in zip(state_seq,action_seq):
                episode.append((state,action))
            trajectories.append(episode)
        return trajectories
    
class POMDPDataset(MDPDataset):
    """ Class that represents the data required to learn a POMDP """
    
    def __init__(self,df):
        MDPDataset.__init__(self,df)
        self.observations_data = self.getColumnsData(df,'obs')
        self.unique_observations = self.getUniqueVals(self.observations_data)
        self.baseline_features = None
        self.initial_belief_model = None
        self.initial_beliefs = None
    

############################ Alpha vectors #################################
class AlphaVector:
    """
    Acknowledgements:
    - This implementation is by Patrick Emani:
      https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/alpha_vector.py
      
    Simple wrapper for an alpha vector, used for representing the value function for a POMDP as a piecewise-linear,
    convex function
    """
    def __init__(self, a, v):
        self.action = a
        self.v = v

    def copy(self):
        return AlphaVector(self.action, self.v)

############################################################################
############################ POMDP solvers #################################

from scipy.optimize import linprog
import numpy as np
from itertools import product
from tqdm import tqdm_notebook

def pomdpVI(total_states,total_actions,total_obs,discount,horizon,T,O,R,epsilon,pairwise):
    """
    Method: that solves a POMDP using value iteration (V.I.).
    
    - Book Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005. (for the algorithm).
    - This implementation is largely influenced by Patrick Emani:
      https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/alpha_vector.py (for alpha vectors class)
    - Most of the implementation is by by Patrick Emani:
      https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/value_iteration.py
    
    This method was also validated with Tony Cassandra pomdp-solve on two examples. The validation was succesful,
    the correct alpha vectors and actions where reported.
    http://www.pomdp.org/code/index.html
    
    In terms of POMDP size, this method cannot handle very large
    state space or observation space POMDPs. For that I suggest use the QMDP approximation method.
    
    Input:
    total_states
    total_actions
    total_obs
    discount: discount factor
    horizon: horizon length
    T: transition metrix
    O: Observation matrix
    R: reward matrix
    epsilon: error
    pairwise: boolean , if True pairwise pruning else use of Lark's pruning algorithm
    
    Output:
    Returns: the set of alpha vectors with the corresponding action.
    """

    states = total_states
    actions = total_actions
    observations = total_obs
    gamma = []

    t = T
    o = O
    r = R

    dummy = AlphaVector(a=-1, v=np.zeros(states))
    gamma.append(dummy)
    
    # start with 1 step planning horizon, up to horizon-length planning horizon
    for k in range(horizon):
        # new set of alpha vectors to add to set gamma
        gamma_k = []
        # Compute the new coefficients for the new alpha-vectors
        v_new = np.zeros(shape=(len(gamma), actions, observations, states))
        idx = 0
        for v in gamma:
            for u in range(actions):
                for z in range(observations):
                    for j in range(states):
                        for i in range(states):
                            # v_i_k * p(z | x_i, u) * p(x_i | u, x_j)
                            v_new[idx][u][z][j] += v.v[i] * o[u][i][z] * t[u][j][i]
            idx += 1
        # add (|A| * |V|^|Z|) alpha-vectors to gamma, |V| is |gamma_k|
        for u in range(actions):
            c = [p for p in product(list(range(idx)),repeat= observations)]
            for indices in c:  # n elements in c is |V|^|Z|
                    temp = np.zeros(states)
                    for i in range(states):
                        v = 0
                        for z in range(observations):
                            v += v_new[indices[z]][u][z][i]

                        temp[i] = discount * (r[u][i] + v)
                    gamma_k.append(AlphaVector(a=u, v=temp))
        
        ###### pruning
        if pairwise:
        ##### pairwise
            gamma_k = pairwisePrune(gamma_k,epsilon,states)
        else:
            ##### larks pruning
            if k>0:
                gamma_set = set()

                for i in gamma_k:
                    gamma_set.add(i)
                print(len(gamma_set))
                gamma_set = prune(gamma_set,total_states)
                gamma_k = [i for i in gamma_set]
        
        gamma = gamma_k

    return gamma

def QMDP(total_states,total_actions,discount,T,R,epsilon,stochastic,Output):
    """
    Method: that approximately solves a POMDP model.
    
    Acknowledgements:
    - This implementation is largely influenced by Massimiliano Patacchiola:
      https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html (for value itearation)
    - Book Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005. (for the algorithm).
    - This implementation is largely influenced by Patrick Emani:
      https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/alpha_vector.py (for alpha vectors class)
    
    Inputs:
    total_states
    total_actions
    discount: discount factor
    T: transition matrix
    R: reward function (matrix)
    epsilon: error
    stochastic: boolean to compute stochastic policy
    Output: boolean to print reward values of V.I.
    
    Output:
    Returns the alpha vectors of the POMDP.
    gamma: list of alpha vectors.
    """
    
    V,_ = valueIteration(total_states,total_actions,T,R,discount,epsilon,stochastic,Output)
    
    q = np.zeros((total_actions,total_states))
    
    for action in range(total_actions):
        for s in range(total_states):
            q[action,s] = R[action,s] + np.dot(V,T[action,s,:])
    
    gamma = []
    for i in range(q.shape[0]):
        gamma.append(AlphaVector(a=i, v=q[i,:]))

    return gamma

def QOMDP(total_states,total_actions,total_obs,discount,T,O,priors,R,R_O,epsilon,stochastic,Output):
    """
    Method: that approximately solves a POMDP model.
    
    Acknowledgements:
    - This implementation is largely influenced by Massimiliano Patacchiola:
      https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html (for value itearation)
    - Book Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005. (for the algorithm).
    - This implementation is largely influenced by Patrick Emani:
      https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/alpha_vector.py (for alpha vectors class)
    
    Inputs:
    total_states
    total_actions
    discount: discount factor
    T: transition matrix
    O: observation matrix
    priors: states' prior distribution
    R: reward function (matrix)
    R_O: rewards function that accounts for observations
    epsilon: error
    stochastic: boolean to compute stochastic policy
    Output: boolean to print reward values of V.I.
    
    Output:
    Returns the alpha vectors of the POMDP.
    gamma: list of alpha vectors.
    """
    
    V,_ = valueIteration(total_states,total_actions,T,R,discount,epsilon,stochastic,Output)
    
    q = []
    a_s = [] #actions
    print(V)
    for obs in range(total_obs):
        for action in range(total_actions):
            v = np.zeros(total_states)
            for s in range(total_states):
                v[s] = R_O[action,obs,s] + obsUtility(V,T,O,priors,total_states,action,s,obs)
            q.append(v)
            a_s.append(action)
                
    q = np.array(q)
    gamma = []
    for i in range(q.shape[0]):
        gamma.append(AlphaVector(a=a_s[i], v=q[i,:]))

    return gamma

def obsUtility(V,T,O,priors,total_states,action,s,obs):
    """
    Method that computes the utility of the QOMDP method.
    V:MDP value function
    T: transition matrix
    O: Observation matrix
    priors: states' prior distribution
    total_states: number of states
    action: action index
    s: state index
    obs: observation index
    
    Output:
    Returns the computed utility over states
    """
    res = 0
    for i in range(total_states):
        res += V[i]*T[action,s,i]*O[action,i,obs]/(np.sum([O[action,j,obs]*priors[j] for j in range(total_states)]))
    return res
############################################################################
############################ MDP solver ####################################

import numpy as np

#define the Bellman equation
def utility(v,r,gamma,total_actions,T,u):
    """
    Acknowledgements:
     - This implementation is largely influenced by Massimiliano Patacchiola:
      https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
     - Book by Russel and Norvig called Artificial Intelligence: A Modern Approach.
    
    Bellman equation:
    
    U(s)_t = R(s) + gamma * max_a T(s,a,s') * U(s)_{t-1}
    
    T(s,s',a) = T * v # v is the initial state
    
    Input:
    v: initial state 1x12 array
    r: rewards to states
    gamma: discount factor
    actions: array whos size represents the number of actions
    T: transition matrix for each actions its a 12x12x4 for each action there is the transition probability to the other actions
    u: t-1 step utlities
    
    Output:
    returns updated utility for state and each action
    """
    #actions
    actions = np.zeros(total_actions)
    
    for action in range(0,len(actions)):
        actions[action] = np.sum(u * np.dot(v,T[action,:,:]))
        
    return r + gamma * actions

def getCurrentState(total_states,s):
    """
    Method that returns current state:
    
    Acknowledgements:
     - This implementation is largely influenced by Massimiliano Patacchiola:
      https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
     - Book by Russel and Norvig called Artificial Intelligence: A Modern Approach.
    
    Input: 
    total_states: total number of states
    s: index of current state
    Output:
    v: vector of size 1xtotal_States
    """
    v = np.zeros((1,total_states),dtype='float')
    v[0,s] = 1.0
    return v


def valueIteration(total_states,total_actions,T,rs,gamma,epsilon,stochastic=False,Output=False):
    """
    Method that solves mdps using value iteration:
    
    Acknowledgements:
     - This implementation is largely influenced by Massimiliano Patacchiola:
      https://mpatacchiola.github.io/blog/2016/12/09/dissecting-reinforcement-learning.html
     - Book by Russel and Norvig called Artificial Intelligence: A Modern Approach.
     - This implementation is largely influenced by Yiru Lu's maxent implementation here:
      https://github.com/stormmax/irl-imitation/blob/master/mdp/value_iteration.py (stochastic policy computation)
    
    Input:
    total_states: total number of states
    total_actions: total number of actions
    T:transition matrix
    rs: reward function
    gamma: discount factor
    epsilon: error of terminating value iteration
    stochastic: boolean - True if you want to return a stochastic policy else a deterministic
    
    Output:
    returns updated utility for state and each action and
    the stochastic or deterministic policy.
    """
    #Utility vectors
    u = np.zeros(total_states)
    u1 = np.zeros(total_states)
    
    #iteration counter
    iteration = 0
    
    
    #list containing data of each iteration
    graph_list = []

    while True:
        delta = 0
        u = u1.copy()
        iteration += 1
        graph_list.append(u)
        for s in range(total_states):
            r = rs[:,s]
            v = getCurrentState(total_states,s)
            u1[s] = np.max(utility(v,r,gamma,total_actions,T,u))
            delta = max(delta, np.abs(u1[s] - u[s])) #Stopping criteria
        if delta < epsilon * (1 - gamma) / gamma:
            if Output:
                print("=================== FINAL RESULT ==================")
                print("Iterations: " + str(iteration))
                print("Delta: " + str(delta))
                print("Gamma: " + str(gamma))
                print("Epsilon: " + str(epsilon))
                print("===================================================")
                #print(u)
                print("===================================================")
            break
            
    if(stochastic):
        policy = np.zeros([total_states,total_actions])
        for s in range(total_states):
            r = rs[:,s]
            v = getCurrentState(total_states,s)
            values_actions = np.array(utility(v,r,gamma,total_actions,T,u))
            policy[s,:] = np.transpose(values_actions/np.sum(values_actions))
            
    else: #deterministic
        policy = np.zeros(total_states)
        for s in range(total_states):
            r = rs[:,s]
            v = getCurrentState(total_states,s)
            policy[s] = np.argmax(utility(v,r,gamma,total_actions,T,u))
             
    return u, policy

############################################################################
############################ Prunners ######################################

def pairwisePrune(gamma,epsilon,states):
    """
    Method: that prunes dominated alpha vectors in a pairwise fashion:
    
    Acknowledgements:
    - Smith, Trey. Probabilistic planning for robotic exploration. Carnegie Mellon University, 2007.
    
    Input:
    gamma: set of unpruned alpha vectors
    epsilon: error
    states: number of states
    
    Output:
    Returns: set of pruned alpha vectors
    """
    
    gamma_dot = [AlphaVector(a=-1, v=np.zeros(states))]
    
    for alpha in gamma:
        beta_dominates = False
        
        for beta in gamma_dot:
            delta = beta.v - alpha.v
            
            if np.sum((delta > epsilon)*1)== len(alpha.v):
                beta_dominates = True
                continue
                
        if beta_dominates:
                continue
                
        counter_beta = 0
        for beta in gamma_dot:
            delta = beta.v - alpha.v
            
            if np.sum((delta < epsilon)*1) == len(alpha.v):
                del gamma_dot[counter_beta]
            counter_beta += 1
            
        gamma_dot.append(alpha)
        
    return gamma_dot

def prune(gamma, n_states):
    
    """
    Acknowledgements:
    - This implementation is by Patrick Emani:
      https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/value_iteration.py
      
    Remove dominated alpha-vectors using Lark's filtering algorithm
    :param n_states
    :return:
    """
    # parameters for linear program
    delta = 0.0000000001
    # equality constraints on the belief states
    A_eq = np.array([np.append(np.ones(n_states), [0.])])
    b_eq = np.array([1.])

    # dirty set
    F = gamma.copy()
    # clean set
    Q = set()

    for i in range(n_states):
        max_i = -np.inf
        best = None
        for av in F:
            if av.v[i] > max_i:
                max_i = av.v[i]
                best = av
        Q.update({best})
        F.remove(best)
    while F:
        av_i = F.pop()  # get a reference to av_i
        F.add(av_i)  # don't want to remove it yet from F
        dominated = False
        for av_j in Q:
            c = np.append(np.zeros(n_states), [1.])
            A_ub = np.array([np.append(-(av_i.v - av_j.v), [-1.])])
            b_ub = np.array([-delta])

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
            if res.x[n_states] > 0.0:
                # this one is dominated
                dominated = True
                F.remove(av_i)
                break

        if not dominated:
            max_k = -np.inf
            best = None
            for av_k in F:
                b = res.x[:3]
                #print(b)
                #print(av_k.v)
                v = np.dot(av_k.v, b)
                if v > max_k:
                    max_k = v
                    best = av_k
            F.remove(best)
            if not check_duplicate(Q, best):
                Q.update({best})
    gamma = Q
    return gamma

def check_duplicate(a, av):
    """
    Acknowledgements:
    - This implementation is by Patrick Emani:
      https://github.com/pemami4911/POMDPy/blob/master/pomdpy/solvers/value_iteration.py
    
    Check whether alpha vector av is already in set a
    :param a:
    :param av:
    :return:
    """
    for av_i in a:
        if np.allclose(av_i.v, av.v):
            return True
        if av_i.v[0] == av.v[0] and av_i.v[1] > av.v[1]:
            return True
        if av_i.v[1] == av.v[1] and av_i.v[0] > av.v[0]:
            return True
        
############################################################################
############################ Belief Update #################################

def beliefUpdate(T,O,init_belief,action,observation):
    """
    Method that updates the belief of an agent
    Input:
    T:transition matrix
    O:Observation matrix
    init_belief: the initial belief of the agent
    action: index of action performed by the agent
    observation: index of observation observed
    
    update belief equation (latex format): \alpha * P(observation|s') * \sum_{s} P(s'|s,a)b(s)
    \alpha = \frac{1}{\sum P(observation|s') * \sum_{s} P(s'|s,a)b(s)}
    
    Output: updated belief of dimension (1xtotal_States)
    """
    if(np.isnan(observation)):
        observation = -1
    update = O[action,:,int(observation)]*np.dot(T[action,:,:],init_belief)
    return update/np.sum(update)

############################################################################
############################ Action Update #################################

def getAction(belief,gamma):
    """
    Method that returns optimal action
    Input:
    belief:initial beleif
    gamma:set of alpha vectors
    
    Output: return optimal action
    """
    alphas = np.array([i.v for i in gamma])
    actions = np.array([i.action for i in gamma])

    alpha_max = np.argmax(np.dot(belief,alphas.T))
    
    return actions[alpha_max]

############################################################################
############################ POMDP Test ####################################

def pomdpTest(init_beliefs,observations,T,O,gamma):
    """
    Method: that uses the beliefs and observations over time,
    to suggest the best actions for an agent over time.
    
    Input:
    init_beliefs: initial beliefs of all cases (Number of cases X Number of states)
    observations: all observations (Number of cases X Lenght of horizon)
    gamma: pruned alpha vectors
    
    Output:
    Return: the reccommended actions by the POMDP and the updated beliefs over time.
    """
    
    allCasesRecActions = []
    allCasesBeliefs = []
    for case in range(observations.shape[0]):
        observation = observations[case,:]
        init_belief = init_beliefs[case,:]
        
        recActions = []
        beliefs = []
        
        for i in range(observations.shape[1]):
            if(i==0):
                action = getAction(init_belief,gamma)
                recActions.append(action)
                beliefs.append(init_belief)
                belief = beliefUpdate(T,O,init_belief,action,observation[i])
            else:
                action = getAction(belief,gamma)
                recActions.append(action)
                beliefs.append(belief)
                belief = beliefUpdate(T,O,belief,action,observation[i])
        
        beliefs.append(belief)
        recActions.append(getAction(belief,gamma))
        
        allCasesRecActions.append(recActions)
        allCasesBeliefs.append(beliefs)
        
    return np.array(allCasesRecActions), np.array(allCasesBeliefs)

############################################################################
############################ MaxEnt IRL ####################################

def maxEntrIRL(total_states,total_actions,T,gamma,epsilon,trajectories,learning_rate,N,stochastic,adaptive,output=False):
    """
    Method: that computes the rewards of adeterministic MDP using the Maximum Entropy IRL method by
    Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning.
    
    Acknowledgements:
     - This implementation is largely influenced by Yiru Lu's maxent implementation here:
      https://github.com/stormmax/irl-imitation/blob/master/maxent_irl.py
     - This implementation is largely influenced by Matthew Alger's maxent implementation here:
      https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py
    Input:
    total_states
    total_actions
    T: transition matrix
    gamma: discount factor
    epsilon: error
    trajectories: nxT, n: number of instances of trajectories, T: total number of time steps per trajectory, 
    partial trajectories are also supported
    learning_rate: integer representing the step size of gradient descent
    N: number of iterations of gradient descent
    stochastic: stochastic value iteration computation (i.e., proabbilities of policy)
    adaptive: boolean to use an adaptive step size or not
    
    Output:
    Returns the rewards of the MDP.
    """
    
    #assuming identity matrix feature map
    feat_map = np.eye(total_states)
    
    #initialize weights
    thetas = np.random.uniform(size=(total_states,))
    
    #estimate feature expectations
    feat_exp = featExpectations(total_states,trajectories,feat_map)
    
    #gradient descent 
    for i in range(N):
        
        #compute reward function
        rs = np.dot(feat_map, thetas)
        rs = np.array([rs for i in range(total_actions)]) #for each action
        #compute policy
        _,policy = valueIteration(total_states,total_actions,T,rs,gamma,epsilon,stochastic=True,Output=False)
        
        #compute state visitation frequencies
        svf = compute_state_visit_freqs(total_states,total_actions,T, trajectories, policy,stochastic=True)
        
        #compute gradient
        grad = feat_exp - np.dot(feat_map,svf)
        
        if adaptive:
            #adaptive learning rate
            if(i==0):
                grad_old = grad
                t_old = np.random.uniform(0, 1)
                #update thetas
                thetas += learning_rate*grad
            else:
                learning_rate, t_old = updateLearning_rate(grad,grad_old,t_old)

                grad_old = grad

                #update thetas
                thetas += learning_rate*grad
        else:
            thetas += learning_rate*grad
        
    rs = np.dot(feat_map,thetas)
    
    if output:
        print("Reward values per state: ",list(rs))
        print("Normalised Reward values per state: ",[float(i)/sum(np.absolute(rs)) for i in rs])
        print("Normalised (0_1 range) Reward values per state: ",[float(i - min(rs))/(max(rs)-min(rs)) for i in rs])
        print("Normalised (-1_1 range) Reward values per state: ",[2*float(i - min(rs))/(max(rs)-min(rs)) -1 for i in rs])
    
    return rs

def mormalizeRewards(rs,option):
    """Method to normalize a vector of rewards"""
    
    if option == -1:  #-1 to 1
        rs = [2*float(i - min(rs))/(max(rs)-min(rs)) -1 for i in rs]
    elif option == 0:  #0 to 1
        rs = [float(i - min(rs))/(max(rs)-min(rs)) for i in rs]
    elif option == 1: #sum of obsolute values divide each value
        [float(i)/sum(np.absolute(rs)) for i in rs]
    else:
        pass
    return rs
    
def updateLearning_rate(grad,grad_old,t_old):
    
    """
    Method: that compute an updated elarning rate using a sigmoid function and quarantees convergence in cases
    of a unique solution.
    
    Aknowledgements:
    - Implemented using:
    S. Klein, J. P. W. Pluim, M. Staring, and M. A. Viergever, “Adaptive stochastic
    gradient descent optimisation for image registration,” in International Journal
    of Computer Vision, 2009.
    
    Input:
    grad: current gradient value
    grad_old: previous iteration's gradient value
    t_old: non-linear max() method result of previous iteration.
    
    Output:
    Returns the update value of the learning rate.
    """
    
    alpha = 1
    Alpha = 1
    f_min = -0.5
    f_max = 1
    omega = 0.1
    
    x = -np.dot(grad,grad_old)
    t = np.max([t_old + sigmoid(x,alpha,Alpha,f_min,f_max,omega),0])
    
    learning_rate = alpha/((t + Alpha)**alpha)
    
    return learning_rate,t

def sigmoid(x,alpha,Alpha,f_min,f_max,omega):
    """
    Method: that computes the sigmoid
    
    Aknowledgements:
    - Implemented using:
    S. Klein, J. P. W. Pluim, M. Staring, and M. A. Viergever, “Adaptive stochastic
    gradient descent optimisation for image registration,” in International Journal
    of Computer Vision, 2009.
    
    Input: user defined parameters
    
    Output:
    Returns the sigmoid value.
    """
    
    return f_min + (f_max - f_min)/(1 - (f_max/f_min)*(np.exp(-x/omega)))
    
def featExpectations(total_states,trajectories,feat_map):
    """
    Method that computes the expected features using the existing trajectories
    Input:
    total_states: total number of states
    trajectories: all of the demonstrations of the experts
    feat_map: the identity matrix representing as feature vectors in each row the state.
    
    Output:
    feat_exp: the expected feature values, each index represents a state.
    """
    feat_exp = np.zeros(total_states)
    for episode in trajectories:
        for step in episode:
            feat_exp += feat_map[step[0],:]
    
    return feat_exp/len(trajectories)

def compute_state_visit_freqs(total_states,total_actions,T, trajectories, policy,stochastic):
    
    """
    Method: that computes the state visitation frequencies from the experts' trajectories.
    
    Acknowledgements:
     - This implementation is largely influenced by Yiru Lu's maxent implementation here:
      https://github.com/stormmax/irl-imitation/blob/master/maxent_irl.py
     - This implementation is largely influenced by Matthew Alger's maxent implementation here:
      https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py
    
    Input:
    total_states
    total_actions
    T: transition matrix
    trajectories: experts' demonstrations
    policy: stochastic policy
    stochastic: boolean if policy is stochastic or deterministic
    
    Output:
    mu: Returns the expected visitation frequency.
    """
    
    #compute the largest episode length (suitable for cases of partial trajectories)
    episode_length = np.max([len(i) for i in trajectories])
    
    #mu_t(s) is the probability of visiting s at t
    mu = np.zeros([total_states,episode_length])
    
    #compute first visitation frequency.Sum up at first state of each trajectory and average
    for episode in trajectories:
        mu[episode[0][0],0] +=1
    mu[:,0] = mu[:,0]/len(trajectories)
    
    #dynamic programming algorithm - compute state visit freqs
    for s in range(total_states):
        for t in range(episode_length-1):
            if stochastic:
                mu[s,t+1] = np.sum([np.sum([mu[s_dot, t]*policy[s_dot,a]*T[a,s,s_dot] for s_dot in range(total_states)]) for a in range(total_actions)])
            else:
                mu[s,t+1] = np.sum([mu[s_dot, t]*T[policy[s_dot],s,s_dot] for s_dot in range(total_states)])
    
    return np.sum(mu,axis=1)

############################################################################
##################### BNT Obs and Trans models #############################

from oct2py import octave

def getPOMDPComponents(intraLength, interLength, ns, horizon, data, max_iter, terminal_state=False, terminal_state_loc=(0,0)):
    """
    Method that takes as input the definition of a Dynamic Bayesian Network(DBN)
    it then creates a DBN and trains it with existing data to learn the transition
    and observation models of a POMDP.
    Training algorithm is the expectation maximization algorithm that also 
    handles Missing data. This method is using the BNT toolbox and the oct2py 
    package to communicate the octave kernel.
    
    DBN example:
    
        (1^t=0)----------------(1^t=1) -----....
         |     \                |     \
        (2^t=0) (3^t=0)        (2^t=1) (3^t=1)
    
    Input:
    intraLength: number of nodes that consitute the DBN slice at time t
    interLength: number of nodes that are temporal
    ns: number of discrete categories each node of the DBN has
    horizon: number of time slices
    data: the data, numpy array, data have to be arranged in specific order.
    First column should correspond to the first defined node up to the last for
    each time slice. columns: [(1^t=0)(2^t=0)(3^t=0)(1^t=1)(2^t=1)(3^t=1)]
    
    max_iter: number of iterations the EM will run. If convergence occurs earlier
    the algorithm terminates.
    Terminal_state: if there is a terminal state indicate the node index 
    and state index.
    
    Output:
    Returns the learned components of the DBN in sequential oder
    """
    
    out = octave.dbnPOMDP_BNT(intraLength, interLength, ns, horizon, data, max_iter)
    
    if terminal_state:
        DBNcomponents = out.tolist()
        terminal_states_distr = np.ones(ns[terminal_state_loc[0]])/ns[terminal_state_loc[0]]
        terminal_states_distr[terminal_state_loc[1]] = 0 #cannot initiate from absorbing state
        terminal_states_distr = terminal_states_distr/np.sum(terminal_states_distr) #normalized
        DBNcomponents[0][terminal_state_loc[0]+intraLength] \
        [terminal_state_loc[1],:] = terminal_states_distr
        return DBNcomponents
    else:
        return out.tolist()

    
def getMDPComponents(intraLength, interLength, ns, horizon, data, max_iter, terminal_state=False, terminal_state_loc=(0,0)):
    """
    Method that takes as input the definition of a Dynamic Bayesian Network(DBN)
    it then creates a DBN and trains it with existing data to learn the transition
    and observation models of a POMDP.
    Training algorithm is the expectation maximization algorithm that also 
    handles Missing data. This method is using the BNT toolbox and the oct2py 
    package to communicate the octave kernel.
    
    DBN example:
    
        (1^t=0)----------------(1^t=1) -----....
    
    Input:
    intraLength: number of nodes that consitute the DBN slice at time t
    interLength: number of nodes that are temporal
    ns: number of discrete categories each node of the DBN has
    horizon: number of time slices
    data: the data, numpy array, data have to be arranged in specific order.
    First column should correspond to the first defined node up to the last for
    each time slice. columns: [(1^t=0)(2^t=0)(3^t=0)(1^t=1)(2^t=1)(3^t=1)]
    
    max_iter: number of iterations the EM will run. If convergence occurs earlier
    the algorithm terminates.
    Terminal_state: if there is a terminal state indicate the node index 
    and state index.
    
    Output:
    Returns the learned components of the DBN in sequential oder
    """
    
    out = octave.dbnMDP_BNT(intraLength, interLength, ns, horizon, data, max_iter)
    
    if terminal_state:
        DBNcomponents = out.tolist()
        terminal_states_distr = np.ones(ns[terminal_state_loc[0]])/ns[terminal_state_loc[0]]
        terminal_states_distr[terminal_state_loc[1]] = 0 #cannot initiate from absorbing state
        terminal_states_distr = terminal_states_distr/np.sum(terminal_states_distr) #normalized
        DBNcomponents[0][terminal_state_loc[0]+intraLength] \
        [terminal_state_loc[1],:] = terminal_states_distr
        return DBNcomponents
    else:
        return out.tolist()    

def treat_sparsity(mat, value, axis=0):

    """
    Method that smoothes out sparse matrix using a small value.

    Returns matrix.
    """
    mat[mat==0] = value

    if axis == 0:
        #normalize
        for i in range(mat.shape[0]):
            mat[i,:] = mat[i,:]/np.sum(mat[i,:])

        return mat
    else:
        #normalize
        for i in range(mat.shape[1]):
            mat[:,i] = mat[:,i]/np.sum(mat[:,i])

        return mat
