# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np
from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.policy=util.Counter()
        for i in self.mdp.getStates():
          self.policy[i]='exit'
        self.qvalues=util.Counter()
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # print ('jhdbfmdsbMHCBWMDFBC MD')
        # print(self.iterations)
        for k in range (self.iterations):
          # print('22222')
          states=self.mdp.getStates()
          # print('jwbjhdfbdmnbf')
          v_copy=self.values.copy()
          # print('hey')
          for s in states:
            # print('lol')
            if  False: #s==wall:
              pass
            # print('V',gamma*np.tile(value[0,:],(num_actions,1)))
            # print('R',R[s]*np.ones((num_actions,num_states)))
            # print(R[s]*np.ones((num_actions,num_states))+ gamma*np.tile(value[0,:],(num_actions,1)))
            else: 
              
              actions=self.mdp.getPossibleActions(s)
              vecSums= util.Counter()  # np.zeros(len(actions))

              for a in actions:
                sumv=0
                s_reachable=[]
                Tlist=self.mdp.getTransitionStatesAndProbs(s,a)
                for i in range(len(Tlist)):
                  s_reachable.append(Tlist[i][0])
                for s_next in s_reachable:
                  if False: #s_next==wall:
                    pass
                  else:
                    # Tlist=self.mdp.getTransitionStatesAndProbs(s,a)
                    for i in range(len(Tlist)):
                      if Tlist[i][0]==s_next:
                        T=Tlist[i][1]
                    # print('22222',T)
                    R=self.mdp.getReward(s,a,s_next)
                    sumv+= T*(R+ self.discount*v_copy[s_next])       #T[s,a,s_next]*(R[s]+ gamma*value[0,s_next])
                vecSums[a]=sumv
                self.qvalues[s,a]=sumv
              self.policy[s]=vecSums.argMax()
              self.values[s]= vecSums[self.policy[s]] # max(sum(T[s,:,:]*(R[s]*np.ones((num_actions,num_states))+ gamma*np.tile(value[0,:],(num_actions,1))  ),1))



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        
        sumQ=0
        s_reachable=[]
        Tlist=self.mdp.getTransitionStatesAndProbs(state,action)
        for i in range(len(Tlist)):
          s_reachable.append(Tlist[i][0])
        for s_next in s_reachable:
          # Tlist=self.mdp.getTransitionStatesAndProbs(state,action)
          # print('Tlist',Tlist)
          for i in range(len(Tlist)):
            if Tlist[i][0]==s_next:
              # print('cool')
              T=Tlist[i][1]
          R=self.mdp.getReward(state,action,s_next)
          if self.iterations==0:
            sumQ+=T*R
          else:
            sumQ+=T*(R+self.discount*self.values[s_next])
        return sumQ
       
        # return self.qvalues[state,action]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
          return None
        if len(list(self.policy.keys())) == 0:
          return 'exit'

        return self.policy[state]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.Statecounter=0
        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
        # self.states=self.mdp.getStates()

    def runValueIteration(self):
        # print('start')

        for k in range (self.iterations):
          # print('22222')
          states=self.mdp.getStates()
          # print('jwbjhdfbdmnbf')
          v_copy=self.values.copy()
          # print('hey',self.Statecounter)
          # for s in states[self.Statecounter]:
          s=states[self.Statecounter]
          # print(states)
          # print('s',s)
          # print('flag')
          # print('lol')
          if  False: #s==wall:
            pass
          # print('V',gamma*np.tile(value[0,:],(num_actions,1)))
          # print('R',R[s]*np.ones((num_actions,num_states)))
          # print(R[s]*np.ones((num_actions,num_states))+ gamma*np.tile(value[0,:],(num_actions,1)))
          else: 
            
            actions=self.mdp.getPossibleActions(s)
            vecSums= util.Counter()  # np.zeros(len(actions))
            # print('gggg')
            for a in actions:
              sumv=0
              s_reachable=[]
              Tlist=self.mdp.getTransitionStatesAndProbs(s,a)
              for i in range(len(Tlist)):
                s_reachable.append(Tlist[i][0])
              for s_next in s_reachable:
                if False: #s_next==wall:
                  pass
                else:
                  # Tlist=self.mdp.getTransitionStatesAndProbs(s,a)
                  for i in range(len(Tlist)):
                    if Tlist[i][0]==s_next:
                      T=Tlist[i][1]
                  # print('22222',T)
                  R=self.mdp.getReward(s,a,s_next)
                  sumv+= T*(R+ self.discount*v_copy[s_next])       #T[s,a,s_next]*(R[s]+ gamma*value[0,s_next])
              vecSums[a]=sumv
              self.qvalues[s,a]=sumv
            self.policy[s]=vecSums.argMax()
            self.values[s]= vecSums[self.policy[s]] # max(sum(T[s,:,:]*(R[s]*np.ones((num_actions,num_states))+ gamma*np.tile(value[0,:],(num_actions,1))  ),1))
            # print('jwbjdbwej')
          self.Statecounter+=1
          if self.Statecounter>len(states)-1:
            self.Statecounter=0



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.

        """

        self.theta = theta
        self.pq=util.PriorityQueue()
        self.predecessors=util.Counter()
        
        # self.mdp=mdp
       

        states=mdp.getStates()
        for s in states:
          
          actions=mdp.getPossibleActions(s)
          for a in actions:
            # print(mdp.getTransitionStatesAndProbs(s,a))
            T=mdp.getTransitionStatesAndProbs(s,a)
            # print(T[0])
            (s_next,p)=T[0]
            if self.predecessors[s_next]==0:
              # print('hi')
              self.predecessors[s_next]=set()
            if p != 0:
              # print(self.predecessors[s_next])
              self.predecessors[s_next].add(s)

        ValueIterationAgent.__init__(self, mdp, discount, iterations)
        
        
        

        
  

    def runValueIteration(self):
      # pq=PriorityQueue()
      # states=self.mdp.getStates()
      # for s in states:
      #   if self.mdp.isTerminal(s):
      #     pass
      #   else:
      #     actions=mdp.getPossibleActions(s)
      #     qmax=-np.inf 
      #     for a in actions:
      #       q=self.getQValue(s,a)
      #       if q>qmax:
      #         qmax=q
      #     diff=abs(self.values[s]-qmax)
      #     pq.push(s,-diff)
      states=self.mdp.getStates()
      for s in states:
          if self.mdp.isTerminal(s):
            pass
          else:
            actions=self.mdp.getPossibleActions(s)
            qmax=-np.inf 
            for a in actions:
              q=self.getQValue(s,a)
              if q>qmax:
                qmax=q
            diff=abs(self.values[s]-qmax)
            self.pq.push(s,-diff)
      

      for k in range(self.iterations):
        if self.pq.isEmpty():
          break
        s=self.pq.pop()
        if not self.mdp.isTerminal(s):
          v_copy=self.values.copy()
          actions=self.mdp.getPossibleActions(s)
          vecSums= util.Counter()  # np.zeros(len(actions))
          # print('gggg')
          for a in actions:
            sumv=0
            s_reachable=[]
            Tlist=self.mdp.getTransitionStatesAndProbs(s,a)
            for i in range(len(Tlist)):
              s_reachable.append(Tlist[i][0])
            for s_next in s_reachable:
              if False: #s_next==wall:
                pass
              else:
                # Tlist=self.mdp.getTransitionStatesAndProbs(s,a)
                for i in range(len(Tlist)):
                  if Tlist[i][0]==s_next:
                    T=Tlist[i][1]
                # print('22222',T)
                R=self.mdp.getReward(s,a,s_next)
                sumv+= T*(R+ self.discount*v_copy[s_next])       #T[s,a,s_next]*(R[s]+ gamma*value[0,s_next])
            vecSums[a]=sumv
            self.qvalues[s,a]=sumv
          self.policy[s]=vecSums.argMax()
          self.values[s]= vecSums[self.policy[s]] # max(sum(T[s,:,:]*(R[s]*np.ones((num_actions,num_states))+ gamma*np.tile(value[0,:],(num_actions,1))  ),1))
        for pred in self.predecessors[s]:
          actions=self.mdp.getPossibleActions(pred)
          qmax=-np.inf 
          for a in actions:
            q=self.getQValue(pred,a)
            if q>qmax:
              qmax=q
          diff=abs(self.values[pred]-qmax)
          if diff > self.theta:
            self.pq.update(pred,-diff)




      

        

