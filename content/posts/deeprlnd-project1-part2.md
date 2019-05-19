---
title: "Udacity DeepRL project 1: Navigation - Part (2/3)"
description: "Part-2 of the post of the navigation project of the DeepRL course 
              by Udacity. We will cover the implementation of the DQN algorithm
              from Mnih et. al., discuss some gotchas during testing and tuning
              hyperparameters, and also give some the results of our implementation."
date: 2019-05-18T17:32:00-05:00
draft: false
math: true
markup: mmark
---

# Using DQN to solve the Banana environment from ML-Agents

This is an Part-2 of an accompanying post for the submission of the **Project 1: navigation**
from the [**Udacity Deep Reinforcement Learning Nanodegree**](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893),
which consisted on building a DQN-based agent to navigate and collect bananas
from the *Banana Collector* environment from [**Unity ML-Agents**](https://github.com/Unity-Technologies/ml-agents).

{{<figure src="https://wpumacay.github.io/research_blog/imgs/gif_banana_agent.gif" alt="fig-banana-agent" position="center" 
    caption="Figure 1. DQN agent collecting bananas" captionPosition="center"
    style="border-radius: 8px;" >}}

The following are the topics to be covered in this post:

4. [DQN Implementation.](#4-dqn-implementation)
5. [Testing and choosing hyperparameters.](#5-testing-and-choosing-hyperparameters)
6. [Results and discussion](#6-results-of-dqn-on-the-banana-collector-environment)

## 4. DQN Implementation

At last, in this section we will cover the full implementation of the DQN algorithm
from [2] that uses soft-updates. This implementation is based on the original DQN 
implementation from Deepmind (written in Torch and Lua), which can be found [here](https://sites.google.com/a/deepmind.com/dqn/).
Our implementation tries to decouple the key aspects of the algorithm from the model 
itself in a kind of library-agnostic way (effectively decoupling the RL part from
the DL part). At the end of this section we will add some considerations we made 
for our implementation.

**Disclaimer**: the trained model has not been tested in the Bnana-Visual environment,
which provides as observations frames instead of the vector observations discussed
earlier. There might be some code that is related to the visual case which I've been
testing for a while, but I'm still debugging it. I'll update this post with the visual 
case as well once I have it working properly.

### 4.1 Interfaces and Concretions

Our implementation makes use of abstractions and decoupling, as we implemented 
the algorithm from scratch. The main reason is that we wanted to grasp extra 
details of the algorithm without following another implementation step by step
(and also to avoid having a pile of tf.ops to deal with or some pytorch specific 
functionality laying around the core features of the algorithm :smile:).

Just to be in the same page, by an **Interface** we mean a class that provides
a blueprint for other classes to extend. These interfaces could have declarations
of methods and data that their objects will use, but it does not implement them
(at least most of them) and leaves some *pure-virtual* methods to be implemented
by a child class. A **Concretion** is a specific class that extends the functionality 
of this interface, and it has to implement the pure-virtual methods. For example,
our **agent interface** defines the functionality that is exposed by any *agent* 
along with some code that all concretions should have (like the actual steps of
the DQN algorithm) but leaves some methods for the concrete agents, like the preprocess
method which is case specific; and an **agent concretion** could be an agent that 
extends this functionality (and gains the common DQN implementation) but implements
its own version of the preprocess step.

The interfaces and some of its concretions are listed below:

* **Agent Interface**, along concrete agents for **BananaSimple**, **BananaVisual** and **Gridworld**.
* **Model Interface**, along concrete models for **Pytorch**, **Tensorflow** or a **numpy q-table**.
* **Memory Interface**, along concrete buffers like **ReplayBuffer** and **PriorityBuffer**.

We wanted to make these interfaces as decoupled as possible from any Deep Learning
package, which allowed us to test each component separately (we did not want to
get into subtle bugs and poor performance due to specifics of the DL package). We
specially made use of this feature when testing the core agent interface with a simple
gridworld environment. This allowed us to find some bugs in our inplementation of the
steps in the DQN algorithm that might have taken us more time if considering also
the possibility that our DL model was buggy.

### 4.2 Agent Interface and Concretions

This interface has the implementation of the steps in the DQN algorithm, and has
as components the model and memory in order to query them as needed. This is an 
"Abstract Class" in the sense that it shouldn't be instantiated. Instead, a specific
class that inherits from this interface has to implement the **preprocess** method.
Below we show this interface (only the methods, to give a sense of how this interface works),
which can be found in the [agent.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/dqn/core/agent.py)
file in the **dqn/core** folder of the navigation package. 

The key methods to be considered are the **\_\_init\_\_**, **act**, **step**, **\_preprocess** 
and **_learn** methods, which implement most of the required steps of the DQN algorithm.
We will discuss some of their details in snippets below, and we encourage the reader to
look at the full implementation using the hyperlinks provided. Some might have
some details removed (like dev. changes during testing), so they might look a bit
different than the originals from the repo.

```python

class IDqnAgent( object ) :

    def __init__( self, agentConfig, modelConfig, modelBuilder, backendInitializer ) :
        """Constructs a generic Dqn agent, given configuration information

        Args:
            agentConfig (DqnAgentConfig)  : config object with agent parameters
            modelConfig (DqnModelConfig)  : config object with model parameters
            modelBuilder (function)       : factory function to instantiate the model
            backendInitializer (function) : function to be called to intialize specifics of each DL library

        """
        ###############################################
        ##          IMPLEMENTATION HERE
        ###############################################

    def save( self, filename ) :
        """Saves learned models into disk

        Args: 
            filename (str) : filepath where we want to save the our model

        """
        ###############################################
        ##          IMPLEMENTATION HERE
        ###############################################

    def load( self, filename ) :
        """Loads a trained model from disk

        Args:
            filename (str) : filepath where we want to load our model from

        """
        ###############################################
        ##          IMPLEMENTATION HERE
        ###############################################

    def act( self, state, inference = False ) :
        """Returns an action to take from the given state

        Args:
            state (object)    : state|observation coming from the simulator
            inference (bool)  : whether or not we are in inference mode

        Returns:
            int : action to take (assuming discrete actions space)

        """
        ###############################################
        ##          IMPLEMENTATION HERE
        ###############################################

    def step( self, transition ) :
        """Does one step of the learning algorithm, from Mnih et. al.
           https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

        """
        ###############################################
        ##          IMPLEMENTATION HERE
        ###############################################

    def _preprocess( self, rawState ) :
        """Preprocess a raw state into an appropriate state representation
    
        Args:
            rawState (np.ndarray) : raw state to be transformed

        Returns:
            np.ndarray : preprocess state into the approrpiate representation
        """
        raise NotImplementedError( 'IDqnAgent::_preprocess> virtual method' )
        
    def _learn( self ) :
        """Makes a learning step using the DQN algorithm from Mnih et. al.
           https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

        """
        ###############################################
        ##          IMPLEMENTATION HERE
        ###############################################

    @property
    def epsilon( self ) :
        return self._epsilon

    @property
    def seed( self ) :
        return self._seed
        
    @property
    def learningMaxSteps( self ) :
        return self._learningMaxSteps
    
    @property
    def actorModel( self ) :
        return self._qmodel_actor

    @property
    def targetModel( self ) :
        return self._qmodel_target

    @property
    def replayBuffer( self ) :
        return self._rbuffer
```

* First we have the [**\_\_init\_\_**](https://github.com/wpumacay/DeeprlND-projects/blob/99830bc995552c2f6f3a54d8750fc660e9a8e89c/project1-navigation/navigation/dqn/core/agent.py#L13) 
  method, whose implementation is shown briefly in the snippet below. This is the
  constructor of our agent and is in charge of copying the hyperparameters from the
  passed configuration objects, create the models (action-value and target action-value
  networks), create the replay buffer (or a priority-based replay buffer if requested) 
  and some other initialization stuff. We get around having to decouple the specific 
  model creation code by passing a factory method that takes care of this.

```python
    def __init__( self, agentConfig, modelConfig, modelBuilder, backendInitializer ) :
        """Constructs a generic Dqn agent, given configuration information

        Args:
            agentConfig (DqnAgentConfig)  : config object with agent parameters
            modelConfig (DqnModelConfig)  : config object with model parameters
            modelBuilder (function)       : factory function to instantiate the model
            backendInitializer (function) : function to be called to intialize specifics of each DL library

        """

        ##################################
        ##     COPY HYPERPARAMETERS     ##
        ##################################
        
        # seed numpy's random number generator
        np.random.seed( self._seed )

        # create the model accordingly
        self._qmodel_actor = modelBuilder( 'actor_model', modelConfig, True )
        self._qmodel_target = modelBuilder( 'target_model', modelConfig, False )

        ##################################
        ##     INITIALIZE  MODELS       ##
        ##################################

        # start the target model from the actor model
        self._qmodel_target.clone( self._qmodel_actor, tau = 1.0 )

        # create the replay buffer
        if self._usePrioritizedExpReplay :
            self._rbuffer = prioritybuffer.PriorityBuffer( self._replayBufferSize,
                                                           self._seed )
        else :
            self._rbuffer = replaybuffer.DqnReplayBuffer( self._replayBufferSize,
                                                          self._seed )

```

* The [**act**](https://github.com/wpumacay/DeeprlND-projects/blob/99830bc995552c2f6f3a54d8750fc660e9a8e89c/project1-navigation/navigation/dqn/core/agent.py#L127) 
  method is in charge of deciding which action to take in a given state. This takes
  care of both the case of doing \\( \epsilon \\)-greedy during training, and taking
  only the greedy action during inference. Note that in order to take the greedy
  actions we query the action-value network with the appropriate state representation
  in order to get the Q-values required to apply the \\( \argmax \\) function


```python
    def act( self, state, inference = False ) :
        """Returns an action to take from the given state

        Args:
            state (object)    : state|observation coming from the simulator
            inference (bool)  : whether or not we are in inference mode

        Returns:
            int : action to take (assuming discrete actions space)

        """

        if inference or np.random.rand() > self._epsilon :
            return np.argmax( self._qmodel_actor.eval( self._preprocess( state ) ) )
        else :
            return np.random.choice( self._nActions )
```

* The [**step**](https://github.com/wpumacay/DeeprlND-projects/blob/99830bc995552c2f6f3a54d8750fc660e9a8e89c/project1-navigation/navigation/dqn/core/agent.py#L148)
  method implements most of the control flow of the DQN algorithm like adding experience 
  tuples to the replay buffer, doing training every few steps (given by a frequency 
  hyperparameter), doing copies of weights (or soft-updates) to the target network 
  every few steps, doing some book keeping of the states, and applying the schedule
  to \\( \epsilon \\) to control the ammount of exploration.

```python
    def step( self, transition ) :
        """Does one step of the learning algorithm, from Mnih et. al.
           https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

        """
        
        # grab information from this transition
        _s, _a, _snext, _r, _done = transition
        # preprocess the raw state
        self._nextState = self._preprocess( _snext )
        if self._currState is None :
            self._currState = self._preprocess( _s ) # for first step
        # store in replay buffer
        self._rbuffer.add( self._currState, _a, self._nextState, _r, _done )

        # check if can do a training step
        if self._istep > self._learningStartsAt and \
           self._istep % self._learningUpdateFreq == 0 and \
           len( self._rbuffer ) >= self._minibatchSize :
            self._learn()

        # update the parameters of the target model (every update_target steps)
        if self._istep > self._learningStartsAt and \
           self._istep % self._learningUpdateTargetFreq == 0 :
           self._qmodel_target.clone( self._qmodel_actor, tau = self._tau )

        # save next state (where we currently are in the environment) as current
        self._currState = self._nextState

        # update the agent's step counter
        self._istep += 1
        # and the episode counter if we finished an episode, and ...
        # the states as well (I had a bug here, becasue I didn't ...
        # reset the states).
        if _done :
            self._iepisode += 1
            self._currState = None
            self._nextState = None

        # check epsilon update schedule and update accordingly
        if self._epsSchedule == 'linear' :
            # update epsilon using linear schedule
            _epsFactor = 1. - ( max( 0, self._istep - self._learningStartsAt ) / self._epsSteps )
            _epsDelta = max( 0, ( self._epsStart - self._epsEnd ) * _epsFactor )
            self._epsilon = self._epsEnd + _epsDelta

        elif self._epsSchedule == 'geometric' :
            if _done :
                # update epsilon with a geometric decay given by a decay factor
                _epsFactor = self._epsDecay if self._istep >= self._learningStartsAt else 1.0
                self._epsilon = max( self._epsEnd, self._epsilon * _epsFactor )
```

* The [**_preprocess**](https://github.com/wpumacay/DeeprlND-projects/blob/99830bc995552c2f6f3a54d8750fc660e9a8e89c/project1-navigation/navigation/dqn/core/agent.py#L200)
  (as mentioned earlier) is a virtual method that has to be implemented by the
  actual concretions. It receives the observations from the simulator and returns
  the appropriate state representation to use. Some sample implementations for the
  [BananaSimple-agent](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/agent_raycast.py#L14), 
  [BananaVisual-agent](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/agent_gridworld.py#L14) and 
  [Gridworld-agent](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/agent_visual.py#L21) 
  are shown in the snippets that follow.

```python
    def _preprocess( self, rawState ) :
        """Preprocess a raw state into an appropriate state representation
    
        Args:
            rawState (np.ndarray) : raw state to be transformed

        Returns:
            np.ndarray : preprocess state into the approrpiate representation
        """

        """ OVERRIDE this method with your specific preprocessing """

        raise NotImplementedError( 'IDqnAgent::_preprocess> virtual method' )
```

```python
    ###############################################
    ## Simple (just copy) preprocessing          ##
    ###############################################

    def _preprocess( self, rawState ) :
        # rawState is a vector-observation, so just copy it
        return rawState.copy()
```

```python
    ###############################################
    ## One-hot encoding preprocessing            ##
    ###############################################

    def _preprocess( self, rawState ) :
        # rawState is an index, so convert it to a one-hot representation
        _stateOneHot = np.zeros( self._stateDim )
        _stateOneHot[rawState] = 1.0

        return _stateOneHot
```

```python
    ################################################
    ##  Stack 4 frames into vol. preprocessing    ##
    ################################################

    def _preprocess( self, rawState ) :
        # if queue is empty, just repeat this rawState -------------------------
        if len( self._frames ) < 1 :
            for _ in range( 4 ) :
                self._frames.append( rawState )
        # ----------------------------------------------------------------------

        # send this rawState to the queue
        self._frames.append( rawState )

        # grab the states to be preprocessed
        _frames = list( self._frames )

        if USE_GRAYSCALE :
            # convert each frame into grayscale
            _frames = [ 0.299 * rgb[0,...] + 0.587 * rgb[1,...] + 0.114 * rgb[2,...] \
                        for rgb in _frames ]
            _frames = np.stack( _frames )

        else :
            _frames = np.concatenate( _frames )

        return _frames
```

* The [**_learn**](https://github.com/wpumacay/DeeprlND-projects/blob/99830bc995552c2f6f3a54d8750fc660e9a8e89c/project1-navigation/navigation/dqn/core/agent.py#L214)
  method is the one in charge of doing the actual training. As explained in the algorithm
  we first sample a minibatch from the replay buffer, compute the TD-targets (in a
  vectorized way) and request a step of SGD using the just computed TD-targets and 
  the experiences in the minibatch.

```python
    def _learn( self ) :
        """Makes a learning step using the DQN algorithm from Mnih et. al.
           https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

        """

        # get a minibatch from the replay buffer
        _minibatch = self._rbuffer.sample( self._minibatchSize )
        _states, _actions, _nextStates, _rewards, _dones = _minibatch

        # compute targets using the target network in a "vectorized" way
        _qtargets = _rewards + ( 1 - _dones ) * self._gamma * \
                    np.max( self._qmodel_target.eval( _nextStates ), 1 )

        # casting to float32 (to avoid errors due different tensor types)
        _qtargets = _qtargets.astype( np.float32 )

        # make the learning call to the model (kind of like supervised setting)
        self._qmodel_actor.train( _states, _actions, _qtargets )
```

* Finally, there are some specific concretions of this interface (as we mentioned
  earlier). We already showed the required implementation of the **_preprocess**
  method but, for completeness, you can find these concretions in the [agent_raycast.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/agent_raycast.py),
  [agent_gridworld.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/agent_gridworld.py), and 
  [agent_visual.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/agent_visual.py).

### 4.3 Model Interface and Concretions

This interface abstracts away the required functionality that the agent interface
needs from a model, regardless of the specific Deep Learning library used. Unlike
the agent interface, which has some base code common to all agent types, this model
interface has no common code implementation, apart from some getters (as you can see
in the snippet below). The descriptions of all virtual methods that the backend-specific
models have to implement are :

* **build** : build the architecture of the model (either using keras or torch.nn).
* **eval**  : computes all q-values for a given state doing a forward pass.
* **train** : performs SGD (or some other optimizer) with the TD-targets as true q-values to fit.
* **clone** : implements soft-updates from another IDqnModel

```python
class IDqnModel( object ) :

    def __init__( self, name, modelConfig, trainable ) :
        super( IDqnModel, self ).__init__()

        ##################################
        ##   COPY CONFIGURATION DATA    ##
        ##################################

    def build( self ) :
        raise NotImplementedError( 'IDqnModel::build> virtual method' )

    def eval( self, state ) :
        raise NotImplementedError( 'IDqnModel::eval> virtual method' )

    def train( self, states, actions, targets ) :
        raise NotImplementedError( 'IDqnModel::train> virtual method' )

    def clone( self, other, tau = 1.0 ) :
        raise NotImplementedError( 'IDqnModel::clone> virtual method' )

    def save( self, filename ) :
        raise NotImplementedError( 'IDqnModel::save> virtual method' )

    def load( self, filename ) :
        raise NotImplementedError( 'IDqnModel::load> virtual method' )

    def initialize( self, args ) :
        raise NotImplementedError( 'IDqnModel::initialize> virtual method' )

    @property
    def losses( self ) :
        return self._losses

    @property
    def name( self ) :
        return self._name

    @property
    def trainable( self ) :
        return self._trainable

    @property
    def useImpSampling( self ) :
        return self._useImpSampling

    @property
    def gradients( self ) :
        return self._gradients

    @property
    def bellmanErrors( self ) :
        return self._bellmanErrors
```

* The [**model_pytorch.py**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/model_pytorch.py)
  file contains a concrete implementation of the model interface using Pytorch
  as Deep Learning library. Below there is a snippet with most of the contents of
  this file. The **DqnModelPytorch** class serves as a proxy for the actual network
  implemented in a standard way using the **torch.nn** module and the **torch.nn.Module**
  class. Recall that the **eval** method is used for two purposes: computing the q-values
  for the action decision, and computing the q-targets for learning. We have to make
  sure this evaluation **is not considered** for gradients computations. We make use of 
  [torch.no_grad](https://pytorch.org/docs/stable/autograd.html#torch.autograd.no_grad)
  to ensure this requirement.We only have to compute gradients w.r.t. the weights of 
  the action-value network \\( Q_{\theta} \\) during the training step, which is 
  done automatically when the computation graph includes it for gradients computation 
  when calling the network on the inputs $$\left \{ (s,a) \right \}$$ from the minibatch 
  (see train method).

```python
class NetworkPytorchCustom( nn.Module ) :

    def __init__( self, inputShape, outputShape, layersDefs ) :
        super( NetworkPytorchCustom, self ).__init__()

        # banana-raycast has a 37-vector as an observation (rank-1 tensor)
        assert len( inputShape ) == 1, 'ERROR> input should be a rank-1 tensor'
        # and also has a discrete set of actions, with a 4-vector for its qvalues
        assert len( outputShape ) == 1, 'ERROR> output should be rank-1 tensor'

        self._inputShape = inputShape
        self._outputShape = outputShape

        # define layers for this network
        self.fc1 = nn.Linear( self._inputShape[0], 128 )
        self.fc2 = nn.Linear( 128, 64 )
        self.fc3 = nn.Linear( 64, 16 )
        self.fc4 = nn.Linear( 16, self._outputShape[0] )

        self.h1 = None
        self.h2 = None
        self.h3 = None
        self.out = None

    def forward( self, X ) :
        self.h1 = F.relu( self.fc1( X ) )
        self.h2 = F.relu( self.fc2( self.h1 ) )
        self.h3 = F.relu( self.fc3( self.h2 ) )

        self.out = self.fc4( self.h3 )

        return self.out

    def clone( self, other, tau ) :
        for _thisParams, _otherParams in zip( self.parameters(), other.parameters() ) :
            _thisParams.data.copy_( ( 1. - tau ) * _thisParams.data + ( tau ) * _otherParams.data )

class DqnModelPytorch( model.IDqnModel ) :

    def __init__( self, name, modelConfig, trainable ) :
        super( DqnModelPytorch, self ).__init__( name, modelConfig, trainable )

    def build( self ) :
        self._nnetwork = NetworkPytorchCustom( self._inputShape,
                                               self._outputShape,
                                               self._layersDefs )

    def initialize( self, args ) :
        # grab current pytorch device
        self._device = args['device']
        # send network to device
        self._nnetwork.to( self._device )
        # create train functionality if necessary
        if self._trainable :
            self._lossFcn = nn.MSELoss()
            self._optimizer = optim.Adam( self._nnetwork.parameters(), lr = self._lr )

    def eval( self, state, inference = False ) :
        _xx = torch.from_numpy( state ).float().to( self._device )

        self._nnetwork.eval()
        with torch.no_grad() :
            _qvalues = self._nnetwork( _xx ).cpu().data.numpy()
        self._nnetwork.train()

        return _qvalues

    def train( self, states, actions, targets, impSampWeights = None ) :
        if not self._trainable :
            print( 'WARNING> tried training a non-trainable model' )
            return None
        
        _aa = torch.from_numpy( actions ).unsqueeze( 1 ).to( self._device )
        _xx = torch.from_numpy( states ).float().to( self._device )
        _yy = torch.from_numpy( targets ).float().unsqueeze( 1 ).to( self._device )

        # reset the gradients buffer
        self._optimizer.zero_grad()
    
        # do forward pass to compute q-target predictions
        _yyhat = self._nnetwork( _xx ).gather( 1, _aa )
    
        # and compute loss and gradients
        _loss = self._lossFcn( _yyhat, _yy )
        _loss.backward()

        # compute bellman errors (either for saving or for prioritized exp. replay)
        with torch.no_grad() :
            _absBellmanErrors = torch.abs( _yy - _yyhat ).cpu().numpy()
    
        # run optimizer to update the weights
        self._optimizer.step()
    
        # grab loss for later saving
        self._losses.append( _loss.item() )

        return _absBellmanErrors

    def clone( self, other, tau = 1.0 ) :
        self._nnetwork.clone( other._nnetwork, tau )

    def save( self, filename ) :
        if self._nnetwork :
            torch.save( self._nnetwork.state_dict(), filename )

    def load( self, filename ) :
        if self._nnetwork :
            self._nnetwork.load_state_dict( torch.load( filename ) )
```

* The [**model_tensorflow.py**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/model_tensorflow.py)
  file contains a concrete implementation of the model interface using Tensorflow
  as Deep Learning library. Below there is a snippet with most of the contents of
  this file. The **DqnModelTensorflow** class serves a container for the Tensorflow
  Ops created for the computation graph that implements the required evaluation and 
  training steps. For the architecture, instead of creating tf ops for each layer,
  we decided to just use keras to create the required ops of the q-networks internally
  (see the *createNetworkCustom* function), and then build on top of them by creating 
  other ops required for training and evaluation (see the *build* method). Because we
  are creating a static graph beforehand it makes it easier to see where we are going to
  have gradients being computed and used. If our model has its **_trainable** flag
  set to false we then just create the required ops for evaluation only (used for
  computing the TD-targets), whereas, if our model is trainable, we create the full 
  computation graph which goes from inputs (minibatch $$\left \{ (s,a) \right \}$$)
  to the MSE loss using the estimates from the network and the TD-targets passed for training.

```python
def createNetworkCustom( inputShape, outputShape, layersDefs ) :
    # vector as an observation (rank-1 tensor)
    assert len( inputShape ) == 1, 'ERROR> input should be a rank-1 tensor'
    # and also discrete actions , with a 4-vector for its qvalues
    assert len( outputShape ) == 1, 'ERROR> output should be rank-1 tensor'

    # keep things simple (use keras for core model definition)
    _networkOps = keras.Sequential()

    # define initializers
    _kernelInitializer = keras.initializers.glorot_normal( seed = 0 )
    _biasInitializer = keras.initializers.Zeros()

    # add the layers for our test-case
    _networkOps.add( keras.layers.Dense( 128, activation = 'relu', input_shape = inputShape, kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( 64, activation = 'relu', kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( 16, activation = 'relu', kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )
    _networkOps.add( keras.layers.Dense( outputShape[0], kernel_initializer = _kernelInitializer, bias_initializer = _biasInitializer ) )

    return _networkOps

class DqnModelTensorflow( model.IDqnModel ) :

    def __init__( self, name, modelConfig, trainable ) :
        super( DqnModelTensorflow, self ).__init__( name, modelConfig, trainable )

    def build( self ) :
        # placeholder for state inputs
        self._tfStates = tf.placeholder( tf.float32, (None,) + self._inputShape )

        # create the nnetwork model architecture
        self._nnetwork = createNetworkCustom( self._inputShape,
                                              self._outputShape,
                                              self._layersDefs )
        
        # create the ops for evaluating the output of the model (Q(s,:))
        self._opQhat_s = self._nnetwork( self._tfStates )

        # if trainable (action network), create the full resources
        if self._trainable :
            # placeholders: actions, act-indices (gather), and computed q-targets
            self._tfActions             = tf.placeholder( tf.int32, (None,) )
            self._tfActionsIndices      = tf.placeholder( tf.int32, (None,) )
            self._tfQTargets            = tf.placeholder( tf.float32, (None,) )

            # @TODO|CHECK: Change the gather call by multiply + one-hot.
            # Create the ops for getting the Q(s,a) for each batch of (states) + (actions) ...
            # using tf.gather_nd, and expanding action indices with batch indices
            self._opActionsWithIndices = tf.stack( [self._tfActionsIndices, self._tfActions], axis = 1 )
            self._opQhat_sa = tf.gather_nd( self._opQhat_s, self._opActionsWithIndices )
    
            # create ops for the loss function
            self._opLoss = tf.losses.mean_squared_error( self._tfQTargets, self._opQhat_sa )
    
            # create ops for the loss and optimizer
            self._optimizer = tf.train.AdamOptimizer( learning_rate = self._lr )
            self._opOptim = self._optimizer.minimize( self._opLoss, var_list = self._nnetwork.trainable_weights )

        # tf.Session, passed by the backend-initializer
        self._sess = None

    def initialize( self, args ) :
        # grab session and initialize
        self._sess = args['session']

    def eval( self, state, inference = False ) :
        # unsqueeze if it's not a batch
        _batchStates = [state] if state.ndim == 1 else state
        _qvalues = self._sess.run( self._opQhat_s, feed_dict = { self._tfStates : _batchStates } )

        return _qvalues

    def train( self, states, actions, targets, impSampWeights = None ) :
        if not self._trainable :
            print( 'WARNING> tried training a non-trainable model' )
            return None
        
        # for gather functionality
        _actionsIndices = np.arange( actions.shape[0] )
        # dictionary to feed to placeholders
        _feedDict = { self._tfStates : states,
                      self._tfActions : actions,
                      self._tfActionsIndices : _actionsIndices,
                      self._tfQTargets : targets }

        # run the session
        _, _loss = self._sess.run( [ self._opOptim, self._opLoss ], _feedDict )
    
        # grab loss for later statistics
        self._losses.append( _loss )

    def clone( self, other, tau = 1.0 ) :
        _srcWeights = self._nnetwork.get_weights()
        _dstWeights = other._nnetwork.get_weights()

        _weights = []
        for i in range( len( _srcWeights ) ) :
            _weights.append( ( 1. - tau ) * _srcWeights[i] + ( tau ) * _dstWeights[i] )

        self._nnetwork.set_weights( _weights )

    def save( self, filename ) :
        self._nnetwork.save_weights( filename )

    def load( self, filename ) :
        self._nnetwork.load_weights( filename )
```

### 4.4 Memory Interface and Concretions

* The [interface](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/dqn/utils/buffer.py) 
  for the replay buffer is simpler than the others. Basically, we just need a buffer 
  where to store experience tuples (**add** virtual method) and a way to sample a 
  small batch of experience tuples (**sample** virtual method).

```python
class IBuffer( object ) :

    def __init__( self, bufferSize, randomSeed ) :
        super( IBuffer, self ).__init__()

        # capacity of the buffer
        self._bufferSize = bufferSize

        # seed for random number generator (either numpy's or python's)
        self._randomSeed = randomSeed

    def add( self, state, action, nextState, reward, endFlag ) :
        """Adds a transition tuple into memory
        
        Args:
            state       (object)    : state at timestep t
            action      (int)       : action taken at timestep t
            nextState   (object)    : state from timestep t+1
            reward      (float)     : reward obtained from (state,action)
            endFlag     (bool)      : whether or not nextState is terminal

        """
        raise NotImplementedError( 'IBuffer::add> virtual method' )

    def sample( self, batchSize ) :
        """Adds a transition tuple into memory
        
        Args:
            batchSize (int) : number of experience tuples to grab from memory

        Returns:
            list : a list of experience tuples

        """
        raise NotImplementedError( 'IBuffer::sample> virtual method' )

    def __len__( self ) :
        raise NotImplementedError( 'IBuffer::__len__> virtual method' )
```

* The non-prioritized [replay buffer](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/dqn/utils/replaybuffer.py) 
  (as in [2]) is implemented (snippet shown below) using a double ended queue 
  (**deque** from python's collections module). We also implemented a prioritized 
  [replay buffer](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/dqn/utils/prioritybuffer.py)
  for Prioritized Experience Replay. This will be discussed in an improvements section
  later.

```python
class DqnReplayBuffer( buffer.IBuffer ) :

    def __init__( self, bufferSize, randomSeed ) :
        super( DqnReplayBuffer, self ).__init__( bufferSize, randomSeed )

        self._experience = namedtuple( 'Step', 
                                       field_names = [ 'state', 
                                                       'action',
                                                       'reward',
                                                       'nextState',
                                                       'endFlag' ] )

        self._memory = deque( maxlen = bufferSize )

        # seed random generator (@TODO: What is the behav. with multi-agents?)
        random.seed( randomSeed )

    def add( self, state, action, nextState, reward, endFlag ) :
        # create a experience object from the arguments
        _expObj = self._experience( state, action, reward, nextState, endFlag )
        # and add it to the deque memory
        self._memory.append( _expObj )

    def sample( self, batchSize ) :
        # grab a batch from the deque memory
        _expBatch = random.sample( self._memory, batchSize )

        # stack each experience component along batch axis
        _states = np.stack( [ _exp.state for _exp in _expBatch if _exp is not None ] )
        _actions = np.stack( [ _exp.action for _exp in _expBatch if _exp is not None ] )
        _rewards = np.stack( [ _exp.reward for _exp in _expBatch if _exp is not None ] )
        _nextStates = np.stack( [ _exp.nextState for _exp in _expBatch if _exp is not None ] )
        _endFlags = np.stack( [ _exp.endFlag for _exp in _expBatch if _exp is not None ] ).astype( np.uint8 )

        return _states, _actions, _nextStates, _rewards, _endFlags

    def __len__( self ) :
        return len( self._memory )
```

### 4.6 Putting it all together

All the previously mentioned components are used via a single [**trainer**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/trainer.py),
which is in charge of all functionality used for training and evaluation, like:

* Creating the environment to be studied.
* Implementing the RL loops for either training or testing.
* Loading and saving configuration files for different tests.
* Saving statistics from training for later analysis.

Below there's a full snippet of the trainer.py file that implements this functionality:

```python

import os
import numpy as np
import argparse
import time
from tqdm import tqdm
from collections import deque
from collections import defaultdict

# imports from navigation package
from navigation import agent_raycast    # agent for the raycast-based environment
from navigation import model_pytorch    # pytorch-based model
from navigation import model_tensorflow # tensorflow-based model
from navigation.envs import mlagents    # simple environment wrapper
from navigation.dqn.utils import config # config. functionality (load-save)


# logging functionality
import logger

from IPython.core.debugger import set_trace

TEST            = True      # global variable, set by the argparser
TIME_START      = 0         # global variable, set in __main__
RESULTS_FOLDER  = 'results' # global variable, where to place the results of training
SEED            = 0         # global variable, set by argparser
CONFIG_AGENT    = ''        # global variable, set by argparser
CONFIG_MODEL    = ''        # global variable, set by argparser

USE_DOUBLE_DQN                      = False # global variable, set by argparser
USE_PRIORITIZED_EXPERIENCE_REPLAY   = False # global variable, set by argparser
USE_DUELING_DQN                     = False # global variable, set by argparser

def train( env, agent, sessionId, savefile, resultsFilename, replayFilename ) :
    MAX_EPISODES = agent.learningMaxSteps
    MAX_STEPS_EPISODE = 1000
    LOG_WINDOW_SIZE = 100

    _progressbar = tqdm( range( 1, MAX_EPISODES + 1 ), desc = 'Training>', leave = True )
    _maxAvgScore = -np.inf
    _scores = []
    _scoresAvgs = []
    _scoresWindow = deque( maxlen = LOG_WINDOW_SIZE )
    _stepsWindow = deque( maxlen = LOG_WINDOW_SIZE )

    _timeStart = TIME_START

    for iepisode in _progressbar :

        _state = env.reset( training = True )
        _score = 0
        _nsteps = 0

        while True :
            # grab action from dqn agent: runs through model, e-greedy, etc.
            _action = agent.act( _state, inference = False )
            # apply action in simulator to get the transition
            _snext, _reward, _done, _ = env.step( _action )
            ## env.render()
            _transition = ( _state, _action, _snext, _reward, _done )
            # send this transition back to the agent (to learn when he pleases)
            ## set_trace()
            agent.step( _transition )

            # prepare for next iteration
            _state = _snext
            _score += _reward
            _nsteps += 1

            if _done :
                break

        _scores.append( _score )
        _scoresWindow.append( _score )
        _stepsWindow.append( _nsteps )

        if iepisode >= LOG_WINDOW_SIZE :
            _avgScore = np.mean( _scoresWindow )
            _avgSteps = np.mean( _stepsWindow )

            _scoresAvgs.append( _avgScore )

            if _avgScore > _maxAvgScore :
                _maxAvgScore = _avgScore

            # log resultss
            if agent._usePrioritizedExpReplay :
                _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr-Avg=%.2f, Curr=%.2f, Eps=%.2f, Beta=%.2f' % (_maxAvgScore, _avgScore, _score, agent.epsilon, agent._rbuffer.beta ) )
            else :
                _progressbar.set_description( 'Training> Max-Avg=%.2f, Curr-Avg=%.2f, Curr=%.2f, Eps=%.2f' % (_maxAvgScore, _avgScore, _score, agent.epsilon ) )
            _progressbar.refresh()

    # save trained model
    agent.save( savefile )

    _timeStop = int( time.time() )
    _trainingTime = _timeStop - _timeStart

    # save training results for later visualization and analysis
    logger.saveTrainingResults( resultsFilename,
                                sessionId,
                                _timeStart,
                                _scores,
                                _scoresAvgs,
                                agent.actorModel.losses,
                                agent.actorModel.bellmanErrors,
                                agent.actorModel.gradients )

    # save replay batch for later visualization and analysis
    _minibatch = agent.replayBuffer.sample( 100 )
    _ss, _aa, _rr, _ssnext = _minibatch[0], _minibatch[1], _minibatch[2], _minibatch[3]
    _q_s_batch = [ agent.actorModel.eval( agent._preprocess( state ) ) \
                   for state in _ss ]
    _replayBatch = { 'states' : _ss, 'actions' : _aa, 'rewards' : _rr, 'nextStates' : _ssnext }

    logger.saveReplayBatch( replayFilename,
                            sessionId,
                            TIME_START,
                            _replayBatch,
                            _q_s_batch )

def test( env, agent ) :
    _progressbar = tqdm( range( 1, 10 + 1 ), desc = 'Testing>', leave = True )
    for _ in _progressbar :

        _state = env.reset( training = False )
        _score = 0.0
        _goodBananas = 0
        _badBananas = 0

        while True :
            _action = agent.act( _state, inference = True )
            _state, _reward, _done, _ = env.step( _action )

            if _reward > 0 :
                _goodBananas += 1
                _progressbar.write( 'Got banana! :D. So far: %d' % _goodBananas )
            elif _reward < 0 :
                _badBananas += 1
                _progressbar.write( 'Got bad banana :/. So far: %d' % _badBananas )

            _score += _reward

            if _done :
                break

        _progressbar.set_description( 'Testing> Score=%.2f' % ( _score ) )
        _progressbar.refresh()

def experiment( sessionId, 
                library, 
                savefile, 
                resultsFilename, 
                replayFilename, 
                agentConfigFilename, 
                modelConfigFilename ) :

    # grab factory-method for the model according to the library requested
    _modelBuilder = model_pytorch.DqnModelBuilder if library == 'pytorch' \
                        else model_tensorflow.DqnModelBuilder

    # grab initialization-method for the model according to the library requested
    _backendInitializer = model_pytorch.BackendInitializer if library == 'pytorch' \
                            else model_tensorflow.BackendInitializer

    # paths to the environment executables
    _bananaExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux/Banana.x86_64' )
    _bananaHeadlessExecPath = os.path.join( os.getcwd(), 'executables/Banana_Linux_NoVis/Banana.x86_64' )

    if CONFIG_AGENT != '' :
        agent_raycast.AGENT_CONFIG = config.DqnAgentConfig.load( CONFIG_AGENT )

    if CONFIG_MODEL != '' :
        agent_raycast.MODEL_CONFIG = config.DqnModelConfig.load( CONFIG_MODEL )

    # instantiate the environment
    _env = mlagents.createDiscreteActionsEnv( _bananaExecPath, seed = SEED )

    # set the seed for the agent
    agent_raycast.AGENT_CONFIG.seed = SEED

    # set improvement flags
    agent_raycast.AGENT_CONFIG.useDoubleDqn             = USE_DOUBLE_DQN
    agent_raycast.AGENT_CONFIG.usePrioritizedExpReplay  = USE_PRIORITIZED_EXPERIENCE_REPLAY
    agent_raycast.AGENT_CONFIG.useDuelingDqn            = USE_DUELING_DQN

    _agent = agent_raycast.CreateAgent( agent_raycast.AGENT_CONFIG,
                                        agent_raycast.MODEL_CONFIG,
                                        _modelBuilder,
                                        _backendInitializer )

    # save agent and model configurations
    config.DqnAgentConfig.save( agent_raycast.AGENT_CONFIG, agentConfigFilename )
    config.DqnModelConfig.save( agent_raycast.MODEL_CONFIG, modelConfigFilename )

    if not TEST :
        train( _env, _agent, sessionId, savefile, resultsFilename, replayFilename )
    else :
        _agent.load( _savefile )
        test( _env, _agent )

if __name__ == '__main__' :
    _parser = argparse.ArgumentParser()
    _parser.add_argument( 'mode',
                          help = 'mode of execution (train|test)',
                          type = str,
                          choices = [ 'train', 'test' ] )
    _parser.add_argument( '--library', 
                          help = 'deep learning library to use (pytorch|tensorflow)', 
                          type = str, 
                          choices = [ 'pytorch','tensorflow' ], 
                          default = 'pytorch' )
    _parser.add_argument( '--sessionId', 
                          help = 'identifier of this training run', 
                          type = str, 
                          default = 'banana_simple' )
    _parser.add_argument( '--seed',
                          help = 'random seed for the environment and generators',
                          type = int,
                          default = 0 )
    _parser.add_argument( '--visual',
                          help = 'whether or not use the visual-banana environment',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--ddqn',
                          help = 'whether or not to use double dqn (true|false)',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--prioritizedExpReplay',
                          help = 'whether or not to use prioritized experience replay (true|false)',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--duelingDqn',
                          help = 'whether or not to use dueling dqn (true|false)',
                          type = str,
                          default = 'false' )
    _parser.add_argument( '--configAgent',
                          help = 'configuration file for the agent (hyperparameters, etc.)',
                          type = str,
                          default = '' )
    _parser.add_argument( '--configModel',
                          help = 'configuration file for the model (architecture, etc.)',
                          type = str,
                          default = '' )

    _args = _parser.parse_args()

    # whether or not we are in test mode
    TEST = ( _args.mode == 'test' )
    # the actual seed for the environment
    SEED = _args.seed
    # timestamp of the start of execution
    TIME_START = int( time.time() )

    _sessionfolder = os.path.join( RESULTS_FOLDER, _args.sessionId )
    if not os.path.exists( _sessionfolder ) :
        os.makedirs( _sessionfolder )

    # file where to save the trained model
    _savefile = _args.sessionId
    _savefile += '_model_'
    _savefile += _args.library
    _savefile += ( '.pth' if _args.library == 'pytorch' else '.h5' )
    _savefile = os.path.join( _sessionfolder, _savefile )

    # file where to save the training results statistics
    _resultsFilename = os.path.join( _sessionfolder, 
                                     _args.sessionId + '_results.pkl' )

    # file where to save the replay information (for further extra analysis)
    _replayFilename = os.path.join( _sessionfolder,
                                    _args.sessionId + '_replay.pkl' )

    # configuration files for this training session
    _agentConfigFilename = os.path.join( _sessionfolder, _args.sessionId + '_agentconfig.json' )
    _modelConfigFilename = os.path.join( _sessionfolder, _args.sessionId + '_modelconfig.json' )

    # whether or not use the visual-banana environment
    VISUAL = ( _args.visual.lower() == 'true' )

    # DQN improvements options
    USE_DOUBLE_DQN                      = ( _args.ddqn.lower() == 'true' )
    USE_PRIORITIZED_EXPERIENCE_REPLAY   = ( _args.prioritizedExpReplay.lower() == 'true' )
    USE_DUELING_DQN                     = ( _args.duelingDqn.lower() == 'true' )

    # Configuration files with training information (provided by the user)
    CONFIG_AGENT = _args.configAgent
    CONFIG_MODEL = _args.configModel

    print( '#############################################################' )
    print( '#                                                           #' )
    print( '#            Environment and agent setup                    #' )
    print( '#                                                           #' )
    print( '#############################################################' )
    print( 'Mode                    : ', _args.mode )
    print( 'Library                 : ', _args.library )
    print( 'SessionId               : ', _args.sessionId )
    print( 'Savefile                : ', _savefile )
    print( 'ResultsFilename         : ', _resultsFilename )
    print( 'ReplayFilename          : ', _replayFilename )
    print( 'AgentConfigFilename     : ', _agentConfigFilename )
    print( 'ModelConfigFilename     : ', _modelConfigFilename )
    print( 'VisualBanana            : ', _args.visual )
    print( 'DoubleDqn               : ', _args.ddqn )
    print( 'PrioritizedExpReplay    : ', _args.prioritizedExpReplay )
    print( 'DuelingDqn              : ', _args.duelingDqn )
    print( 'Agent config file       : ', 'None' if _args.configAgent == '' else _args.configAgent )
    print( 'Model config file       : ', 'None' if _args.configModel == '' else _args.configModel )
    print( '#############################################################' )

    experiment( _args.sessionId, 
                _args.library,
                _savefile,
                _resultsFilename,
                _replayFilename,
                _agentConfigFilename,
                _modelConfigFilename )
```

This trainer came in handy when running various training sessions with different
configurations for some ablation tests (we did not mention it, but the agent
receives configuration objects that can be loaded and saved using [.json](https://github.com/wpumacay/DeeprlND-projects/tree/master/project1-navigation/configs) 
files).

Finally, note that we developed a [small wrapper](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/envs/mlagents.py) 
on top of our ml-agents environment itself, in order to make it follow a more standard 
(and gym-like) way. Unity ml-agents has also an API for this, but as we are using 
an older version of the library we might had run into some issues, so we decided 
just to go with a simple env. wrapper and call it a day. Note that we also wanted
to go for a standard gym-like API in order to reuse a simple [Gridworld](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/navigation/envs/gridworld.py) 
env. for testing purposes.

### 4.6 Considerations and tradeoffs

* **Why not directly couple everything?, isn't it faster?**: Well, sections that could
  benefit from coupling would be sections that make heavy use of cpu-gpu transfers (kind of like
  passing constantly tensors between cpu and gpu, so it could be faster to just leave them in gpu). 
  The only section so far that could benefit would be the TD-targets calculations, which
  does evaluation in data from minibatches from memory in the current device (either cpu
  or gpu) and then copies this data back to numpy ndarrays. We could actually write
  more ops (in tensorflow) for the whole TD-target calculation and training all in
  one pass, or avoid making copies from gpu to cpu from pytorch tensors, but the only
  part that would benefit would be the TD-targets transfer, which is a batch of vectors.
  Moreover, we wanted to make sure each piece of the DQN algorithm was working, so we
  wanted to be able to test each piece separately.

* **Why not place everything in a notebook?**: Short answer: it's not my usual workflow. 
  Long answer: I'd argue that doing it without notebooks can help you debug things easily,
  specially when you don't have all modules already completed. I've used jupyter notebooks 
  for experiments and they're great for making experiments on top of already-made functionality 
  and for showing results (in fact, some of the results are accesible through some notebooks 
  in the repo), but for the full code I don't think they are the best option. Besides, I'd 
  argue that debugging issues using **pdb** (or in more detail using **code**) is way easier 
  and better than using a notebook (these tools really saved my day quite sometimes). 
  I'd suggest you give it a try as well by running your scripts in *debug* mode, as shown below.

  ```python
  python -mpdb MY_AWESOME_FUNKY_SCRIPT.py <PARAMS>
  ```

## 5. Testing and choosing hyperparameters

In this section we discuss some of the steps we made in order to have a functional
implementation that could solve the task at hand. We had many issues to get it working,
so perhaps some of the steps we made could help you when making your own implementation.

### 5.1 Debugging and testing

Initially, our implementation was supposed to be working on gym environments that
had a continuous state space and a discrete action space, being the environment
suggested for testing gym's LunarLander. We had various bugs before we could have
a working implementation of the DQN on this gym environment:

### Crashes everywhere : 
  These issues where easier to fix. They where mostly due to our inexperience with 
  the Deep learning package at hand. We decided to first implement the concrete 
  model in PyTorch as we were already provided with a solution (by Udacity) that we 
  could refer to in the worst case scenario. For this part, using **pdb** and running 
  the scripts in debug mode helped a lot to debug shapes, arguments that some functions 
  expected in some specific way (squeeze and unsqueeze here and there), etc.. PyTorch was a charm 
  to work with because if something crashed we could just evaluate some tensors 
  using pdb (or worst case using **code**). I'd suggest checking [this](https://www.codementor.io/stevek/advanced-python-debugging-with-pdb-g56gvmpfa)
  post, which explains how to use pdb really well.

### Simple test cases : 
  Once everything ran without crashes, and after various passes to the code to check 
  that everything should work fine, we decided to let the beast loose and destroy 
  the task. Well, that didn't go that well, as I only ended up destroying a virtual 
  lunar lander every episode. There were some bugs in the code that I just didn't 
  notice, even after various passes to the code. This is were having a decouple 
  and modularized implementation came in handy.

  You see, it's a bit frustrating when you look at your code and everything seems
  fine but it just doesn't work, and you actually have lots of places you can check and
  knobs you can tune. Where the errors due to my hyperparameters?. Was there an error
  that might have slipped all my checks?. Hopefully, I had watched a [lecture](https://youtu.be/8EcdaCk9KaQ) 
  from the DeepRL bootcamp in which John Schulman discussed the nuts and bolts of DeepRL.
  A suggestion that saved my day was to test in a simple toy problem and then to 
  start to scale up to the problem at hand. So, I happened to have a gridworld 
  environment laying around that I knew that tabular q-learning should solve.

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_testing_gridworlds.png" alt="fig-testing-gridworlds" position="center" 
    caption="Figure 16. Some gridworlds used to test our implenentation" captionPosition="center"
    style="border-radius: 8px;" >}}

In order to make our DQN agent work in this simple gridworld we just had to modify
the preprocess function such that each discrete state of the gridworld is converted
into a one-hot encoding state-vector representation for our network, as shown in
the following image.

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_testing_gridworld_one_hot_encoding.png" alt="fig-testing-gridworld-one-hot-encoding" position="center" 
    caption="Figure 17. One-hot encoding representation of each state (to serve as input to our network)" captionPosition="center"
    style="border-radius: 8px;" >}}

After fixing our implementation, this is what we obtained as the Q-table, constructed
by evaluating the Q-network in the gridworld discrete states, which is what were expecting
(what tabular q-learning also returned for the same configurations).

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_testing_gridworlds_results.png" alt="fig-testing-gridworlds-results" position="center" 
    caption="Figure 18. Results after fixing the implenentation of DQN in the gridworld environments" captionPosition="center"
    style="border-radius: 8px;" >}}

To train this, just use the [trainer_full.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/trainer_full.py)
file (provided in the navigation package) as follows (for testing just replace "train"
with "test" once training is complete):

```bash
# train a DQN-based agent with a MLP-model to solve gridworld
python trainer_full.py train --gridworld=true

# ... After training is completed

# test the trained model
python trainer_full.py test --gridworld=true
```

We then tried our implementation with some of the gym environments, specifically
the Lunar Lander environment. Our implementation worked correctly on this environment
after the fixes to the issues found using the gridworld test-case. The hyperparameters
were set to roughly the same values found on the baseline of the [Udacity DQN example](https://github.com/udacity/deep-reinforcement-learning/blob/dc65050c8f47b365560a30a112fb84f762005c6b/dqn/solution/dqn_agent.py#L11).
An example of the working agent is shown below:

{{<figure src="https://wpumacay.github.io/research_blog/imgs/gif_lunarlander_agent.gif" alt="fig-lunar-lander-agent" position="center" 
    caption="Figure 19. DQN agent tested on the gym LunarLander-v2 environment" captionPosition="center"
    style="border-radius: 8px;" >}}

To train this, just use the [trainer_full.py](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/trainer_full.py)
as well in the following way (for testing just replace "train" with "test" once
training is complete):

```bash
# train a DQN-based agent with a MLP-model to solve LunarLander-v2
python trainer_full.py train --gym=LunarLander-v2

# ... After training is completed

# test the trained model
python trainer_full.py test --gym=LunarLander-v2
```

Finally, we also tried to visualize the decision making process during runtime,
so we made some simple visualization helpers to plot the Q-values for each action,
and the V-value for each state (following the greedy policy).

{{<figure src="https://wpumacay.github.io/research_blog/imgs/gif_lunarlander_agent_test.gif" alt="fig-lunar-lander-agent-test" position="center" 
    caption="Figure 20. DQN agent during test of the LunarLander-v2 environment" captionPosition="center"
    style="border-radius: 8px;" >}}


### 5.2 Choosing hyperparameters

The hyperparameters were tuned from the starting hyperparameters of the DQN solution
provided for the Lunar Lander. We didn't run exhaustive tests (neither grid nor random
search) to tune the hyperparameters, but just incrementally increased/decreased 
some hyperparameters we considered important:

* **Replay Buffer size**: too low of a replay buffer size (around \\( 10^{4} \\)) gave
  us poor performance even though we made various changes in the other hyperparameters
  to make it work. We then decided to gradually increase it until it could more or less
  solve the task, and finally set it at around \\( 10^{5} \\) in size.

* **Epsilon schedule**: We exponentially decreased exploration via a decay factor applied
  at every end of episode, and keeping \\( \epsilon \\) fixed thereafter. Low exploration
  over all training steps led to poor performance, so we gradually increased it until
  we started getting good performance. The calculations we made to consider how much to increase
  were based on for how long would the exploration be active (until fixed at a certain value),
  how many training steps were available and how big was our replay buffer.

All other hyperparameters were kept roughly to the same values as the baseline provided
by the DQN solution from Udacity, as based on these values they provided some initial
training curves that showed that with their configuration we should be able to get
good results. 

* The max. number of steps was kept fixed to the baseline value provided, as they showed
that at max. 1800 steps a working solution should be able to already solve the task.

* The learning rate and soft-updates factor was kept also the same as the baseline, as
we consider these values to be small enough to not introduce instabilities during learning.
We actually had an issue related to a wrong interpolation which made learning stable
for test cases like lunar lander, but unstable for the banana environment. We were
using as update rule $$\theta^{-} := \tau \theta^{-} + (1-\tau) \theta$$, but the
correct update rule was $$\theta^{-} := (1-\tau) \theta^{-} + \tau \theta$$. As
we were using a very small $$\tau$$, we were effectively running our experiements 
with hard-updates at a very high frequency (1 update per 4 steps) instead of soft-updates.
This seemed to be working fine for the Lunar Lander environment (we increased $$\tau$$
to make it work), but didn't work at all in the banana environment.

* The minibatch size was kept the same (64). As we are not using any big network
nor using high dimensional inputs (like images) we don't actually have to worry much
about our GPU being able to allocate resources for a bigger batch (so we could have set 
a bigger batch size), but we decided it to keep it that way. It would be interesting 
though to see the effect that the batch size has during learning. Of course, it'd take 
a bit longer to take a SGD step, but perhaps by taking a "smoother" gradient step we could get better/smoother learning.

The hyperparameters chosen for our submission (found in the [config_submission.json](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_submission.json)
file) are shown below:

    "stateDim"                  : 37,
    "nActions"                  : 4,
    "epsilonStart"              : 1.0,
    "epsilonEnd"                : 0.01,
    "epsilonSteps"              : -1,
    "epsilonDecay"              : 0.9975,
    "epsilonSchedule"           : "geometric",
    "lr"                        : 0.0005,
    "minibatchSize"             : 64,
    "learningStartsAt"          : 0,
    "learningUpdateFreq"        : 4,
    "learningUpdateTargetFreq"  : 4,
    "learningMaxSteps"          : 2000,
    "replayBufferSize"          : 100000,
    "discount"                  : 0.999,
    "tau"                       : 0.001,
    "seed"                      : 0,
    "useDoubleDqn"              : false,
    "usePrioritizedExpReplay"   : false,
    "useDuelingDqn"             : false

## 6. Results of DQN on the Banana Collector Environment

In this section we show the results of our submission, which were obtained through
various runs and with various seeds. The results are presented in the form of *time 
series plots* each over a single run, and *standard deviation* plots over a set of 
similar runs. We will also show the results of one of three experiments we made with 
different configurations. This experiment did not include the improvements (DDQN, PER), 
which are going to be explained in a later section. The results can be found also
in the [results_analysis.ipynb](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/results_analysis.ipynb).
However, due to the size of the training sessions, we decided not to upload most
of the results (apart from a single run for the submission) due to the file size
of the combined training sessions. Still, to reproduce the same experiments just
run the provided bash scripts:

* **training_submission.sh** : to run the base experiments for the submissions
* **training_tests_1.sh** : to run the first experiment, related to \\( \epsilon \\) tests
* **training_tests_2.sh** : to run the second experiment, related to how much the improvements (DDQN and PER) to DQN help.
* **training_tests_3.sh** : to run the third experiment, to check if our implementations of DDQN and PER actually helps in some setups with little exploration.

### 6.1 Results with the configuration for the submission

* **Single runs**: We choose one of the results of the various runs, plotted as time
  series plots. The x-axis represents episodes (during training) and the y-axis represents
  the score obtained at that episode. Also, the noisy blue plot is the actual score per episode,
  whereas the red curve is a smoothed (running average) curve from the previous one with a
  window of size 100. Below we show one random run from the episode (not cherry-picked) 
  for both our pytorch and tensorflow implementations. As we can see, the agent 
  successfully solves the environment at around episode 900. Note that this submission 
  configuration is not the best configuration (hyperparameters tuned to squeeze the 
  most score out of the environment), but a configuration we considered appropriate 
  (moderate exploration). We found that for more aggressive exploration schedules 
  (fast decay over around 300 episodes) and a moderate sized replay buffer the task 
  could be solved in around 300 steps (see experiment 1 later).

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_results_submission_single_pytorch.png" alt="fig-results-submission-single-pytorch" position="center" 
    caption="Figure 21. Training results from one run using PyTorch as backend" captionPosition="center"
    style="border-radius: 8px;" >}}

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_results_submission_single_tensorflow.png" alt="fig-results-submission-single-tensorflow" position="center" 
    caption="Figure 22. Training results from one run using Tensorflow as backend" captionPosition="center"
    style="border-radius: 8px;" >}}

* **All runs**: Below we shows graphs of all runs in a single plot for three different
  random seeds. We again present one graph for each backend (pytorch and tensorflow).
  We recorded 5 training runs per seed. Recall again that all runs correspond to the
  same set of hyperparameters given by the **config_submission.json** file.
  <!--The results are pretty uniform along random seeds. Of course, this might be caused
  by the nature of the algorithm itself. I was expecting some variability, as mentioned
  in [this](https://youtu.be/Vh4H0gOwdIg?t=1133) lecture on reproducibility. Perhaps
  we don't find that much variability because of the type of methods we are using,
  namely Q-learning which is off-policy. Most of the algorithms studied in the
  lecture were on-policy and based on policy gradients, which depend on the policy
  for the distribution of data they see. This might cause the effect of exploring
  and finding completely different regions due to various variabilities (different
  seeds, different implementations, etc.). Perhaps this might be caused as well-->

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_results_submission_all_runs.png" alt="fig-results-submission-all-runs" position="center" 
    caption="Figure 23. Training results from all runs per random seed. Top: results using PyTorch. Bottom: results using Tensorflow." captionPosition="center"
    style="border-radius: 8px;" >}}

* **Std-plots**: Finally, we show the previous plots in the form of std-plots,
  which try to give a sense of the deviation of the various runs. We collected
  5 runs per seed, and each std-plot is given for a specific seed. Recall again 
  that all runs correspond to the same set of hyperparameters given by the 
  **config_submission.json** file.

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_results_submission_all_runs_pytorch_std.png" alt="fig-results-submission-all-runs-std-pytorch" position="center" 
    caption="Figure 24. Std-plots from all runs using PyTorch. One color per different seed." captionPosition="center"
    style="border-radius: 8px;" >}}

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_results_submission_all_runs_tensorflow_std.png" alt="fig-results-submission-all-runs-std-tensorflow" position="center" 
    caption="Figure 25. Std-plots from all runs using Tensorflow. One color per different seed." captionPosition="center"
    style="border-radius: 8px;" >}}

### 6.2 Experiment 1: tweaking exploration

In this experiment we tried to double-check if decreasing the amount of exploration
would allow the agent to solve the task in fewer episodes. Because the amount of exploration
is small, the agent would be forced to start trusting more its own estimates early on
and, if the setup is right (big enough replay buffer, etc.), this might have the
effect of making the agent solve the task quickly. The configurations used are the
following:

* [**config_agent_1_1.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_1_1.json) : 
  This configuration has a moderate exploration schedule. We decay \\( \epsilon \\)
  from 1.0 to 0.01 using a multiplicative decay factor of 0.9975 applied per episode.
  This schedule makes \\( \epsilon \\) decay from 1.0 to approx. 0.1 over 1000 episodes,
  and to the final value of 0.01 over approx. 1600 episodes.

* [**config_agent_1_2.json**](https://github.com/wpumacay/DeeprlND-projects/blob/master/project1-navigation/configs/config_agent_1_2.json) : 
  This configuration has a little more aggresive exploration schedule. We decay \\( \epsilon \\)
  from 1.0 to 0.01 using a multiplicative decay factor of 0.98 applied per episode.
  This schedule makes \\( \epsilon \\) decay from 1.0 to approx. 0.1 over 100 episodes,
  and to the final value of 0.01 over approx. 200 episodes.

Below we show the training results from 5 runs over 2 random seeds using these
configurations. These plots reveal a trend which suggests that with the second
configuration (less exploration) we get to solve the task faster (around 400 episodes) 
than the first configuration (around 900 episodes).

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_results_experiment_1_all_runs.png" alt="fig-results-experiment-1-all-runs" position="center" 
    caption="Figure 26. Results of the first experiment. Top: moderate exploration. Bottom: little exploration" captionPosition="center"
    style="border-radius: 8px;" >}}

{{<figure src="https://wpumacay.github.io/research_blog/imgs/img_results_experiment_1_all_runs_std.png" alt="fig-results-experiment-1-std" position="center" 
    caption="Figure 27. Std-plots of the first experiment for different seeds." captionPosition="center"
    style="border-radius: 8px;" >}}

## References

* [1] Sutton, Richard & Barto, Andrew. [*Reinforcement Learning: An introduction.*](http://incompleteideas.net/book/RLbook2018.pdf)
* [2] Mnih, Volodymyr & Kavukcuoglu, Koray & Silver, David, et. al.. [*Human-level control through deep-reinforcement learning*](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
* [3] Achiam, Josh. [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
* [4] Simonini, Thomas. [*A Free course in Deep Reinforcement Learning from beginner to expert*](https://simoninithomas.github.io/Deep_reinforcement_learning_Course/)
* [5] [*Stanford RL course by Emma Brunskill*](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u)
* [6] [*UCL RL course, by David Silver*](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
* [7] [UC Berkeley DeepRL course by Sergey Levine](http://rail.eecs.berkeley.edu/deeprlcourse/)
* [8] [Udacity DeepRL Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)
* [9] [DeepRL bootcamp](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [10] Van Hasselt, Hado. [*Double Q-learning*](https://papers.nips.cc/paper/3964-double-q-learning.pdf)
* [11] Van Hasselt, Hado & Guez, Arthur & Silver, David. [*Deep Reinforccement Learning with Double Q-learning*](https://arxiv.org/abs/1509.06461)
* [12] Schaul, Tom & Quan, John & Antonoglou, Ioannis & Silver, David. [*Prioritized Experience Replay*](https://arxiv.org/abs/1511.05952)
* [13] Hacker Earth. [Segment Trees data structures](https://www.hackerearth.com/practice/data-structures/advanced-data-structures/segment-trees/tutorial/)
* [14] OpenAI. [Baselines](https://github.com/openai/baselines)
* [15] Hessel, Matteo & Modayil, Joseph & van Hasselt, Hado & Schaul, Tom & Ostrovski, Georg & Dabney, Will & Horgan, Dan & Piot, Bilal & Azar, Mohammad & Silver, David [Rainbow](https://arxiv.org/abs/1710.02298)
* [16] Hausknecht, Matthew & Stone, Peter [Deep Recurrent Q-Learning with Partially Observable MDPs](https://arxiv.org/abs/1507.06527)