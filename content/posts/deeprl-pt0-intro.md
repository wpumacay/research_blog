---
title: "Udacity DeepRL project 1: Navigation"
date: 2019-05-06T15:35:13-05:00
draft: true
math: true
markup: mmark
---

# Using DQN to solve the Banana environment from ML-Agents

In this post we will look at how to implement an agent that uses the DQN algorithm
to solve a simple navigation tasks from the [ml-agents](https://github.com/Unity-Technologies/ml-agents) 
package. This post is part of the submission for the project-1 of the 
[**Deep Reinforcement Learning Nanodegree**](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) 
by *Udacity*.

## Outline

These are the following topics we will cover:

* A quick overview of RL and DeepRL.
* The DQN algorithm in more detail.
* An overview of the Banana environment.
* Implementation details.
* Improvements: Double DQN.
* Improvements: Prioritized experience replay.
* Results.
* Future improvements.

## A quick overview of RL and DeepRL

### Reinforcement learning: the problem

Reinforcement Learning (RL) is a learning approach in which an **Agent** learns by
*trial and error* while interacting with an **Environment**. The core setup is shown 
in Figure 1, where we have an agent in a certain **state** \\( S_{t} \\) 
interacting with an environment by applying some **action** \\( A_{t} \\). 
Because of this interaction, the agent receives a reward \\( R_{t+1} \\) from 
the environment and it also ends up in a new state \\( S_{t+1} \\).

{{<figure src="/imgs/img_rl_loop.png" alt="img-rl-loop" position="center" 
    caption="Figure 1. RL interaction loop" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

This can be further formalize using the framework of Markov Decision Proceses (MDPs).
Using this framework we can define our RL problem as follows:

> A Markov Decision Process (MDP) is defined as a tuple of the following components:
>
> * **A state space** \\( \mathbb{S} \\) of configurations \\( s_{t} \\) for an agent.
> * **An action space** \\( \mathbb{A} \\) of actions \\( a_{t} \\) that the agent can take in the environment.
> * **A transition model** \\( p(s',r | s,a) \\) that defines the distribution of states
>   that an agent can land on and rewards it can obtain \\( s',r \\) by taking an
>   action \\( a \\) in a state \\( s \\).

The objective of the agent is to maximize the **total sum of rewards** that it can
get from its interactions, and because the environment can potentially be stochastic
(recall that the transition model defines a probability distribution) the objective
is usually formulated as an **expectation** ( \\( \mathbb{E}  \\) ) over the random 
variable defined by the total sum of rewards. Mathematically, this is described in
the following equation.

$$
\max \mathbb{E} \left \{ r_{t+1} + r_{t+2} + r_{t+3} + \dots \right \}
$$

### Reinforcement learning: the solution

A solution to the RL problem consists of a **Policy** \\( \pi \\), which is a mapping
from the current state we are ( \\( s_{t} \\) ) to an appropriate action ( \\( a_{t} \\) )
from the action space. Such mapping is basically a function, and we can defined it as follows:

> A **deterministic** policy is a mapping \\( \pi : \mathbb{S} \rightarrow \mathbb{A} \\)
> that returns an action to take \\( a_{t} \\) from a given state \\( s_{t} \\).
>
> $$
> a = \pi(s)
> $$

We could also define an stochastic policy