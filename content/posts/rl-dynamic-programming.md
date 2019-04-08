---
title: "Rl Dynamic Programming"
date: 2019-04-01T23:31:27-05:00
draft: true
math: true
markup: mmark
---

# Planning by Dynamic Programming

## Introduction

The first set of methods we will study are exact methods, which can be used
when we have a perfect model of the dynamics of the environment, namely the
joint distribution \\( p(s',r|s,a) \\) defined in our MDP. By having a model
we will take advantage of the recursive form of our problem, given by the
*Bellman Equations* studied earlier, and solve for the *State-Value function* ( \\( V(s) \\) )
and *Action-Value function* ( \\( Q(s,a) \\) ) using the following iterative methods:

* **Policy Evaluation**
* **Policy Iterarion**
* **Value Iteration**

## Exploiting our knowledge of the environment

Let's try to exploit our knowledge of the transition dynamics \\( p(s',r|s,a) \\). Recall
the **Bellman Equations** from the previous post:

* **Bellman Expectation Equation**

$$
V^{\pi}(s) = \mathbb{E}_{\pi} \left \{ R_{t} + \gamma V^{\pi}(s') | s_{t}=s  \right \}
$$

* **Bellman Optimality Equation**

$$
V^{*}(s) = \max_{\pi} \mathbb{E}_{\pi} \left \{ R_{t} + \gamma V^{*}(s') | s_{t}=s \right \}
$$

The nature of the expectations in the equations above is caused by the fact that
the internal quantities that appear recursively ( \\( R_{t} + \gamma V(s') \\) ) 
are random variables, whose distribution is induced by the policy \\( pi \\) we 
follow (which can be deterministic or sthocastic) and the dynamics of the environment
\\( p(s',r|s,a) \\).

In the case we try to solve, we are given the dynamics of the environment, so we
know how the future will unroll once we take a transition \\( (s,a) \\). This means
that we know the probability for which the terms \\( R_{t} + \gamma V(s') \\) will
appear, which allows us to compute the expectation.