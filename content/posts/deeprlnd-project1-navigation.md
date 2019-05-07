---
title: "Udacity DeepRL project 1: Navigation"
date: 2019-05-06T15:35:13-05:00
draft: true
math: true
markup: mmark
---

# Using DQN to solve the Banana environment from ML-Agents

This is an accompanying post for the submission of the **Project 1: navigation**
from the [**Udacity Deep Reinforcement Learning Nanodegree**](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893),
which consisted on building a DQN-based agent to navigate and collect bananas
from the *Banana Collector* environment from [**Unity ML-Agents**](https://github.com/Unity-Technologies/ml-agents).

{{<figure src="/imgs/gif_banana_agent.gif" alt="fig-banana-agent" position="center" 
    caption="Figure 1. DQN agent collecting bananas" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

The following are the topics to be covered in this post:

1. Description of the *Banana Collector Environment*.
2. Setting up the dependencies to run the accompanying code.
3. An overview of the DQN algorithm.
4. The chosen hyperparameters and some discussion.
5. The results obtained and some discussion.
6. An overview of the improvements: Double DQN.
7. An overview of the improvements: Prioritized Experience Replay.
8. Some ablation tests and some discussion.
9. Final remarks and future improvements.

## 1. Description of the Banana Collector Environment

The environment chosen for the project was a **modified version of the Banana 
Collector Environment** from the Unity ML-Agents toolkit. The original version
can be found [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector),
and our version consists of a custom build provided by Udacity with the following
description:

* A single agent that can move in a planar arena, with **observations** given by
  a set of distance-based sensors and some intrinsic measurements, and **actions** 
  consisting of 4 discrete commands.
* A set of NPCs consisting of bananas of two categories: **yellow bananas**, which give the
  agent a **reward of +1**, and **purple banans**, which give the agent a **reward of -1**.
* The task is **episodic**, with a maximum of 300 steps per episode.

### 1.1 Agent observations

The observations the agent gets from the environment come from the **agent's linear velocity**
in the plane (2 entries), and a **set of 7 ray perceptions**. These ray perceptions consist 
of rays shot in certain fixed directions from the agent. Each of these perceptions 
returns a vector of **5 entries each**, whose values are explained below:

* The first 4 entries consist of a one-hot encoding of the type of object the ray hit, and
  these could either be a yellow banana, a purple banana, a wall or nothing at all.
* The last entry consist of the percent of the ray length at which the object was found. If
  no object is found at least at this maximum length, then the 4th entry is set to 1, and this
  entry is set to 0.0.

Below there are two separate cases that show the ray-perceptions. The first one
to the left shows all rays reaching at least one object (either purple banana, yellow
banana or a wall) and also the 7 sensor reading in array form (see the encodings in the
4 first entries do not contain the *none found* case). The second one to the right
shows all but one ray reaching an object and also the 7 sensor readings in array form (see
the encodings in the 4 first entrines do include the *none found* case for the 4th perception).

{{<figure src="/imgs/img_banana_env_observations.png" alt="fig-banana-agent-ray-observations" position="center" 
    caption="Figure 2. Agent ray-perceptions. a) 7 days reaching at least one object (banana or wall). b) One rayreaching the max. length before reaching any object" captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

### 1.2 Agent actions

Lorem ipsum

{{<figure src="/imgs/img_banana_env_actions.png" alt="fig-banana-agent-actions" position="center" 
    caption="Figure 3. Agent actions." captionPosition="center"
    style="border-radius: 8px;" captionStyle="color: black;">}}

### 1.3 Environment dynamics and rewards

Lorem ipsum

## 2. Accompanying code and setup

### Custom environment build

The environment provided is a custom build of the ml-agents Banana Collector
environment, with the features described in the earlier section. The environment
is provided as an executable which we can download from the links below according
to our platform:

Platform | Link
-------- | -----
Linux             | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX           | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Keep the .zip file where you downloaded it for now. We will later explain where
to extract its contents when finishing the setup.

### Dependencies

As already mentioned, the environment provided is an instance of an ml-agents
environment and as asuch it requires the **appropriate** ml-agents python API to be
installed in order to interact with it, and we make emphasis in appropriate because
if your 

The version for which


### Downloading accompanying code

Lorem ipsum

### Finishing setup

Lorem ipsum 