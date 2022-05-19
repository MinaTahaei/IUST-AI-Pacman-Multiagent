# IUST-AI-Pacman-Multiagent
IUST AI Project Phase 2- Multiagent Pacman

In this project, we will design agents for the classic version of Pacman, including ghosts. Along the way, we will implement both minimax and expectimax search and try our hand at evaluation function design.

Question 1: Improve the ReflexAgent in multiAgents.py to play respectably. A capable reflex agent will have to consider both food locations and ghost locations to perform well.

Question 2: write an adversarial search agent in the provided MinimaxAgent class stub in multiAgents.py. Your minimax agent should work with any number of ghosts, so you'll have to write an algorithm that is slightly more general than what you've previously seen in lecture. In particular, your minimax tree will have multiple min layers (one for each ghost) for every max layer.
Your code should also expand the game tree to an arbitrary depth. Score the leaves of your minimax tree with the supplied self.evaluationFunction, which defaults to scoreEvaluationFunction. MinimaxAgent extends MultiAgentSearchAgent, which gives access to self.depth and self.evaluationFunction. Make sure your minimax code makes reference to these two variables where appropriate as these variables are populated in response to command line options.

Question 3: Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in AlphaBetaAgent.

Question 4: In this question you will implement the ExpectimaxAgent, which is useful for modeling probabilistic behavior of agents who may make suboptimal choices.

Question 5: Write a better evaluation function for pacman in the provided function betterEvaluationFunction. The evaluation function should evaluate states, rather than actions like your reflex agent evaluation function did. 
