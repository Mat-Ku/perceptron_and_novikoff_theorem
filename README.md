# Perceptron and Novikoff Theorem
Implementation of perceptron algorithm including the rule established by the Novikoff theorem.

## Description
The perceptron algoriithm is a binary classificatio algorithm that is able to fit a separating line between two classes of linearly separable data. In order to do so, it iteratively adjusts the slope and intercept of the line as long as there are falsely classified training
data instances. In case the data is not linearly separable, the algorithm will not converge. The Novikoff theorem builds on this algorithm, claiming that there is an upper bound for the number of iterations needed until the algorithm has fitted a line that
separates the two classes from each other.

## Data
The data consists of a small, exemplary dataset containing two linearly separable classes of two-dimensional data points.

## Sources
Rosenblatt, 1958: The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain
Retrievable under: https://deeplearning.cs.cmu.edu/F23/document/readings/Rosenblatt_1959-09865-001.pdf
Novikoff, 1962: On convergence proofs on perceptrons. Symposium on the Mathematical Theory of Automata
Retrievable under: https://cs.uwaterloo.ca/~y328yu/classics/novikoff.pdf

## Results
The perceptron was implemented with a learning rate of 0.01 and it took 96 iterations until it had converged. Therefore, all test data instances were classified correctly. 
Furthermore, the upper bound given by the Novikoff theorem, holds.
