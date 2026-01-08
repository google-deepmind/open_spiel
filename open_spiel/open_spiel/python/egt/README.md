# Open Spiel Evolutionary Game Theory (EGT) Toolbox

This is a library for conducting Evolutionary Game Theory (EGT) analysis of
games.

## A Breakdown of the code

The following code implements Alpha-Rank, a multi-agent evaluation algorithm
detailed in `α-Rank: Multi-Agent Evaluation by Evolution (2019)`, available at:
https://www.nature.com/articles/s41598-019-45619-9.

*   `alpharank.py`: core implementation
*   `alpharank_visualizer.py`: Alpha-Rank plotting tools

The following are utility scripts:

*   `heuristic_payoff_table.py`: defines a class for storing heuristic payoff
    tables for games (e.g., as detailed in `A Generalised Method for Empirical
    Game Theoretic Analysis` (Tuyls et al., 2018))
*   `utils.py`: helper functions
