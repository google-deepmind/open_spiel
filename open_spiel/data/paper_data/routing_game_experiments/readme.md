# Reproducing routing game experiments

To reproduce the experiments done in [*Solving N-player dynamic routing games with congestion: a mean field approach, Cabannes et. al.*](https://arxiv.org/pdf/2110.11943.pdf):

1. If you have not, download [python](https://www.python.org/downloads/) and an IDE to run iPython notebok (either [jupyter](https://jupyter.org) or [VSCode](https://code.visualstudio.com)).
2. Install OpenSpiel using [pip install open_spiel](https://github.com/deepmind/open_spiel/blob/master/docs/install.md) or from [source](https://github.com/deepmind/open_spiel/blob/master/docs/install.md#installation-from-source).
3. Create a folder where you will put the data and the code.
4. Download the Sioux Falls network csv data from [GitHub](https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls/CSV-data) and put `SiouxFalls_net.csv`, `SiouxFalls_node.csv`, and `SiouxFalls_od.csv` in the folder created in (3).
5. Download the [`Experiments.ipynb` iPython notebook](https://github.com/deepmind/open_spiel/tree/master/open_spiel/data/paper_data/routing_game_experiments/Experiments.ipynb) and put it in the folder created in (3).
6. Run the iPython notebook. You might need to download the dependant python libraries.

# License

This code is under the Open Spiel license.
Please cite the paper [*Solving N-player dynamic routing games with congestion: a mean field approach, Cabannes et. al.*](https://arxiv.org/pdf/2110.11943.pdf) when re-using this code.
Feel free to send an email to theophile@berkeley.edu for any questions.
