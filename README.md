
# OpenSpiel: A Framework for Reinforcement Learning in Games

[![Documentation Status](https://readthedocs.org/projects/openspiel/badge/?version=latest)](https://openspiel.readthedocs.io/en/latest/?badge=latest)
![build_and_test](https://github.com/deepmind/open_spiel/workflows/build_and_test/badge.svg)

OpenSpiel is a collection of environments and algorithms for research in general
reinforcement learning and search/planning in games. OpenSpiel supports n-player
(single- and multi- agent) zero-sum, cooperative and general-sum, one-shot and
sequential, strictly turn-taking and simultaneous-move, perfect and imperfect
information games, as well as traditional multiagent environments such as
(partially- and fully- observable) grid worlds and social dilemmas. OpenSpiel
also includes tools to analyze learning dynamics and other common evaluation
metrics. Games are represented as procedural extensive-form games, with some
natural extensions. The core API and games are implemented in C++ and exposed to
Python. Algorithms and tools are written both in C++ and Python.

To try OpenSpiel in Google Colaboratory, please refer to `open_spiel/colabs` subdirectory or start [here](https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/install_open_spiel.ipynb).

<p align="center">
  <img src="docs/_static/OpenSpielB.png" alt="OpenSpiel visual asset">
</p>

# Index

Please choose among the following options:

*   [Installing OpenSpiel](docs/install.md)
*   [Introduction to OpenSpiel](docs/intro.md)
*   [API Overview and First Example](docs/concepts.md)
*   [Overview of Implemented Games](docs/games.md)
*   [Overview of Implemented Algorithms](docs/algorithms.md)
*   [Developer Guide](docs/developer_guide.md)
*   [Using OpenSpiel as a C++ Library](docs/library.md)
*   [Guidelines and Contributing](docs/contributing.md)
*   [Authors](docs/authors.md)

For a longer introduction to the core concepts, formalisms, and terminology,
including an overview of the algorithms and some results, please see
[OpenSpiel: A Framework for Reinforcement Learning in Games](https://arxiv.org/abs/1908.09453).

For an overview of OpenSpiel and example uses of the core API, please check out
our tutorials:

*   [Motivation, Core API, Brief Intro to Replictor Dynamics and Imperfect
    Information Games](https://www.youtube.com/watch?v=8NCPqtPwlFQ) by Marc
    Lanctot.
    [(slides)](http://mlanctot.info/files/OpenSpiel_Tutorial_KU_Leuven_2022.pdf)
    [(colab)](https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/OpenSpielTutorial.ipynb)
*   [Motivation, Core API, Implementing CFR and REINFORCE on Kuhn poker, Leduc
    poker, and Goofspiel](https://www.youtube.com/watch?v=o6JNHoGUXCo) by Edward
    Lockhart.
    [(slides)](http://mlanctot.info/files/open_spiel_tutorial-mar2021-comarl.pdf)
    [(colab)](https://colab.research.google.com/github/deepmind/open_spiel/blob/master/open_spiel/colabs/CFR_and_REINFORCE.ipynb)

If you use OpenSpiel in your research, please cite the paper using the following
BibTeX:

```
@article{LanctotEtAl2019OpenSpiel,
  title     = {{OpenSpiel}: A Framework for Reinforcement Learning in Games},
  author    = {Marc Lanctot and Edward Lockhart and Jean-Baptiste Lespiau and
               Vinicius Zambaldi and Satyaki Upadhyay and Julien P\'{e}rolat and
               Sriram Srinivasan and Finbarr Timbers and Karl Tuyls and
               Shayegan Omidshafiei and Daniel Hennes and Dustin Morrill and
               Paul Muller and Timo Ewalds and Ryan Faulkner and J\'{a}nos Kram\'{a}r
               and Bart De Vylder and Brennan Saeta and James Bradbury and David Ding
               and Sebastian Borgeaud and Matthew Lai and Julian Schrittwieser and
               Thomas Anthony and Edward Hughes and Ivo Danihelka and Jonah Ryan-Davis},
  year      = {2019},
  eprint    = {1908.09453},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  journal   = {CoRR},
  volume    = {abs/1908.09453},
  url       = {http://arxiv.org/abs/1908.09453},
}
```

## Versioning

We use [Semantic Versioning](https://semver.org/).

