<!--- BEGIN GOOGLE-INTERNAL

This file is the main page of the repo, and contains sub-parts of the
documentation. Update `docs/intro.md` too if you update this page.

END GOOGLE-INTERNAL -->

# OpenSpiel: A Framework for Reinforcement Learning in Games

[![Documentation Status](https://readthedocs.org/projects/openspiel/badge/?version=latest)](https://openspiel.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/deepmind/open_spiel.svg?branch=master)](https://travis-ci.com/deepmind/open_spiel)

OpenSpiel is a collection of environments and algorithms for research in general
reinforcement learning and search/planning in games. OpenSpiel supports n-player
(single- and multi- agent) zero-sum, cooperative and general-sum, one-shot and
sequential, strictly turn-taking and simultaneous-move, perfect and imperfect
information games, as well as traditional multiagent environments such as
(partially- and fully- observable) grid worlds and social dilemmas. OpenSpiel
also includes tools to analyze learning dynamics and other common evaluation
metrics. Games are represented as procedural extensive-form games, with some
natural extensions. The core API and games are implemented in C++ and exposed to
Python. Algorithms and tools are written both in C++ and Python. There is also a
branch of pure Swift in the `swift` subdirectory.

<p align="center">
  <img src="docs/_static/OpenSpielB.png" alt="OpenSpiel visual asset">
</p>

# Index

Please choose among the following options:

*   [Installing OpenSpiel](docs/install.md)
*   [Introduction to OpenSpiel](docs/intro.md)
*   [API Overview and First Example](docs/concepts.md)
*   [Overview of Implemented Games](docs/games.md)
*   [Developer Guide](docs/developer_guide.md)
*   [Guidelines and Contributing](docs/contributing.md)
*   [Swift OpenSpiel](docs/swift.md)
*   [Authors](docs/authors.md)

## BEGIN GOOGLE-INTERNAL

Player of Games is also using the API. The code is available in:
cs/google3/learning/deepmind/research/mcts/impinfo/pog/engine/open_spiel_state.h

TODO(jblespiau): In the OpenSource version, should `scripts/` be at the top
level directory?

## Git-on-Borg (GoB) and release process

We now have the GoB team review site at
[OpenSpiel Team Review Repository](https://team-review.git.corp.google.com/admin/repos/deepmind-eng/OpenSpiel).

You can browse the source at the source
[GoB source browser](https://team.git.corp.google.com/deepmind-eng/OpenSpiel)

NOTE: The current approach is more toward considering Piper as the Source of
Truth. Given that our code is currently only in Piper, it is mandatory to create
the process to go smoothly from Piper to GitHub. Note that setting the process
in one way allows to do the import the other-way around quite easily. So
importing Pull Request from GitHub will be supported, but will be added later.

To configure Git-on-Borg, we used the following resources:

-   http://go/dm-opensource#StageGoB
-   go/gob/users/team-repository (delete?)
-   go/copybara-piper-sot and
    https://g3doc.corp.google.com/devtools/copybara/docs/userdoc/codelabs/piper_to_git.md

Please setup the alias `alias
copybara='/google/data/ro/teams/copybara/copybara'` in your `~/.bashrc`.

### Performing a push and visualizing the result

Before pushing to Git-on-Borg, please first:

-   `g4 sync`
-   Run`scripts/google_run_tests.sh`
-   Run`scripts/google_run_swift_tests.sh`

For the initial push:

-   `copybara copy.bara.sky piper_to_gob --init-history --squash --force` for
    the initial push. (And note the latest CL number!)
-   `git clone "sso://team/deepmind-eng/OpenSpiel"` and look at the result.

For the second push:

-   This step might be unnecessary. Try first without --last-rev.
-   `copybara copy.bara.sky piper_to_gob --last-rev <CL number from above>`.
    Note: It is possible that the --last-rev is only needed because we did not
    have the piper_to_gob transform set with mode="ITERATIVE"

For all subsequent pushes:

-   Run `copybara copy.bara.sky piper_to_gob`.

If you get an error claiming that the commit message is empty (like this:
https://paste.googleplex.com/5354562577760256) then, for every CL that causes
this, it was necessary to force the commit message using:

`copybara copy.bara.sky piper_to_gob --force-message "BEGIN_PUBLIC\nInternal
change\nEND_PUBLIC\n" --iterative-limit-changes 1`

that I found here:
https://groups.google.com/a/google.com/forum/#!msg/copybara-users/E-XdD0KsAQs/rIMyEh_QAwAJ

### Pushing to Github

First, you will need to have write permissions on
https://github.com/deepmind/open_spiel and have added an
[SSH private key](https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
to be able to push to this directory (note that you will need to add the your
ssh-agent every time you create a new shell).

**Preferred solution**

Push to GoB and then

```
git clone "sso://team/deepmind-eng/OpenSpiel"
cd OpenSpiel
# You can look at the commit history using e.g. `gitk`
git remote add github git@github.com:deepmind/open_spiel.git
# See https://help.github.com/en/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
# for how to setup an SSH key to identify on Github.
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github
git push github master
```

NOTE: For the first push, you will need to use `--force` to override the
history.

#### Readthedoc

Website: https://readthedocs.org/ \
Emails in the account: open_spiel@google.com
(primary),lanctot@google.com,jblespiau@google.com \
Username: open_spiel \
Password: Ask jblespiau@, lanctot@, locked@ or vzambaldi@

#### Travis CI

Website: https://travis-ci.org/

### Historical commands to setup CopyBara/Git-on-Borg

This is to document what has been done. Please update it if you perform other
modifications to the GoB configuration/adjust permissions, etc.

-   Added mdformatting support for OpenSpiel in
    cs/devtools/markdown/mdformat/depot_path.cc (cr/248682891)
-   Create a GoB repository using the
    [UI](https://team-review.git.corp.google.com/x/createproject/create), owned
    by `deepmind-eng`
-   [Whitelist](https://g3doc.corp.google.com/devtools/copybara/docs/userdoc/piper_to_git.md?cl=head#before-you-start-whitelisting-your-piper-paths)
    the Piper path for CopyBara (see cr/248707174)
-   Executed the first export (as in
    [this documentation](http://go/copybara-piper-sot#first_run)): `copybara
    copy.bara.sky piper_to_gob --init-history --squash --force`

TODO(jblespiau):

-   Add github account
    (http://go/dm-opensource#4-create-the-external-repository).
-   Setup Git submodules.
-   Change the GoB setup to use the
    [`ITERATIVE`](http://go/copybara-git-sot#importing-all-the-history-iterative-mode).
    mode, and document how to write commit messages. See
    cs/google3/third_party/py/sonnet/METADATA?type=cs&q=sonnet+BEGIN_PUBLIC&g=0&l=53&rcl=247583839
-   Decide about how we want to expose ldaps in the history.
-   Setup https://readthedocs.org.
-   Setup Travis CI.

END GOOGLE-INTERNAL
