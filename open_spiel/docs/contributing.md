# Guidelines

Above all, OpenSpiel is designed to be easy to install and use, easy to
understand, easy to extend (“hackable”), and general/broad. OpenSpiel is built
around two major important design criteria:

-   **Keep it simple.** Simple choices are preferred to more complex ones. The
    code should be readable, usable, extendable by non-experts in the
    programming language(s), and especially to researchers from potentially
    different fields. OpenSpiel provides reference implementations that are used
    to learn from and prototype with, rather than fully-optimized /
    high-performance code that would require additional assumptions (narrowing
    the scope / breadth) or advanced (or lower-level) language features.

-   **Keep it light.** Dependencies can be problematic for long-term
    compatibility, maintenance, and ease-of- use. Unless there is strong
    justification, we tend to avoid introducing dependencies to keep things easy
    to install and more portable.

# Support expectations

We, the OpenSpiel authors, definitely engage in supporting the community. As it
can be time-consuming, we try to find a good balance between ensuring we are
responsive and being able to continue to do our day-to-day work and research.

Generally speaking, if you are willing to get a specific feature implemented,
the most effective way is to implement it and send a Pull Request. For large
changes, or ones involving design decisions, open a bug to check the idea is ok
first.

The higher the quality, the easier it will be to be accepted. For instance,
following the
[C++ Google style guide](https://google.github.io/styleguide/cppguide.html) and
[Python Google style guide](http://google.github.io/styleguide/pyguide.html)
will help with the integration.

As examples, MacOS support, Window support, example improvements, various
bug-fixes or new games has been straightforward to be included and we are very
thankful to everyone who helped.

## Bugs

We aim to answer bugs at a reasonable pace, several times a week. However, for
bugs involving large changes (e.g. adding new games, adding public state
supports) we cannot commit to implementing it and encourage everyone to
contribute directly.

## Pull requests

You can expect us to answer/comment back and you will know from the comment if
it will be merged as is or if it will need additional work.

For pull requests, they are merged as batches to be more efficient, at least
every two weeks (for bug fixes, it will likely be faster to be integrated). So
you may need to wait a little after it has been approved to actually see it
merged.

# OpenSpiel visual Graph

To help you understand better the framework as a whole you can go to
[openspielgraph](https://openspielgraph.netlify.app) and use an interactive
graph that shows the OpenSpiel repository in a wide and easy to undestand way.

By providing intuitive visual representations, it simplifies the debugging
process, aids in the optimization of algorithms, and fosters a more efficient
workflow.

For a practical example, see one of the reasons OpenSpielGraph was thought of
and also how to use OpenSpiel and WebAssembly...

# Roadmap and Call for Contributions

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). See
[CONTRIBUTING.md](https://github.com/deepmind/open_spiel/blob/master/CONTRIBUTING.md)
for the details.

Here, we outline our current highest priorities: this is where we need the most
help. There are also suggestion for larger features and research projects. Of course,
all contributions are welcome. 

Before making a contribution to OpenSpiel, please read the guidelines. We also
kindly request that you contact us before writing any large piece of code, in
case (a) we are already working on it and/or (b) it's something we have already
considered and may have some design advice on its implementation. Please also
note that some games may have copyrights which might require legal approval.
Otherwise, happy hacking!

-   **Long-term and Ongoing Maintenance**. This is the most important way to help.
    Having OpenSpiel bug-free and working smoothly is the highest priority. Things
    can stop working for a variety of reasons due to version changes and backward
    incompatibility, but also due to discovering new problems that require some time
    to fix. To see these items, look for issues with the "help wanted" tag on the
    [Issues page](https://github.com/google-deepmind/open_spiel/issues).

-   **New Features and Algorithms**. There are regular requests for new features
    and algorithms that we just don't have time to provide. Look for issues with the
    "contribution welcome" tag on the
    [Issues page](https://github.com/google-deepmind/open_spiel/issues).

-   **Windows support**. Native Windows support was added in early 2022, but
    remains experimental and only via building from source. It would be nice to
    have Github Actions CI support on Windows to ensure that Windows support is
    actively maintained, and eventually support installing OpenSpiel via pip on
    Windows as well. The tool that builds the binary wheels (cibuildwheel)
    already supports Windows as a target platform.

-   **Visualizations of games**. There exists an interactive viewer for
    OpenSpiel games called [SpielViz](https://github.com/michalsustr/spielviz).
    Contributions to this project, and more visualization tools with OpenSpiel,
    are very welcome as they could help immensely with debugging and testing
    the AI beyond the console.

-   **Structured Action Spaces**. Currently, actions are integers between 0 and
    some value. There is no easy way to interpret what each action means in a
    game-specific way. Nor is there any way to easily represent a composite
    action in terms of its parts. A structured action space could represent
    actions as a sequence of values (like information states and observations--
    and can also include shapes) which can be learned instead of mappings to
    flat numbers. Then, each game could have a mapping from the structured
    action to the action taken.

-   **APIs for other languages** (Go, Rust, Julia). We currently have these
    supported but little beyond the core API and random simulation tests. Several
    are very basic (or experimental). It would be nice to properly support these
    by having a few simple algorithms run via the bindings on OpenSpiel games. 

-   **New Games**. New games are always welcome. If you do not have one in mind,
    check out the
    [Call for New Games](https://github.com/google-deepmind/open_spiel/issues/843)
    issue.

