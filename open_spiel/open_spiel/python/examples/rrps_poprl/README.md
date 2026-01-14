This code implements the PopRL algorithm described in Lanctot et al.
[Population-based Evaluation in Repeated Rock-Paper-Scissors as a Benchmark for
Multiagent Reinforcement Learning](https://openreview.net/forum?id=gQnJ7ODIAx)

The implementation of IMPALA is an online agent version of the IMPALA example in
the Haiku codebase. It has been modified to add prediction labels, which get
stored in the environment.

Checkpointing is not working for technical reasons (some nontrivial parts are
needed to handle Haiku functions / models). It needs to be fixed if this is to
run for long periods of time or in interactive mode.

This implementation is NOT designed for scale.

The code is provided as-is. It's a direct conversion of the code used for the
paper but it has not been extensively tested after the transformation. The basic
tests work and the transformation was straight-forward. However, if you run into
any trouble, please contact lanctot@google.com.
