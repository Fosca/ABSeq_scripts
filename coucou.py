#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script. Present a few example applications for the Markov Model toolbox.
To do next:
    - add a user-defined prior bias in hmm case
@author: Florent Meyniel
"""
# general
import matplotlib.pyplot as plt
import numpy as np

# specific to toolbox
from MarkovModel_Python import IdealObserver as IO
from MarkovModel_Python import GenerateSequence as sg

# %% Binary sequence and order 0 transition probabilities

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order0(np.hstack((0.25*np.ones(L), 0.75*np.ones(L))))
seq = sg.ConvertSequence(seq)['seq']

# Compute observer with fixed decay and HMM observers
# The decay is exponential: a value of N means that each observation is discounted by a factor
# exp(-1/N) at every time step.
# In the HMM observer, p_c is the a priori probability of a change point on each trial
# For HMM, 'resol' is the number on bins used for numeric estimation of integrals on a grid.
options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# PRIORS for fixed decay (or window)
# The prior is specified as the parameters of the corresponding beta/dirichlet distribution.
# If the prior is symmetric, simply use e.g. options['prior_weight'] = 10 (all parameters will be
# 10). The defaults is options['prior_weight'] = 1
# For a custom prior, use e.g. options['custom_prior'] = {(0,): 1, (1,): 5} to specify the parameter
# corresponding to each item (or transition)

# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(out_fixed[(0,)]['mean'], label='p(0) mean')
plt.plot(out_fixed[(0,)]['SD'], linestyle='--', label='p(0) sd')
plt.legend(loc='best'), plt.ylim([0, 1])
plt.title('Exponential decay')

plt.subplot(3, 1, 2)
plt.imshow(out_hmm[(0,)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('HMM -- full distribution')

plt.subplot(3, 1, 3)
plt.plot(out_hmm[(0,)]['mean'], label='p(0) mean')
plt.plot(out_hmm[(0,)]['SD'], linestyle='--', label='p(0) sd')
plt.legend(loc='best'), plt.ylim([0, 1])
plt.title('HMM -- moments')
plt.tight_layout()

# %% Binary sequence and order 1 coupled transition probabilities

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))),
                np.hstack((0.75*np.ones(L), 0.25*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute Decay observer and HMM
options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=1, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)


# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(out_fixed[(0, 0,)]['mean'], label='p(0|0)')
plt.plot(out_fixed[(1, 0,)]['mean'], label='p(0|1)')
plt.legend(loc='best')
plt.title('mean)')
plt.ylim([0, 1])

plt.subplot(3, 1, 2)
vmin, vmax = 0, np.max([np.max(out_hmm[(0, 0)]['dist']), np.max(out_hmm[(1, 0)]['dist'])])
plt.imshow(out_hmm[(0, 0)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(0|0)')

plt.subplot(3, 1, 3)
plt.imshow(out_hmm[(1, 0)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(0|1)')
plt.tight_layout()

# %% Sequence of 3 items and order 0 transition probabilities

# Generate sequence
L = int(1e2)
Prob = {0: np.hstack((0.10*np.ones(L), 0.50*np.ones(L))),
        1: np.hstack((0.10*np.ones(L), 0.20*np.ones(L))),
        2: np.hstack((0.80*np.ones(L), 0.30*np.ones(L)))}
seq = sg.ProbToSequence_Nitem3_Order0(Prob)
seq = sg.ConvertSequence(seq)['seq']

options = {'Decay': 10, 'p_c': 1/200, 'resol': 20}
out_fixed = IO.IdealObserver(seq, 'fixed', order=0, options=options)
out_hmm = IO.IdealObserver(seq, 'hmm', order=0, options=options)

# Plot result
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(out_fixed[(0,)]['mean'], label='p(0)')
plt.plot(out_fixed[(1,)]['mean'], label='p(1)')
plt.plot(out_fixed[(2,)]['mean'], label='p(2)')
plt.legend(loc='best')
plt.title('mean')
plt.ylim([0, 1])
plt.subplot(4, 1, 2)
vmin, vmax = 0, np.max([np.max(out_hmm[(0,)]['dist']),
                        np.max(out_hmm[(1,)]['dist']),
                        np.max(out_hmm[(2,)]['dist'])])
plt.imshow(out_hmm[(0,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(0)')
plt.subplot(4, 1, 3)
plt.imshow(out_hmm[(1,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(1)')
plt.subplot(4, 1, 4)
plt.imshow(out_hmm[(2,)]['dist'], origin='lower', vmin=vmin, vmax=vmax)
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.ylabel('p(2)')
plt.tight_layout()

# %% Binary sequence and order 1 transition probability: coupled vs. uncoupled

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.75*np.ones(L), 0.25*np.ones(L))),
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute HMM observer for coupled (simply specify 'hmm') and uncoupled case (specified with
# 'hmm_uncoupled')
options = {'p_c': 1/200, 'resol': 20}
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)
out_hmm_unc = IO.IdealObserver(seq, 'hmm_uncoupled', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(out_hmm[(0, 0)]['mean'], 'g', label='p(0|0), coupled')
plt.plot(out_hmm[(1, 0)]['mean'], 'b', label='p(0|1), coupled')
plt.plot(out_hmm_unc[(0, 0)]['mean'], 'g--', label='p(0|0), unc.')
plt.plot(out_hmm_unc[(1, 0)]['mean'], 'b--', label='p(0|1), unc.')
plt.legend(loc='upper left')
plt.ylim([0, 1])
plt.title('Comparison of means')

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(-np.log(out_hmm[(0, 0)]['SD']), 'g', label='p(0|0), coupled')
plt.plot(-np.log(out_hmm[(1, 0)]['SD']), 'b', label='p(0|1), coupled')
plt.plot(-np.log(out_hmm_unc[(0, 0)]['SD']), 'g--', label='p(0|0), uncoupled')
plt.plot(-np.log(out_hmm_unc[(1, 0)]['SD']), 'b--', label='p(0|1), uncoupled')
plt.legend(loc='upper left')
plt.title('Comparison of confidence')

# %% Estimate volatility of a binary sequence with order 1 coupled transition probability

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.75*np.ones(L), 0.25*np.ones(L))),
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute HMM, but assumes (rather than learn) a given volatility level. Specify 'hmm' for this.
options = {'resol': 20, 'p_c': 1/L}
out_hmm = IO.IdealObserver(seq, 'hmm', order=1, options=options)

# Compute the full HMM that learn volatility from the data. Specify 'hmm+full' for this.
grid_nu = 1/2 ** np.array([k/2 for k in range(20)])
options = {'resol': 20, 'grid_nu': grid_nu, 'prior_nu': np.ones(20)/20}
out_hmm_full = IO.IdealObserver(seq, 'hmm+full', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(3, 1, 1)
plt.imshow(out_hmm_full['volatility'])
expo = np.log(grid_nu)/np.log(1/2)
plt.yticks(ticks=[0, len(expo)/2, len(expo)],
                  labels=[f"1/{2**expo[0]:.0f}",
                          f"1/{2**expo[int(len(expo)/2)]:.0f}",
                          f"1/{2**expo[-1]:.0f}"])
plt.title('Volatility estimate')

plt.subplot(3, 1, 2)
plt.imshow(out_hmm_full[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), full inference')

plt.subplot(3, 1, 3)
plt.imshow(out_hmm[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), assuming vol.=1/L')
plt.tight_layout()

# %% Estimate volatility of a binary sequence with order 1 uncoupled transition probability

# Generate sequence
L = int(1e2)
seq = sg.ProbToSequence_Nitem2_Order1(
        np.vstack((
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L))),
                np.hstack((0.25*np.ones(L), 0.75*np.ones(L)))
                )))
seq = sg.ConvertSequence(seq)['seq']

# Compute the full HMM that learn volatility from the data, while assuming that transition
# probabilities have uncoupled change points. Specify 'hmm_uncoupled' for this.
grid_nu = 1/2 ** np.array([k/2 for k in range(20)])
options = {'resol': 20, 'grid_nu': grid_nu, 'prior_nu': np.ones(20)/20}
out_hmm_unc_full = IO.IdealObserver(seq, 'hmm_uncoupled+full', order=1, options=options)
out_hmm_full = IO.IdealObserver(seq, 'hmm+full', order=1, options=options)

# Compute the HMM that assumes a given volatility level, for the uncoupled case.
options = {'resol': 20, 'p_c': 1/L}
out_hmm_unc = IO.IdealObserver(seq, 'hmm_uncoupled', order=1, options=options)

# Plot result
plt.figure()
plt.subplot(4, 1, 1)
plt.imshow(out_hmm_unc_full['volatility'])
expo = np.log(grid_nu)/np.log(1/2)
plt.yticks(ticks=[0, len(expo)/2, len(expo)],
                  labels=[f"1/{2**expo[0]:.0f}",
                          f"1/{2**expo[int(len(expo)/2)]:.0f}",
                          f"1/{2**expo[-1]:.0f}"])
plt.title('Volatility estimate (unc)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(4, 1, 2)
plt.imshow(out_hmm_unc_full[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), full inference (unc)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(4, 1, 3)
plt.imshow(out_hmm_unc[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), vol=1/L (unc)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.subplot(4, 1, 4)
plt.imshow(out_hmm_full[(0, 0)]['dist'], origin='lower')
plt.yticks(ticks=[0, options['resol']/2, options['resol']], labels=[0, 0.5, 1])
plt.title('p(0|0), full inference (coupled)', **{'fontname': 'Arial', 'size': '12'})
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tight_layout()

# %% Example of inference for sequences with chunks (or pauses)


def remove_pause(seq, post, Nitem):
    """
    Remove the pause (i.e. the items whose value is >= Nitem) from the posterior
    dictionary returned by the ideal observer.
    """
    ind = seq < Nitem
    new_post = {}
    for key in post.keys():
        if type(post[key]) is dict:
            new_post[key] = {}
            for sub_key in post[key].keys():
                new_post[key][sub_key] = post[key][sub_key][ind]
        else:
            new_post[key] = post[key][ind]
    return new_post


# Make a sequence by repeating three times the same pattern
# in seq_with_pause, we insert a new item between patterns to denote the pause.
pattern = [0]*4 + [1]*4 + [0, 0, 1, 1]*2
seq_of_patterns = [pattern]*3
seq_continuous = np.hstack(seq_of_patterns)
seq_with_pause = sg.ConvertSequence(np.hstack([chunk+[max(pattern)+1]
                                               for chunk in seq_of_patterns])[:-1])['seq']

# options of the ideal observer
order = 1
options = {'Decay': 20, 'prior_weight': 1}

# Compute the inference on the sequence with pauses
# In this case, Nitem=2 will force the code to ignore the 3rd item which we inserted to denote
# the pause.
# Then we remove those pauses from the sequence of posterior values, in order to match the size of
# the continous sequence.
post_with_pause = IO.IdealObserver(seq_with_pause, 'fixed', Nitem=2,
                                   order=order, options=options)
post_chunk = remove_pause(seq_with_pause, post_with_pause, 2)

# Compute the inference on the continous sequence (it really ignores the pause)
post_continous = IO.IdealObserver(seq_continuous, 'fixed', Nitem=2,
                                  order=order, options=options)

# Compute the inference that completely resets after a pause
post_reset = []
for chunk in seq_of_patterns:
    post_reset.append(IO.IdealObserver(sg.ConvertSequence(chunk)['seq'],
                                       'fixed', order=order, options=options))

# Plot result
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(seq_continuous, 'o')
plt.plot([15.5]*2, [-1, 2], 'k')
plt.plot([31.5]*2, [-1, 2], 'k')
plt.title('Sequence')

plt.subplot(4, 1, 2)
plt.plot(post_chunk['surprise'], 'o')
plt.ylim([0, 4])
plt.plot([15.5]*2, [-1, 4], 'k')
plt.plot([31.5]*2, [-1, 4], 'k')
plt.title('Surprise (inference with chunks)')

plt.subplot(4, 1, 3)
plt.plot(post_continous['surprise'], 'o')
plt.ylim([0, 4])
plt.plot([15.5]*2, [-1, 4], 'k')
plt.plot([31.5]*2, [-1, 4], 'k')
plt.title('Surprise (continous inference)')

plt.subplot(4, 1, 4)
plt.plot(np.hstack([post['surprise'] for post in post_reset]), 'o')
plt.plot([15.5]*2, [-1, 4], 'k')
plt.plot([31.5]*2, [-1, 4], 'k')
plt.ylim([0, 4])
plt.title('Surprise (Inference that resets)')
plt.tight_layout()

