# spyden

Evaluate the signal-to-noise ratio of a pulse in noisy time series data, in a statistically optimal fashion: with [matched filters](https://en.wikipedia.org/wiki/Matched_filter).
The name stands for **s**ignificance of a **p**ulse buried **de**ep in **n**oise. The **y** is just there to sound cool, really.

### Citation

This will be published as part of a paper at some point. In the meantime, if you are using this module in scientific work, please mention the module name and add a footnote with the link to this repository.

### Rationale

A recurring problem in pulsar astronomy is evaluating the statistical significance of a pulse found in noisy data, in order to determine if something interesting was actually detected of if the pulse is only the product of chance. This boils down to statistical hypothesis testing, and we want the statistical test that can optimally differentiate between noise and actual signal.

In the presence of uncorrelated Gaussian noise (white noise), and if the pulse shape is known _a priori_, the test with the highest statistical power is the convolution of the input data with a properly normalised matched filter with the exact shape of that pulse. The output is the so-called signal to noise ratio. 

`spyden` provides classes and functions to generate banks of properly normalised pulse templates, and to efficiently convolve them with noisy input data in order to find the optimal pulse parameters (shape, width, and time / phase).


### Installation

Clone the repository and type `make install` to install the module in editatble mode with `pip`:

```bash
git clone https://vmorello@bitbucket.org/vmorello/spyden.git
cd spyden/
make install
```

This automatically installs the required dependencies if they are not present.

### Usage

```python
>>> from spyden import TemplateBank, snratio
>>> import numpy as np
>>> import matplotlib.pyplot as plt

# Generate some data, with a Gaussian-ish pulse centered on bin #400
>>> x = np.random.normal(size=1024)
>>> x[398:403] += [4, 14, 22, 14, 4]

# Generate a bank of noise-free pulse templates: here a set of Gaussians 
# with geometrically spaced widths between 1 and 10 bins
>>> bank = TemplateBank.gaussians(np.logspace(0, 1.0, 20))
>>> bank

[Template(size=5, kind=gaussian, w=1.000),
 Template(size=5, kind=gaussian, w=1.129),
 Template(size=5, kind=gaussian, w=1.274),
 Template(size=7, kind=gaussian, w=1.438),
 Template(size=7, kind=gaussian, w=1.624),
 Template(size=7, kind=gaussian, w=1.833),
 Template(size=9, kind=gaussian, w=2.069),
 Template(size=9, kind=gaussian, w=2.336),
 Template(size=9, kind=gaussian, w=2.637),
 Template(size=11, kind=gaussian, w=2.976),
 Template(size=11, kind=gaussian, w=3.360),
 Template(size=13, kind=gaussian, w=3.793),
 Template(size=15, kind=gaussian, w=4.281),
 Template(size=17, kind=gaussian, w=4.833),
 Template(size=19, kind=gaussian, w=5.456),
 Template(size=21, kind=gaussian, w=6.158),
 Template(size=23, kind=gaussian, w=6.952),
 Template(size=25, kind=gaussian, w=7.848),
 Template(size=29, kind=gaussian, w=8.859),
 Template(size=31, kind=gaussian, w=10.000)]


# Call the main function, which estimates the background noise mean and
# standard deviation with robust statistics, and convolves the properly
# normalised templates with the data.
# We will examine each output below
>>> snr, mu, sigma, models = snratio(x, bank)

# mu and sigma are arrays with num_profiles elements, here only one element
# since we passed a single profile as input
>>> mu
array([0.04727856])

>>> sigma
array([0.95543038])

# 'snr' is the full S/N map: a 3D array with shape (num_profiles, num_templates, num_bins)
# Its maximum element indicates both the best-fit template and the phase
# of the pulse as we can see below.
# 'itemp' is the index of the best template in the template bank
# 'ibin' is the bin index in the data with which the reference bin of the best 
# template must be lined up
>>> iprof, itemp, ibin = np.unravel_index(snr.argmax(), snr.shape)
>>> iprof, itemp, ibin
(0, 8, 400)

>>> best_template = bank[itemp]
>>> best_template
Template(size=9, kind=gaussian, w=2.637)

# So here, the best-fit pulse is a gaussian with FWHM = 2.637 bins, centered
# at phase bin #400, as expected.
# Signal to noise ratio:
>>> snr[iprof, itemp, ibin]
31.925846

# And finally, 'models' is the best-fit pulse template for each pulse profile 
# in the input data 'x', properly offset, scaled and phase-shifted.
# 'models' has shape (num_profiles, num_bins)
# We can check that it is a good fit to the data:
>>> plt.figure()
>>> plt.plot(x, label='Input data')
>>> plt.plot(models[0], label='Best-fit pulse', linestyle='--')
>>> plt.legend()
>>> plt.xlim(0, 1024)
>>> plt.show()
```