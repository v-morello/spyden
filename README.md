# spyden

Evaluate the signal-to-noise ratio of a pulse in noisy time series data, in a statistically optimal fashion: with [matched filters](https://en.wikipedia.org/wiki/Matched_filter).

### Citation

This will be published as part of a paper at some point. In the meantime, if you are using this module in scientific work, please mention the module name and add a footnote with the link to this repository.

### Rationale

A recurring problem in pulsar astronomy is evaluating the statistical significance of a pulse found in noisy data, in order to determine if something interesting was actually detected of if the pulse is only the product of chance. This boils down to statistical hypothesis testing, and we want the statistical test that can optimally differentiate between noise and actual signal.

In the presence of uncorrelated Gaussian noise (white noise), and if the pulse shape is known _a priori_, the test with the highest statistical power is the convolution of the input data with a properly normalised matched filter with the exact shape of that pulse. The output is the so-called signal to noise ratio. 

`spyden` provides classes and functions to generate banks of properly normalised pulse templates, and to efficiently convolve them with noisy input data in order to find the optimal pulse parameters (shape, width, and time / phase).


### Installation

Clone the repository and type `make install` to install the module in editatble mode with `pip`:

```bash
git clone https://vmorello@bitbucket.org/vmorello/psnr.git
cd psnr/
make install
```

This automatically installs the required dependencies if they are not present.

### Usage

TODO.