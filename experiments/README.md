# Experiments

This folder contains the implementation and the results of the experiments in
the paper
["Calibration tests beyond classification"](https://openreview.net/forum?id=-bxf89v3Nx)
by [David Widmann](http://www.it.uu.se/katalog/davwi492),
[Fredrik Lindsten](https://liu.se/en/employee/freli29), and
[Dave Zachariah](https://www.it.uu.se/katalog/davza513), which will be presented at
[ICLR 2021](https://iclr.cc/Conferences/2021).

## Overview

We performed three experiments, called `ols`, `synthetic` and `friedman`. Their implementations
can be found in the folders `src/ols`, `src/synthetic`, and `src/friedman`, respectively. The results
and figures of the experiments are contained in the folders `data/synthetic`, `data/friedman`, and
`figures/ols`, `figures/synthetic`, and `figures/friedman`.

## Run experiments

Apart from the machine-dependent benchmarks, all results can be reproduced exactly.

### Install Julia

The experiments were performed with Julia 1.5.3. You can download the official binaries from
the [Julia webpage](https://julialang.org/downloads/).

For increased reproducibility, the `nix` environment in this folder provides a fixed Julia
binary:
1. Install [nix](https://github.com/NixOS/nix#installation).
2. Navigate to this folder and activate the environment by running
   ```shell
   nix-shell
   ```
   in a terminal. Alternatively, if you use [lorri](https://github.com/target/lorri), you can
   activate the environment by executing
   ```shell
   direnv allow
   ```
   in a terminal.

### (Re)move results

The experiments are only performed if the results in the `data` folder do not exist. Therefore
you have to (re)move the results that you want to regenerate. Existing figures are overwritten
automatically when you run the experiments.

### Install dependencies

Of course, the experiments depend on different Julia packages. The exact set and version of these
dependencies that were used to generate the results and plots in the paper are pinned for every
experiment separately. To install the Julia packages, execute
```shell
julia --startup-file=no --project=src/EXPERIMENT -e 'using Pkg; Pkg.instantiate()'
```
in a terminal in this folder, where `EXPERIMENT` has to be replaced with the name of the experiment
that you want to run (i.e., with `ols`, `synthetic`, or `friedman`).

The option `--startup-file=no` increases reproducibility by eliminating any user-specific
customizations. You may add other arguments such as `--color=yes` if you prefer colorized output.
If you have activated the project environment with `nix-shell` or `lorri`, the alias `j` can be used
for calling `julia` with the arguments `--startup-file=no --color=yes`.

### Run script

You can run the desired experiment in a terminal with
```shell
julia --startup-file=no --project=src/EXPERIMENT src/EXPERIMENT/script.jl
```
where again `EXPERIMENT` has to be replaced with the name of the experiment.

As mentioned above, `--startup-file=no` increases reproducibility and other command line arguments
such as `--color=yes` could be added.
