# Software

The empirical evaluations in the paper are performed with [the
Julia programming language](https://julialang.org/).[^Julia2017]

[^Julia2017]: Bezanson, J., Edelman, A., Karpinski, S., & Shah, V. (2017). Julia: A fresh approach to numerical computing. *SIAM Review*, 59(1), 65–98.

## Calibration analysis

### Julia packages

* [**CalibrationErrors.jl**](https://github.com/devmotion/CalibrationErrors.jl): This
  package implements different estimators of the expected calibration error (ECE),
  the squared kernel calibration (SKCE), and the unnormalized calibration mean
  embedding (UCME) in the Julia language.
* [**CalibrationErrorsDistributions.jl**](https://github.com/devmotion/CalibrationErrorsDistributions.jl):
  This package extends calibration error estimation for classification models in the
  package CalibrationErrors.jl to more general probabilistic predictive models that
  output arbitrary probability distributions, as proposed in our paper.
* [**CalibrationTests.jl**](https://github.com/devmotion/CalibrationTests.jl):
  This package contains statistical hypothesis tests of calibration.

### Python interface

The Python package [**pycalibration**](https://github.com/devmotion/pycalibration)
is a wrapper of the Julia packages CalibrationErrors.jl, CalibrationErrorsDistributions.jl,
and CalibrationTests.jl and exposes all their functionality to Python users with
[PyJulia](https://github.com/JuliaPy/pyjulia).

### R interface

Similarly, the R package [**rcalibration**](https://github.com/devmotion/rcalibration)
is an interface of CalibrationErrors.jl, CalibrationErrorsDistributions.jl,
and CalibrationTests.jl for R. It is based on [JuliaCall](https://github.com/Non-Contradiction/JuliaCall).
