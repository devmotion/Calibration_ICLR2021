# # Synthetic models
#
# ## Packages

using CSV
using CairoMakie
using CalibrationErrors
using CalibrationErrorsDistributions
using CalibrationTests
using DataFrames
using Distributions
using FillArrays
using ProgressLogging
using Query
using Showoff
using StatsBase

using LinearAlgebra
using Printf
using Random

using CairoMakie.AbstractPlotting.ColorSchemes: Dark2_8

using Logging: with_logger
using TerminalLoggers: TerminalLogger

## set random seed
Random.seed!(1234)

## create path before saving
function wsavefig(file, fig=current_figure())
    mkpath(dirname(file))
    return save(file, fig)
end

## define progress logging frontend
const PROGRESSLOGGER = TerminalLogger()

## define non-intrusive plotting style
set_theme!(
    Theme(;
        Axis=(
            rightspinevisible=false,
            topspinevisible=false,
            xgridvisible=false,
            ygridvisible=false,
        ),
        Legend=(framevisible=false,),
    ),
)

# ## Synthetic models
#
# We study two setups with $d$-dimensional targets $Y$ and normal distributions $P_X$
# of the form $\mathcal{N}(c \mathbf{1}_d, 0.1^2 \mathbf{I}_d)$ as predictions,
# where $c \sim \mathrm{U}(0, 1)$.
# Since calibration analysis is only based on the targets and predicted distributions,
# we neglect features $X$ in these experiments and specify only the distributions of
# $Y$ and $P_X$.
#
# ### Calibrated setup
#
# In the first setup we simulate a calibrated model. We achieve this by sampling
# targets from the predicted distributions, i.e., by defining the conditional distribution
# of $Y$ given $P_X$ as
# ```math
# Y \,|\, P_X = \mathcal{N}(\mu, \Sigma) \sim \mathcal{N}(\mu, \Sigma).
# ```

function calibrated_model(dim::Int, nsamples::Int)
    ## sample predictions
    predictions = [MvNormal(Fill(rand(), dim), 0.1) for _ in 1:nsamples]

    ## sample targets
    targets = map(rand, predictions)

    return predictions, targets
end

# ### Uncalibrated setup
#
# In the second setup we simulate an uncalibrated model of the form
# ```math
# Y \,|\, P_X = \mathcal{N}(\mu, \Sigma) \sim \mathcal{N}([0.1, \mu_2, \ldots, \mu_d], \Sigma).
# ```

function uncalibrated_model(dim::Int, nsamples::Int)
    ## sample predictions
    predictions = [MvNormal(Fill(rand(), dim), 0.1) for _ in 1:nsamples]

    ## sample targets
    targets = map(rand, predictions)
    altdist = Normal(0.1, 0.1)
    for t in targets
        t[1] = rand(altdist)
    end

    return predictions, targets
end

# ## Convergence and computation time of estimators
#
# We perform an evaluation of the convergence and computation time of the biased estimator
# $\widehat{\mathrm{SKCE}}_k$, the unbiased estimator $\widehat{\mathrm{SKCE}}_{k,B}$ with
# blocks of size $B \in \{2, \sqrt{n}, n\}$. We use the tensor product kernel
# ```math
# \begin{aligned}
# k\big((p, y), (p', y')\big) &= \exp{\big(- W_2(p, p')\big)} \exp{\big(-(y - y')^2/2\big)} \\
# &= \exp{\big(-\sqrt{(m_p - m_{p'})^2 + (\sigma_p - \sigma_{p'})^2}\big)} \exp{\big( - (y - y')^2/2\big)},
# \end{aligned}
# ```
# where $W_2$ is the 2-Wasserstein distance and $m_p, m_{p'}$ and $\sigma_p, \sigma_{p'}$
# denote the mean and the standard deviation of the normal distributions $p$ and $p'$.

# ### Ground truth
#
# For both models, we have to "evaluate" the true calibration error. Generally, the error
# depends on the model (and hence also dimension $d$) and the kernel. If the model is
# calibrated, we know that the calibration error is zero. For the uncalibrated model, we
# estimate the ground truth with the minimum-variance unbiased estimator as the mean of
# SKCE estimates for 1000 randomly sampled datasets with 1000 data points.

true_SKCE(::typeof(calibrated_model), kernel; dim::Int) = 0.0
function true_SKCE(model::typeof(uncalibrated_model), kernel; dim::Int)
    estimator = UnbiasedSKCE(kernel)
    return mean(calibrationerror(estimator, model(dim, 1_000)...) for _ in 1:1_000)
end

# ### Benchmarking
#
# The following two functions implement the benchmarking. We sample 500 datasets of
# 4, 16, 64, 256, and 1024 data points each for the models of dimensions $d=1$ and $d=10$.
# For each of the datasets, we evaluate the different SKCE estimators. We compute the
# mean absolute error, the variance, and the minimum computation time for the estimates,
# grouped by the dimension of the model and the number of samples in the dataset.

function benchmark_estimator(estimator, model; dim::Int, nsamples::Int, groundtruth)
    ## compute the estimator (potentially depending on number of samples)
    _estimator = estimator(nsamples)

    ## cache for calibration error estimates
    estimates = Vector{Float64}(undef, 500)

    mintime = Inf

    name = @sprintf("benchmarking (dim = %2d, nsamples = %4d)", dim, nsamples)
    @progress name = name for i in eachindex(estimates)
        ## sample predictions and targets
        predictions, targets = model(dim, nsamples)

        ## define benchmark function
        benchmark_f =
            let estimator = _estimator, predictions = predictions, targets = targets
                () -> @timed calibrationerror(estimator, predictions, targets)
            end

        ## precompile function
        benchmark_f()

        ## compute calibration error and obtain elapsed time
        val, t = benchmark_f()

        ## only keep minimum execution time
        mintime = min(mintime, t)

        ## save error estimate
        estimates[i] = val
    end

    ## save the mean absolute deviation and the variance of the estimates
    meanerror = mean(abs(x - groundtruth) for x in estimates)
    variance = var(estimates)

    return (; dim, nsamples, meanerror, variance, mintime)
end

function benchmark_estimators(model)
    ## output file
    filename = joinpath("data", "synthetic", "errors_$(model).csv")

    ## check if results exist
    isfile(filename) && return DataFrame(CSV.File(filename))

    ## define kernel
    kernel = WassersteinExponentialKernel() ⊗ SqExponentialKernel()

    ## define estimators
    estimators = (
        "SKCE" => _ -> BiasedSKCE(kernel),
        "SKCE (B = 2)" => _ -> BlockUnbiasedSKCE(kernel, 2),
        "SKCE (B = √n)" => n -> BlockUnbiasedSKCE(kernel, max(2, Int(floor(sqrt(n))))),
        "SKCE (B = n)" => _ -> UnbiasedSKCE(kernel),
    )

    ## define number of samples
    nsamples = 2 .^ (2:2:10)

    ## ensure that output directory exists and open file for writing
    mkpath(dirname(filename))
    open(filename, "w") do file
        ## write headers
        println(file, "estimator,dim,nsamples,meanerror,variance,mintime")

        ## for dimensions ``d=1`` and ``d=10``
        for d in (1, 10)
            ## compute/estimate ground truth
            groundtruth = true_SKCE(model, kernel; dim=d)

            for (i, (name, estimator)) in enumerate(estimators)
                ## benchmark estimator
                @info "benchmarking estimator: $(name)"

                for n in nsamples
                    stats = benchmark_estimator(
                        estimator, model; dim=d, nsamples=n, groundtruth=groundtruth
                    )

                    ## save statistics
                    print(file, name, ",")
                    join(file, stats, ",")
                    println(file)
                end
            end
        end
    end

    ## load results
    return DataFrame(CSV.File(filename))
end

# We benchmark the estimators with the calibrated model.

Random.seed!(100)
with_logger(PROGRESSLOGGER) do
    benchmark_estimators(calibrated_model)
end

# We repeat the benchmark with the uncalibrated model.

Random.seed!(100)
with_logger(PROGRESSLOGGER) do
    benchmark_estimators(uncalibrated_model)
end

# ### Visualization
#
# We show a visualization of the results below.

function logtickformat(base::Int)
    function format(values)
        return map(Base.Fix2(logformat, base), showoff(values))
    end
    return format
end

function logformat(digits::String, base::Int)
    buf = IOBuffer()
    print(buf, base)
    for c in digits
        if '0' ≤ c ≤ '9'
            print(buf, Showoff.superscript_numerals[c - '0' + 1])
        elseif c == '-'
            print(buf, '⁻')
        elseif c == '.'
            print(buf, '·')
        end
    end
    return String(take!(buf))
end

function plot_benchmark_estimators(model; dim::Int)
    ## load and preprocess data
    filename = joinpath("data", "synthetic", "errors_$(model).csv")
    groups = @from i in DataFrame(CSV.File(filename)) begin
        @where i.dim == dim
        @orderby i.nsamples
        @select {
            i.estimator,
            log2_nsamples = log2(i.nsamples),
            log10_meanerror = log10(i.meanerror),
            log10_variance = log10(i.variance),
            log10_mintime = log10(i.mintime),
        }
        @collect DataFrame
    end

    ## create figure
    fig = Figure(; resolution=(960, 800))

    ## create axes to plot mean error and variance vs number of samples
    ax1 = Axis(
        fig[1, 1];
        xlabel="# samples",
        ylabel="mean error",
        xticks=2:2:10,
        xtickformat=logtickformat(2),
        ytickformat=logtickformat(10),
    )
    ax2 = Axis(
        fig[2, 1];
        xlabel="# samples",
        ylabel="variance",
        xticks=2:2:10,
        xtickformat=logtickformat(2),
        ytickformat=logtickformat(10),
    )

    ## create axes to plot mean error and variance vs timings
    ax3 = Axis(
        fig[1, 2];
        xlabel="time [s]",
        ylabel="mean error",
        xtickformat=logtickformat(10),
        ytickformat=logtickformat(10),
    )
    ax4 = Axis(
        fig[2, 2];
        xlabel="time [s]",
        ylabel="variance",
        xtickformat=logtickformat(10),
        ytickformat=logtickformat(10),
    )

    ## plot benchmark results
    estimators = ["SKCE", "SKCE (B = 2)", "SKCE (B = √n)", "SKCE (B = n)"]
    markers = ['●', '■', '▲', '◆']
    for (i, (estimator, marker)) in enumerate(zip(estimators, markers))
        group = filter(:estimator => ==(estimator), groups)
        color = Dark2_8[i]

        ## plot mean error vs samples
        scatterlines!(
            ax1,
            group.log2_nsamples,
            group.log10_meanerror;
            color=color,
            linewidth=2,
            marker=marker,
            markercolor=color,
        )

        ## plot variance vs samples
        scatterlines!(
            ax2,
            group.log2_nsamples,
            group.log10_variance;
            color=color,
            linewidth=2,
            marker=marker,
            markercolor=color,
        )

        ## plot mean error vs time
        scatterlines!(
            ax3,
            group.log10_mintime,
            group.log10_meanerror;
            color=color,
            linewidth=2,
            marker=marker,
            markercolor=color,
        )

        ## plot variance vs time
        scatterlines!(
            ax4,
            group.log10_mintime,
            group.log10_variance;
            color=color,
            linewidth=2,
            marker=marker,
            markercolor=color,
        )
    end

    ## link axes and hide decorations
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1)
    linkxaxes!(ax3, ax4)
    hidexdecorations!(ax3)
    linkyaxes!(ax1, ax3)
    hideydecorations!(ax3)
    linkyaxes!(ax2, ax4)
    hideydecorations!(ax4)

    ## add legend
    elems = map(1:length(estimators)) do i
        [
            LineElement(; color=Dark2_8[i], linestyle=nothing, linewidth=2),
            MarkerElement(; color=Dark2_8[i], marker=markers[i], strokecolor=:black),
        ]
    end
    Legend(fig[end + 1, :], elems, estimators; orientation=:horizontal, tellheight=true)

    return fig
end

# We obtain the following plots:

plot_benchmark_estimators(calibrated_model; dim=1)
wsavefig("figures/synthetic/estimators_calibrated_model_dim=1.pdf") #jl
#!jl wsavefig("figures/synthetic/estimators_calibrated_model_dim=1.svg");
#!jl # ![](figures/synthetic/estimators_calibrated_model_dim=1.svg)

plot_benchmark_estimators(calibrated_model; dim=10)
wsavefig("figures/synthetic/estimators_calibrated_model_dim=10.pdf") #jl
#!jl wsavefig("figures/synthetic/estimators_calibrated_model_dim=10.svg");
#!jl # ![](figures/synthetic/estimators_calibrated_model_dim=10.svg)

plot_benchmark_estimators(uncalibrated_model; dim=1)
wsavefig("figures/synthetic/estimators_uncalibrated_model_dim=1.pdf") #jl
#!jl wsavefig("figures/synthetic/estimators_uncalibrated_model_dim=1.svg");
#!jl # ![](figures/synthetic/estimators_uncalibrated_model_dim=1.svg)

plot_benchmark_estimators(uncalibrated_model; dim=10)
wsavefig("figures/synthetic/estimators_uncalibrated_model_dim=10.pdf") #jl
#!jl wsavefig("figures/synthetic/estimators_uncalibrated_model_dim=10.svg");
#!jl # ![](figures/synthetic/estimators_uncalibrated_model_dim=10.svg)

# ## Test errors and computation time of calibration tests
#
# We fix the significance level $\alpha = 0.05$.
# Test predictions are sampled from the same distribution as $P_X$, and test targets are
# sampled independently from $\mathcal{N}(0, 0.1^2 \mathbf{I}_d)$.
#
# ### Benchmarking

iscalibrated(::typeof(calibrated_model)) = true
iscalibrated(::typeof(uncalibrated_model)) = false

function benchmark_test(test, model; dim::Int, nsamples::Int)
    ## number of simulations
    nrepeat = 500

    ## initial values
    ntesterrors = 0
    mintime = Inf

    name = @sprintf("benchmarking (dim = %2d, nsamples = %4d)", dim, nsamples)
    @progress name = name for _ in 1:nrepeat
        ## sample predictions and targets
        predictions, targets = model(dim, nsamples)

        ## define benchmark function
        benchmark_f = let test = test, predictions = predictions, targets = targets
            () -> @timed pvalue(test(predictions, targets))
        end

        ## precompile function
        benchmark_f()

        ## compute calibration error and obtain elapsed time
        val, t = benchmark_f()

        ## only keep minimum execution time
        mintime = min(mintime, t)

        ## update number of empirical test errors for
        ## significance level ``\alpha = 0.05``
        ntesterrors += iscalibrated(model) ⊻ (val ≥ 0.05)
    end

    ## compute empirical test error rate
    testerror = ntesterrors / nrepeat

    return (; dim, nsamples, testerror, mintime)
end

function benchmark_tests(model)
    ## output file
    filename = joinpath("data", "synthetic", "tests_$(model).csv")

    ## check if results exist
    isfile(filename) && return DataFrame(CSV.File(filename))

    ## define kernel
    kernel = WassersteinExponentialKernel() ⊗ SqExponentialKernel()

    ## define number of samples
    nsamples = 2 .^ (2:2:10)

    ## ensure that output directory exists and open file for writing
    mkpath(dirname(filename))
    open(filename, "w") do file
        ## write headers
        println(file, "test,dim,nsamples,testerror,mintime")

        ## for dimensions ``d=1`` and ``d=10``
        for d in (1, 10)
            ## define tests
            testpredictions = [MvNormal(rand(d), 0.1) for _ in 1:10]
            testtargets = [rand(MvNormal(d, 0.1)) for _ in 1:10]
            tests = (
                "SKCE (B = 2)" =>
                    (predictions, targets) -> AsymptoticBlockSKCETest(
                        BlockUnbiasedSKCE(kernel, 2), predictions, targets
                    ),
                "SKCE (B = √n)" =>
                    (predictions, targets) -> AsymptoticBlockSKCETest(
                        BlockUnbiasedSKCE(kernel, Int(floor(sqrt(length(predictions))))),
                        predictions,
                        targets,
                    ),
                "SKCE (B = n)" =>
                    (predictions, targets) ->
                        AsymptoticSKCETest(kernel, predictions, targets),
                "CME" =>
                    (predictions, targets) -> AsymptoticCMETest(
                        UCME(kernel, testpredictions, testtargets), predictions, targets
                    ),
            )

            for (i, (name, test)) in enumerate(tests)
                ## benchmark estimator
                @info "benchmarking test: $(name)"

                for n in nsamples
                    stats = benchmark_test(test, model; dim=d, nsamples=n)

                    ## save statistics
                    print(file, name, ",")
                    join(file, stats, ",")
                    println(file)
                end
            end
        end
    end

    ## load results
    return DataFrame(CSV.File(filename))
end

# First we benchmark the calibrated model.

Random.seed!(100)
with_logger(PROGRESSLOGGER) do
    benchmark_tests(calibrated_model)
end

# We repeat the analysis with the uncalibrated model.

Random.seed!(100)
with_logger(PROGRESSLOGGER) do
    benchmark_tests(uncalibrated_model)
end

# ### Visualization
#
# Again we visualize the results of our benchmarks. However, this time we
# compare the results for the calibrated and the uncalibrated model in the
# same plot.

function plot_benchmark_tests(; dim::Int)
    ## load and preprocess data
    df = mapreduce(vcat, (calibrated_model, uncalibrated_model)) do model
        filename = joinpath("data", "synthetic", "tests_$(model).csv")
        df = DataFrame(CSV.File(filename))
        df[!, :model] .= string(model)
        return df
    end
    groups = @from i in df begin
        @where i.dim == dim
        @orderby i.nsamples
        @select {
            i.test,
            i.model,
            log2_nsamples = log2(i.nsamples),
            i.testerror,
            log10_mintime = log10(i.mintime),
        }
        @collect DataFrame
    end

    ## create figure
    fig = Figure(; resolution=(960, 400))

    ## add labels
    Label(fig[1:2, 1], "empirical test error"; rotation=π / 2, tellheight=false)
    Label(fig[1, 2:3, Top()], "calibrated model"; padding=(0, 0, 10, 0))
    Label(fig[2, 2:3, Top()], "uncalibrated model"; padding=(0, 0, 10, 0))

    ## create axes to plot test error vs number of samples
    ax1 = Axis(
        fig[1, 2];
        ylabel="type I error",
        xticks=2:2:10,
        xtickformat=logtickformat(2),
        xticklabelsize=12,
        yticklabelsize=12,
    )
    ax2 = Axis(
        fig[2, 2];
        xlabel="# samples",
        ylabel="type II error",
        xticks=2:2:10,
        xtickformat=logtickformat(2),
        xticklabelsize=12,
        yticklabelsize=12,
    )

    ## create axes to plot test error vs timings
    ax3 = Axis(
        fig[1, 3]; xtickformat=logtickformat(10), xticklabelsize=12, yticklabelsize=12
    )
    ax4 = Axis(
        fig[2, 3];
        xlabel="time [s]",
        xtickformat=logtickformat(10),
        xticklabelsize=12,
        yticklabelsize=12,
    )

    ## plot benchmark results
    tests = ["SKCE (B = 2)", "SKCE (B = √n)", "SKCE (B = n)", "CME"]
    markers = ['●', '■', '▲', '◆']
    for (i, (test, marker)) in enumerate(zip(tests, markers))
        color = Dark2_8[i]

        ## for both calibrated and uncalibrated model
        for (axes, model) in
            zip(((ax1, ax3), (ax2, ax4)), (calibrated_model, uncalibrated_model))
            group = filter(x -> x.test == test && x.model == string(model), groups)

            ## plot test error vs samples
            scatterlines!(
                axes[1],
                group.log2_nsamples,
                group.testerror;
                color=color,
                linewidth=2,
                marker=marker,
                markercolor=color,
            )

            ## plot test error vs timings
            scatterlines!(
                axes[2],
                group.log10_mintime,
                group.testerror;
                color=color,
                linewidth=2,
                marker=marker,
                markercolor=color,
            )
        end
    end

    ## plot horizontal lines for significance level
    for axis in (ax1, ax3)
        hlines!(axis, 0.05; color=:black, linestyle=:dash, linewidth=2)
    end

    ## link axes and hide decorations
    linkxaxes!(ax1, ax2)
    hidexdecorations!(ax1)
    linkxaxes!(ax3, ax4)
    hidexdecorations!(ax3)
    linkyaxes!(ax1, ax3)
    hideydecorations!(ax3)
    linkyaxes!(ax2, ax4)
    hideydecorations!(ax4)

    ## add legend
    elems = map(1:length(tests)) do i
        [
            LineElement(; color=Dark2_8[i], linestyle=nothing, linewidth=2),
            MarkerElement(; color=Dark2_8[i], marker=markers[i], strokecolor=:black),
        ]
    end
    push!(elems, [LineElement(; color=:black, linestyle=:dash, linewidth=2)])
    Legend(
        fig[1:2, end + 1],
        elems,
        vcat(tests, "significance level");
        tellwidth=true,
        gridshalign=:left,
    )

    return fig
end

plot_benchmark_tests(; dim=1)
wsavefig("figures/synthetic/tests_dim=1.pdf") #jl
#!jl wsavefig("figures/synthetic/tests_dim=1.svg");
#!jl # ![](figures/synthetic/tests_dim=1.svg)

plot_benchmark_tests(; dim=10)
wsavefig("figures/synthetic/tests_dim=10.pdf") #jl
#!jl wsavefig("figures/synthetic/tests_dim=10.svg");
#!jl # ![](figures/synthetic/tests_dim=10.svg)
