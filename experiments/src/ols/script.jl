# # Ordinary least squares
#
# ## Packages

using CairoMakie
using CalibrationErrors
using CalibrationErrorsDistributions
using CalibrationTests
using Distributions
using StatsBase

using Random

using CairoMakie.AbstractPlotting.ColorSchemes: Dark2_8

## set random seed
Random.seed!(1234)

## create path before saving
function wsavefig(file, fig=current_figure())
    mkpath(dirname(file))
    return save(file, fig)
end

# ## Regression problem
#
# We consider a regression problem with scalar feature $X$ and scalar
# target $Y$ with input-dependent Gaussian noise that is inspired by a
# problem by [Gustafsson, Danelljan, and Schön](https://arxiv.org/abs/1906.01620).
# Feature $X$ is distributed uniformly at random in $[-1, 1]$,
# and target $Y$ is distributed according to
# ```math
# Y \sim \sin(\pi X) + | 1 + X | \epsilon,
# ```
# where $\epsilon \sim \mathcal{N}(0, 0.15^2)$.

# We start by generating a data set consisting of 100 i.i.d. pairs of
# feature $X$ and target $Y$:

xs = rand(Uniform(-1, 1), 100)
ys = rand.(Normal.(sinpi.(xs), 0.15 .* abs.(1 .+ xs)))

# ## Ordinary least squares regression
#
# We perform ordinary least squares regression for this nonlinear heteroscedastic
# regression problem, and train a model $P$ with homoscedastic variance. The fitted
# parameters of the model are

bs = hcat(ones(length(xs)), xs) \ ys

# and the standard deviation of the model is given by

stddev = std(bs[1] .+ bs[2] .* xs .- ys)

# The following plot visualizes the training data set and model $P$, together
# with the function $f(x) = \mathbb{E}[Y | X = x] = \sin(\pi x)$.

fig = Figure(; resolution=(960, 450))

## plot the data generating distribution
ax1 = Axis(fig[1, 1]; title="ℙ(Y|X)", xlabel="X", ylabel="Y")
heatmap!(
    -1:0.01:1,
    -2:0.01:2,
    (x, y) -> pdf(Normal(sinpi(x), 0.15 * abs(1 + x)), y);
    colorrange=(0, 1),
)
scatter!(xs, ys; color=Dark2_8[2])
tightlimits!(ax1)

## plot the predictions of the model
ax2 = Axis(fig[1, 2]; title="P(Y|X)", xlabel="X", ylabel="Y")
heatmap!(
    -1:0.01:1,
    -2:0.01:2,
    let offset = bs[1], slope = bs[2], stddev = stddev
        (x, y) -> pdf(Normal(offset + slope * x, stddev), y)
    end;
    colorrange=(0, 1),
)
scatter!(xs, ys; color=Dark2_8[2])
tightlimits!(ax2)

## link axes and hide y labels and ticks of the second plot
linkaxes!(ax1, ax2)
hideydecorations!(ax2; grid=false)

## add a colorbar
Colorbar(fig[1, 3]; label="density", width=30)

## adjust space
colgap!(fig.layout, 50)

wsavefig("figures/ols/heatmap.pdf") #jl
#!jl wsavefig("figures/ols/heatmap.svg");
#!jl # ![](figures/ols/heatmap.svg)

# ## Validation
#
# We evaluate calibration of the model with a validation data set of $n = 50$ i.i.d. pairs
# of samples $(X_1, Y_1), \ldots, (X_n, Y_n)$ of $(X, Y)$.

valxs = rand(Uniform(-1, 1), 50)
valys = rand.(Normal.(sinpi.(valxs), 0.15 .* abs.(1 .+ valxs)))

# For these validation data points we compute the predicted distributions $P(Y | X = X_i)$.

valps = Normal.(bs[1] .+ bs[2] .* valxs, stddev)

# ## Quantile calibration
#
# We evaluate the predicted cumulative probability $\tau_i = P(Y \leq Y_i | X = X_i)$ for
# each validation data point.

τs = cdf.(valps, valys)

# The following plot visualizes the empirical cumulative distribution function of the
# predicted quantiles.

fig = Figure(; resolution=(600, 450))

ax = Axis(
    fig[1, 1];
    xlabel="quantile level",
    ylabel="cumulative probability",
    xticks=0:0.25:1,
    yticks=0:0.25:1,
    autolimitaspect=1,
    rightspinevisible=false,
    topspinevisible=false,
    xgridvisible=false,
    ygridvisible=false,
)

## plot the ideal
lines!([0, 1], [0, 1]; label="ideal", linewidth=2, color=Dark2_8[1])

## plot the empirical cdf
sort!(τs)
ecdf_xs = vcat(0, repeat(τs; inner=2), 1)
ecdf_ys = repeat(range(0, 1; length=length(τs) + 1); inner=2)
lines!(ecdf_xs, ecdf_ys; label="data", linewidth=2, color=Dark2_8[2])

## add legend
Legend(fig[1, 2], ax; valign=:top, framevisible=false)

## set limits and aspect ratio
colsize!(fig.layout, 1, Aspect(1, 1))
tightlimits!(ax)

wsavefig("figures/ols/quantiles.pdf") #jl
#!jl wsavefig("figures/ols/quantiles.svg");
#!jl # ![](figures/ols/quantiles.svg)

# ## Calibration test
#
# We compute a $p$-value estimate of the null hypothesis that model $P$ is calibrated using
# an estimation of the quantile of the asymptotic distribution of
# $n \widehat{\mathrm{SKCE}}_{k,n}$ with 100000 bootstrap samples on the validation data
# set. Kernel $k$ is chosen as a tensor product kernel of two Gaussian kernels
# ```math
# \begin{aligned}
# k\big((p, y), (p', y')\big) &= \exp{\big(- W_2(p, p')\big)} \exp{\big(-(y - y')^2/2\big)} \\
# &= \exp{\big(-\sqrt{(m_p - m_{p'})^2 - (\sigma_p - \sigma_{p'})^2}\big)} \exp{\big( - (y - y')^2/2\big)},
# \end{aligned}
# ```
# where $W_2$ is the 2-Wasserstein distance and $m_p, m_{p'}$ and $\sigma_p, \sigma_{p'}$
# denote the mean and the standard deviation of the normal distributions $p$ and $p'$.

## define kernel
kernel = WassersteinExponentialKernel() ⊗ SqExponentialKernel()

## compute p-value estimate using bootstrapping
pvalue(AsymptoticSKCETest(kernel, valps, valys); bootstrap_iters=100_000)

# We obtain $p < 0.05$ in our experiment, and hence the calibration test rejects $H_0$ at
# the significance level $\alpha = 0.05$.
