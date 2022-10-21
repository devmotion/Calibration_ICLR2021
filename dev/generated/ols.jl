using CairoMakie
using CalibrationErrors
using CalibrationErrorsDistributions
using CalibrationTests
using Distributions
using StatsBase

using Random

using CairoMakie.AbstractPlotting.ColorSchemes: Dark2_8

# set random seed
Random.seed!(1234)

# create path before saving
function wsavefig(file, fig=current_figure())
    mkpath(dirname(file))
    return save(file, fig)
end

xs = rand(Uniform(-1, 1), 100)
ys = rand.(Normal.(sinpi.(xs), 0.15 .* abs.(1 .+ xs)))

bs = hcat(ones(length(xs)), xs) \ ys

stddev = std(bs[1] .+ bs[2] .* xs .- ys)

fig = Figure(; resolution=(960, 450))

# plot the data generating distribution
ax1 = Axis(fig[1, 1]; title="ℙ(Y|X)", xlabel="X", ylabel="Y")
heatmap!(
    -1:0.01:1,
    -2:0.01:2,
    (x, y) -> pdf(Normal(sinpi(x), 0.15 * abs(1 + x)), y);
    colorrange=(0, 1),
)
scatter!(xs, ys; color=Dark2_8[2])
tightlimits!(ax1)

# plot the predictions of the model
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

# link axes and hide y labels and ticks of the second plot
linkaxes!(ax1, ax2)
hideydecorations!(ax2; grid=false)

# add a colorbar
Colorbar(fig[1, 3]; label="density", width=30)

# adjust space
colgap!(fig.layout, 50)

wsavefig("figures/ols/heatmap.pdf")

valxs = rand(Uniform(-1, 1), 50)
valys = rand.(Normal.(sinpi.(valxs), 0.15 .* abs.(1 .+ valxs)))

valps = Normal.(bs[1] .+ bs[2] .* valxs, stddev)

τs = cdf.(valps, valys)

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

# plot the ideal
lines!([0, 1], [0, 1]; label="ideal", linewidth=2, color=Dark2_8[1])

# plot the empirical cdf
sort!(τs)
ecdf_xs = vcat(0, repeat(τs; inner=2), 1)
ecdf_ys = repeat(range(0, 1; length=length(τs) + 1); inner=2)
lines!(ecdf_xs, ecdf_ys; label="data", linewidth=2, color=Dark2_8[2])

# add legend
Legend(fig[1, 2], ax; valign=:top, framevisible=false)

# set limits and aspect ratio
colsize!(fig.layout, 1, Aspect(1, 1))
tightlimits!(ax)

wsavefig("figures/ols/quantiles.pdf")

# define kernel
kernel = WassersteinExponentialKernel() ⊗ SqExponentialKernel()

# compute p-value estimate using bootstrapping
pvalue(AsymptoticSKCETest(kernel, valps, valys); bootstrap_iters=100_000)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

