using Literate: Literate
using Pkg: Pkg

# Print `@debug` statements (https://github.com/JuliaDocs/Documenter.jl/issues/955)
if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

const EXPERIMENTS = ("ols", "synthetic", "friedman")
const INPUT = joinpath(@__DIR__, "..", "experiments")
const OUTPUT = joinpath(@__DIR__, "src", "generated")

ispath(OUTPUT) && rm(OUTPUT; recursive=true)
mkpath(OUTPUT)

# Link existing data to avoid expensive computations
if isdir(joinpath(INPUT, "data"))
    symlink(joinpath(INPUT, "data"), joinpath(OUTPUT, "data"))
end

# Add links and explanations below the first heading of level 1
function preprocess(content)
    sub = SubstitutionString(
        """
\\0
#
#md # [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/generated/@__NAME__.ipynb)
#
# You are seeing the
#md # HTML output generated by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl) and
#nb # notebook output generated by
# [Literate.jl](https://github.com/fredrikekre/Literate.jl) from the
# [Julia source file](@__REPO_ROOT_URL__/experiments/src/@__NAME__/script.jl).
# The corresponding
#md # notebook can be viewed in [nbviewer](@__NBVIEWER_ROOT_URL__/generated/@__NAME__.ipynb),
#nb # HTML output can be viewed [here](https://devmotion.github.io/Calibration_ICLR2021/dev/generated/@__NAME__/),
# and the plain script output can be found [here](./@__NAME__.jl).
# 
#md # !!! note
#     If you want to run the experiments, make sure you have an identical environment.
#     Please use Julia $(VERSION) and activate and instantiate the environment using
#     [this Project.toml file](@__REPO_ROOT_URL__/experiments/src/@__NAME__/Project.toml)
#     and [this Manifest.toml file](@__REPO_ROOT_URL__/experiments/src/@__NAME__/Manifest.toml).
#
#     [The Github repository](@__REPO_ROOT_URL__) contains [more
#     detailed instructions](@__REPO_ROOT_URL__/experiments/README.md) and a
#     `nix` project environment with a pinned Julia binary for improved reproducibility.
#md #
#md # ```@setup @__NAME__
#md # using Pkg: Pkg
#md # Pkg.activate("$(INPUT)/src/@__NAME__")
#md # Pkg.instantiate()
#md # ```
#
        """,
    )
    return replace(content, r"^# # [^\n]*"m => sub; count=1)
end

for name in EXPERIMENTS
    file = joinpath(INPUT, "src", name, "script.jl")

    # Activate project environment
    Pkg.activate(dirname(file)) do
        Pkg.instantiate()

        # Export and execute output formats
        Literate.markdown(file, OUTPUT; name=name, documenter=true, preprocess=preprocess)
        Literate.notebook(file, OUTPUT; name=name, documenter=true, preprocess=preprocess)
        Literate.script(file, OUTPUT; name=name, documenter=true)
    end
end

using Documenter

makedocs(;
    authors="David Widmann <david.widmann@it.uu.se>",
    sitename="Calibration tests beyond classification",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://devmotion.github.io/Calibration_ICLR2021",
        assets=String[],
    ),
    pages=[
        "index.md",
        "software.md",
        "Experiments" => [joinpath("generated", "$(name).md") for name in EXPERIMENTS],
    ],
)

# Remove link to existing (would be broken when deployed)
if islink(joinpath(@__DIR__, "build", "generated", "data"))
    rm(joinpath(@__DIR__, "build", "generated", "data"))
end

deploydocs(;
    repo="github.com/devmotion/Calibration_ICLR2021.git",
    push_preview=true,
    devbranch="main",
)
