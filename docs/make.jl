using CoordinateTransformations
using Documenter
using Documenter.Remotes: GitHub

DocMeta.setdocmeta!(
    CoordinateTransformations,
    :DocTestSetup,
    :(using CoordinateTransformations),
    recursive = true,
)

makedocs(
    sitename = "CoordinateTransformations.jl",
    modules = [CoordinateTransformations],
    repo = GitHub("JuliaGeometry/CoordinateTransformations.jl"),
    pages = [
        "Introduction" => "index.md",
        "API" => "api.md",
    ],
)
