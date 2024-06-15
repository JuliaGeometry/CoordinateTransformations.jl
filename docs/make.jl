using CoordinateTransformations
using Documenter

DocMeta.setdocmeta!(
    CoordinateTransformations,
    :DocTestSetup,
    :(using CoordinateTransformations),
    recursive = true,
)

makedocs(
    sitename = "CoordinateTransformations.jl",
    pages = [
        "Introduction" => "index.md",
        "API" => "api.md",
    ],
)
