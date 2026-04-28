# Installation

`IntegratedNestedLaplace.jl` is published in the
[JuliaRegistry](https://github.com/haavardhvarnes/JuliaRegistry) private
registry, together with its three subpackages
([`INLACore`](https://github.com/haavardhvarnes/IntegratedNestedLaplace.jl/tree/main/dev/INLACore),
[`INLAModels`](https://github.com/haavardhvarnes/IntegratedNestedLaplace.jl/tree/main/dev/INLAModels),
and
[`INLASpatial`](https://github.com/haavardhvarnes/IntegratedNestedLaplace.jl/tree/main/dev/INLASpatial)).

## One-time registry setup

You only need to add the registry once per Julia installation:

```julia
using Pkg
pkg"registry add https://github.com/haavardhvarnes/JuliaRegistry.git"
```

The General registry is also required — it ships with Julia, so unless
you removed it deliberately you already have it.

## Adding the package

After the registry has been added, install the package with the regular
`Pkg.add`:

```julia
using Pkg
Pkg.add("IntegratedNestedLaplace")
```

The three subpackages (`INLACore`, `INLAModels`, `INLASpatial`) are
pulled in automatically as transitive dependencies. They live in the
`dev/` subdirectory of the
[main repository](https://github.com/haavardhvarnes/IntegratedNestedLaplace.jl)
and are registered with the same `repo` URL plus a `subdir` entry —
nothing extra is needed on the user side.

## Updating

Updates are picked up by `Pkg.update`:

```julia
using Pkg
Pkg.update("IntegratedNestedLaplace")
```

## Developing from a local clone

If you want to modify the package or any of its subpackages:

```julia
using Pkg
Pkg.activate(".")
Pkg.develop([
    Pkg.PackageSpec(path = "dev/INLACore"),
    Pkg.PackageSpec(path = "dev/INLAModels"),
    Pkg.PackageSpec(path = "dev/INLASpatial"),
])
Pkg.instantiate()
```

The `Pkg.develop` calls override the registered versions of the three
subpackages with the local working copies under `dev/`, so changes are
picked up immediately on the next `using IntegratedNestedLaplace`.

## Troubleshooting

* **`could not find package IntegratedNestedLaplace`** — the JuliaRegistry
  is not added. Run the `pkg"registry add ..."` line above.
* **`expected package … to be registered`** — Pkg may have an outdated
  cache. Run `pkg"registry update"` and try again.
* **Authentication prompts when cloning the registry** — the registry is
  public; HTTPS works without credentials. If you hit credential
  prompts, double-check the URL spelling.
