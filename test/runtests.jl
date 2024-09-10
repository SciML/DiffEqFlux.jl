using ReTestItems, InteractiveUtils, Hwloc
using DiffEqFlux

@info sprint(versioninfo)

const GROUP = lowercase(get(ENV, "GROUP", "all"))

const RETESTITEMS_NWORKERS = parse(
    Int, get(ENV, "RETESTITEMS_NWORKERS", string(min(Hwloc.num_physical_cores(), 16))))
const RETESTITEMS_NWORKER_THREADS = parse(Int,
    get(ENV, "RETESTITEMS_NWORKER_THREADS",
        string(max(Hwloc.num_virtual_cores() ÷ RETESTITEMS_NWORKERS, 1))))

@info "Running tests for group: $GROUP with $RETESTITEMS_NWORKERS workers"

ReTestItems.runtests(DiffEqFlux; tags = (GROUP == "all" ? nothing : [Symbol(GROUP)]),
    nworkers = RETESTITEMS_NWORKERS,
    nworker_threads = RETESTITEMS_NWORKER_THREADS, testitem_timeout = 3600)
