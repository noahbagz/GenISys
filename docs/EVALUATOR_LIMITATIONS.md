# Evaluator scope

The simulator supports preliminary bulkcarrier design. Its engineering
equations, numerical grids, parameter cleanup, and constraint calculations
are retained from the existing source. This packaging task does not change
the ship physics or establish a new engineering validation.

The paper's feasibility label uses its stated counted constraint set.
Additional diagnostic outputs, including optional flooding calculations, must
be distinguished from that label. Preserve each run's evaluation status and
constraint records; a failed or incomplete evaluation is not a feasible design.

Structural CAD consists of neutral surfaces and tabulated section properties.
Full-system visualization also contains display proxies. These representations
are preliminary design and illustration tools rather than detailed production
drawings or classification approval.

Sampling produces 133 independent controls. The retained decoder reconstructs
the 189-parameter evaluator interface, with the existing discrete representation
and dependency rules. Sampling alone does not establish feasibility, and
requests outside the supported representation may fail evaluation.
