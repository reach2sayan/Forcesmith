# Extending Forcesmith

Forcesmith has three polymorphic surfaces a user is expected to extend:
**potentials** (the radial functions being fitted), **solvers** (the
optimisation algorithms), and **energy heads** (the descriptor → energy map of an
ML model). All three are implemented as Sean-Parent-style **value-semantic type
erasure** — there is no virtual hierarchy to inherit from and no enum to edit.
You write a plain struct that satisfies a small compile-time contract, then hand
an instance to the API.

A consequence worth stating up front: the **CLI menus are intentionally fixed**.
`--algorithm` offers `lm | powell | de | ls | ipopt`, and a model file selects
analytic potentials by `"type"` name from a registry. These menus are *not* the
extension mechanism. A brand-new potential, solver, or head is supplied
**programmatically**, through `forcesmith::Forcesmith`, without recompiling the
engine or widening any menu.

---

## Adding a new RadialPotential

### The contract

A potential is any type with these members (see the `RadialPotential` erasure in
`include/forcesmith/core/radial_potential.hpp`):

```cpp
double eval(double r) const;                          // value at r
double deriv(double r) const;                         // dV/dr at r
std::pair<double,double> span() const;                // {rmin, rmax}
std::size_t param_count() const;                      // # of FREE params
void gather_params(Eigen::VectorXd& dst, std::size_t off) const;   // free params → vector
void scatter_params(const Eigen::VectorXd& src, std::size_t off);  // vector → free params
```

These are optional and detected with `if constexpr` / customisation points:

- `set_param(std::size_t i, double v)` / `set_fixed(std::size_t i, bool)` —
  needed if a global parameter is broadcast into this potential, or to pin a
  parameter out of the fit. Absent ⇒ treated as no-ops.
- Curvature (smoothness) regularisation via the `curvature_count` /
  `write_curvature` free-function CPO in
  `include/forcesmith/potentials/curvature.hpp`. The default contributes **0**
  curvature residuals, which is correct for analytic potentials (only tabulated
  splines need it).

### The easy route: `AnalyticBase`

For an analytic function, derive from `AnalyticBase<Derived, N>`
(`include/forcesmith/potentials/analytic_potential.hpp`). The base owns the `N`
parameters and the cutoff, and supplies `param_count` / `gather_params` /
`scatter_params` / `set_param` / `set_fixed` for free. You only write
`eval_impl(double)` and `deriv_impl(double)`. This is exactly how `Morse`,
`LennardJones`, `Eopp`, and ~30 others are defined:

```cpp
// V(r) = A·exp(-B·r) + C   — params: {A, B, C}
struct MyExp : forcesmith::AnalyticBase<MyExp, 3> {
  constexpr MyExp(double A, double B, double C, double lo, double hi)
      : AnalyticBase({A, B, C}, lo, hi) {}
  constexpr double eval_impl(double r) const {
    const auto [A, B, C] = params;
    return A * std::exp(-B * r) + C;
  }
  constexpr double deriv_impl(double r) const {
    const auto [A, B, C] = params;
    return -A * B * std::exp(-B * r);
  }
};
```

A tabulated potential should instead model itself on `SplinePotential`
(`include/forcesmith/potentials/spline.hpp`), which carries the spline's own
curvature CPO so `--smooth-weight` regularisation works.

### Using it — the recommended path (API injection)

Wrap the value in the type-erased `RadialPotential` and place it into a session. This
is the direct analogue of injecting a solver, and needs **no registry edit and
no engine recompile**:

```cpp
forcesmith::Forcesmith session;
session.set_pair_potential("Cu", "Cu", forcesmith::RadialPotential{MyExp(0.34, 1.4, 0.0, 2.0, 6.0)});
// also: set_density(elem, …), set_embedding(elem, …),
//       set_dipole / set_quadrupole (ADP), set_radial / set_angular (angular)
```

`RadialPotential` accepts any type satisfying the contract above (`RadialPotential.hpp:87`),
so your struct drops straight in.

### Optional: expose it to file-driven runs

The CLI (and `io::load_model`) build potentials from a model file by their
`"type"` name. If — and only if — you also want your potential loadable *from a
file*, add one line to the maker registry in `src/io/potential_reader.cpp`
(around line 54):

```cpp
add(m, /*nparams*/ 3, {"A","B","C"},
    [](auto p, auto lo, auto hi) { return RadialPotential(MyExp(p[0], p[1], p[2], lo, hi)); },
    "myexp");                       // one or more JSON "type" aliases
```

This widens the fixed file menu; it is not required for API use. The `_sc`
smooth-cutoff variants show the one-liner decorator pattern — `SmoothCutoff`
wraps any base analytic potential and appends a switching width `h`, so a cutoff
variant is also a single registry line.

---

## Adding a new Solver

### The contract

A solver is any type satisfying the `CSolver` concept
(`include/forcesmith/optimization/solver.hpp`):

```cpp
int minimize(Eigen::VectorXd& x, ResidualFn f, JacobianFn jac, int n_vals,
             const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) const;
```

- `x` — the parameter vector, in/out: seeded with the start values, overwritten
  with the optimum. Its size `D` is the number of free parameters.
- `f` — `ResidualFn = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>`;
  maps parameters to the residual vector F(x) (length `n_vals`). The objective
  is ½‖F‖².
- `jac` — `JacobianFn`; fills ∂F/∂x. **It may be empty** — check `if (jac)`. An
  empty Jacobian means none was supplied, so a solver that needs one should fall
  back to its own finite differences.
- `n_vals` — the residual count (rows of the Jacobian).
- `lower` / `upper` — full-length box constraints aligned element-for-element
  with `x`, ±∞ where a parameter is unbounded. Only solvers that report
  `honors_bounds() == true` (Ipopt, DE) apply them; the gradient solvers ignore
  the two extra arguments. `honors_bounds()` is **optional** — add
  `bool honors_bounds() const { return true; }` to apply the box; omit it and the
  erased `Solver` defaults it to `false`.
- returns an integer status code (solver-specific; ≥0 conventionally success).

Add a `static_assert` so a contract break is a compile error, exactly like the
built-ins:

```cpp
struct MyGradientDescent {
  int max_iter = 500;
  double step = 1e-3;
  int minimize(Eigen::VectorXd& x, forcesmith::ResidualFn f,
               forcesmith::JacobianFn jac, int n_vals,
               const Eigen::VectorXd& /*lower*/,
               const Eigen::VectorXd& /*upper*/) const {
    Eigen::MatrixXd J(n_vals, x.size());
    for (int it = 0; it < max_iter; ++it) {
      const Eigen::VectorXd F = f(x);
      if (jac) jac(x, J); else { /* finite-difference J here */ }
      x -= step * (J.transpose() * F);     // gradient of ½‖F‖²
    }
    return 0;
  }
  // bool honors_bounds() const { return true; }  // opt in to honour lower/upper
};
static_assert(forcesmith::CSolver<MyGradientDescent>);
```

The built-in solvers (`EigenLMSolver`, `EigenHybridSolver`, `BoostDESolver`,
`LineSearchSolver`, `IpoptSolver`) are good references for handling the
empty-Jacobian case and for carrying their own tuning fields.

### Using it — API injection (the only path needed)

Wrap the value in the type-erased `Solver` and set it on the session:

```cpp
session.set_solver(forcesmith::Solver{MyGradientDescent{.max_iter = 1000}});
BOOST_LEAF_CHECK(session.optimize());
```

If no solver is set, `optimize()` falls back to the default Levenberg–Marquardt
(`make_default_solver`). A client solver **does not** go on the CLI: the
`--algorithm` menu is fixed by design, and custom algorithms are supplied
programmatically. (The CLI's `build_solver` in `src/cli/app.cpp` only maps the
five built-in names; there is no need to touch it for a library extension.)

---

## Adding a new ML energy head

An ML model (the `ml` family) is a per-atom **descriptor** (ACSF, SOAP, LMBTR)
feeding an **energy head** that maps the descriptor vector D to an atomic energy.
The head is the customisation surface — the type-erased `EnergyHead`
(`include/forcesmith/force/ml_force.hpp`). The shipped head is `LinearHead`
(E = c·D + bias); a non-linear head is a new struct, not an engine edit.

### The contract

A head supplies the maths and its fittable slots; everything else is generated.
The minimum a concrete head provides:

```cpp
double          energy(const Eigen::VectorXd& D) const;   // E_i(D)
Eigen::VectorXd grad(const Eigen::VectorXd& D) const;      // dE/dD
std::vector<Param*>       field_ptrs();                    // fittable slots
std::vector<const Param*> field_ptrs() const;
std::string      type_tag() const;                        // serialised tag
std::vector<int> architecture() const;                    // shape, for I/O
```

Derive from the `HeadParams<Derived>` CRTP mixin
(`include/forcesmith/force/ml_force.hpp`) and it synthesises
`param_count` / `gather_params` / `scatter_params` / `gather_bounds` and
(de)serialisation from your `field_ptrs()` — the same free-vs-`fixed` convention
as potentials. `Param` carries `{value, fixed, min, max}`.

A head may optionally expose an **analytic parameter-Jacobian** so the fit avoids
finite-differencing the head:

```cpp
bool            has_param_jacobian() const;               // false ⇒ FD fallback
Eigen::VectorXd param_grad(const Eigen::VectorXd& D) const;     // dE/dθ
Eigen::MatrixXd dgrad_dparam(const Eigen::VectorXd& D) const;   // d(dE/dD)/dθ
```

For multi-element re-ranking, a head also implements `remapped(map)` (re-index its
coefficients to a new descriptor layout) and `zero_like(n)` (a fresh zero head of
`n` features for a newly-added element). `LinearHead` is the worked reference for
all of the above.

### Using it — API injection

Heads are attached to the ML model in memory, then seeded as the force model:

```cpp
Model m = /* descriptor (ACSF/SOAP/LMBTR), ntypes 1 */;
m.heads.emplace_back(forcesmith::EnergyHead{MyHead{/* sized to */ m.descriptor_size()}});
```

Descriptors themselves (ACSF, SOAP, LMBTR) follow the project's `MLBase<Derived>`
CRTP pattern; adding a brand-new descriptor is the heavier change and is the
exception to the "purely additive, client-side" rule, since the descriptor layout
and its gradient feed the force engine directly.

---

## Adding a new ForceCalculator

A **force calculator** is the top-level model the optimiser fits: it owns the
potentials/heads, builds neighbour lists, and turns a `Configuration` into
forces/energy/stress. `ForceCalculator`
(`include/forcesmith/force/force_calculator.hpp`) is the type-erased value that
holds one — the open replacement for the old closed `std::variant`. A new
calculator (analytic family, a new bond-order form, …) drops in **without
editing `force_calculator.hpp`**.

### The contract

Satisfy the `ForceCalculatorModel` concept
(`include/forcesmith/force/force_calculator_concept.hpp`) — the methods the
optimiser and evaluator always call:

```cpp
void eval_forces(Configuration& cfg) const;        // forces/energy/stress/limit
double max_cutoff() const;                          // neighbour-list cutoff
std::size_t param_count() const;                    // # of FREE params
void gather_params(Eigen::VectorXd& dst, std::size_t off) const;
void scatter_params(const Eigen::VectorXd& src, std::size_t off);
void gather_bounds(Eigen::VectorXd& lo, Eigen::VectorXd& hi, std::size_t off) const;
```

Plus a public `std::size_t ntypes` member (inherit `ForceCalculatorBase<Derived>`
to get `ntypes`/`conf_index`), and the curvature CPO for `--smooth-weight`
regularisation (`model_smoothness_count` / `model_write_smoothness` in
`include/forcesmith/force/smoothness.hpp` — the templated default contributes 0,
overload it if your calculator owns tabulated tables).

### The descriptor-cache capability — mix in `FitCache`

The erased `ForceCalculator` also calls a fit fast-path interface
(`has_cache`/`has_param_jacobian`/`prepare`/`eval_cached`/`eval_cached_jacobian`/
`head_param_counts` and an indexed `eval_forces(cfg, cache_index)`). These exist
only on ML models (`MLBase` implements the real descriptor cache). Every other
calculator gets the correct **no-cache** defaults by mixing in
`FitCache<Derived>` (the analogue of `NoGlobals`/`WithGlobals`):

```cpp
struct MyForce : ForceCalculatorBase<MyForce>, NoGlobals, FitCache<MyForce> {
  std::size_t /* …state… */;
  void eval_forces(Configuration& cfg) const;
  using FitCache<MyForce>::eval_forces;  // expose the indexed (no-cache) overload
  std::size_t param_count() const; /* … the rest of the contract … */
};
static_assert(forcesmith::ForceCalculatorModel<MyForce>);
```

The `using` re-exposes the mixin's `eval_forces(cfg, cache_index)` overload that
your own `eval_forces(cfg)` would otherwise hide. `FitCache<Derived, true>` is
the opt-in for a calculator that *does* carry a cache — it must then override the
cached hooks itself.

### Using it — API injection

```cpp
forcesmith::Forcesmith session;
session.seed_force_model(forcesmith::ForceCalculator{MyForce{/* … */}});
```

The escape hatch `calc.target<MyForce>()` (mirrors `std::function::target`)
recovers the concrete type for the few typed call sites that stay outside the
core concept on purpose — pair-table edits, native file writers
(`src/io/write_model.cpp`), and spec decomposition (the `PotentialType` trait).
Those three enumerate the known families, so exposing a *new* calculator to
**file output** or **multi-element re-rank** still means adding to those lists
(and to the model reader in `src/io/force_model_reader.cpp`); fitting and
evaluating it needs none of that.

## Why it works this way

`RadialPotential` and `Solver` are both type-erased value types built on the small
`detail::ErasedValue` / `detail::ErasedMoveOnly` helpers
(`include/forcesmith/core/erased.hpp`). The engine manipulates them through their
public value interface and never knows the concrete type — so adding a potential
or a solver is a purely *additive*, client-side change. This is the project's
standard polymorphism style: value semantics with type erasure for these
customisation-point surfaces, CRTP (`AnalyticBase`) for families of closely
related concrete types, and no enum/switch dispatch or open virtual hierarchies.
