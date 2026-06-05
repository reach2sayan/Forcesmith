# Extending Forcesmith

Forcesmith has two polymorphic surfaces a user is expected to extend: **potentials**
(the radial functions being fitted) and **solvers** (the optimisation
algorithms). Both are implemented as Sean-Parent-style **value-semantic type
erasure** — there is no virtual hierarchy to inherit from and no enum to edit.
You write a plain struct that satisfies a small compile-time contract, then hand
an instance to the API.

A consequence worth stating up front: the **CLI menus are intentionally fixed**.
`--algorithm` offers `lm | powell | de | ls`, and a model file selects analytic
potentials by `"type"` name from a registry. These menus are *not* the extension
mechanism. A brand-new potential or solver is supplied **programmatically**,
through `forcesmith::Forcesmith`, without recompiling the engine or widening any menu.

---

## Adding a new Potential

### The contract

A potential is any type with these members (see the `Potential` erasure in
`include/forcesmith/core/potential_base.hpp`):

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

Wrap the value in the type-erased `Potential` and place it into a session. This
is the direct analogue of injecting a solver, and needs **no registry edit and
no engine recompile**:

```cpp
forcesmith::Forcesmith session;
session.set_pair_potential("Cu", "Cu", forcesmith::Potential{MyExp(0.34, 1.4, 0.0, 2.0, 6.0)});
// also: set_density(elem, …), set_embedding(elem, …),
//       set_dipole / set_quadrupole (ADP), set_radial / set_angular (angular)
```

`Potential` accepts any type satisfying the contract above (`Potential.hpp:87`),
so your struct drops straight in.

### Optional: expose it to file-driven runs

The CLI (and `io::load_model`) build potentials from a model file by their
`"type"` name. If — and only if — you also want your potential loadable *from a
file*, add one line to the maker registry in `src/io/potential_reader.cpp`
(around line 54):

```cpp
add(m, /*nparams*/ 3, {"A","B","C"},
    [](auto p, auto lo, auto hi) { return Potential(MyExp(p[0], p[1], p[2], lo, hi)); },
    "myexp");                       // one or more JSON "type" aliases
```

This widens the fixed file menu; it is not required for API use. The `_sc`
smooth-cutoff variants show the one-liner decorator pattern — `SmoothCutoff`
wraps any base analytic potential and appends a switching width `h`, so a cutoff
variant is also a single registry line.

---

## Adding a new Solver

### The contract

A solver is any type satisfying the `SolverImpl` concept
(`include/forcesmith/optimization/solver.hpp`):

```cpp
int minimize(Eigen::VectorXd& x, ResidualFn f, JacobianFn jac, int n_vals) const;
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
- returns an integer status code (solver-specific; ≥0 conventionally success).

Add a `static_assert` so a contract break is a compile error, exactly like the
built-ins:

```cpp
struct MyGradientDescent {
  int max_iter = 500;
  double step = 1e-3;
  int minimize(Eigen::VectorXd& x, forcesmith::ResidualFn f,
               forcesmith::JacobianFn jac, int n_vals) const {
    Eigen::MatrixXd J(n_vals, x.size());
    for (int it = 0; it < max_iter; ++it) {
      const Eigen::VectorXd F = f(x);
      if (jac) jac(x, J); else { /* finite-difference J here */ }
      x -= step * (J.transpose() * F);     // gradient of ½‖F‖²
    }
    return 0;
  }
};
static_assert(forcesmith::SolverImpl<MyGradientDescent>);
```

The built-in solvers (`EigenLMSolver`, `EigenHybridSolver`, `BoostDESolver`,
`LineSearchSolver`) are good references for handling the empty-Jacobian case and
for carrying their own tuning fields.

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
four built-in names; there is no need to touch it for a library extension.)

---

## Why it works this way

`Potential` and `Solver` are both type-erased value types built on the small
`detail::ErasedValue` / `detail::ErasedMoveOnly` helpers
(`include/forcesmith/core/erased.hpp`). The engine manipulates them through their
public value interface and never knows the concrete type — so adding a potential
or a solver is a purely *additive*, client-side change. This is the project's
standard polymorphism style: value semantics with type erasure for these
customisation-point surfaces, CRTP (`AnalyticBase`) for families of closely
related concrete types, and no enum/switch dispatch or open virtual hierarchies.
