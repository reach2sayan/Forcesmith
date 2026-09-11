#pragma once

#include "forcesmith/core/fit_params.hpp"
#include "forcesmith/core/param.hpp"
#include "forcesmith/core/radial_potential.hpp" // NoSiteCache
#include "forcesmith/core/symbolic.hpp" // name_of, slot_of
#include "forcesmith/core/types.hpp"

#include "ddx.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace forcesmith {

namespace sym {
using ddx::var;

inline constexpr auto r = var<"r">;       // the radius: the free variable
inline constexpr auto rmax = var<"rmax">; // the form's own cutoff (not a param)

inline constexpr auto A = var<"A">;
inline constexpr auto B = var<"B">;
inline constexpr auto C = var<"C">;
inline constexpr auto D = var<"D">;
inline constexpr auto E = var<"E">;
inline constexpr auto F = var<"F">;
inline constexpr auto G = var<"G">;
inline constexpr auto H = var<"H">;
inline constexpr auto I = var<"I">;
inline constexpr auto J = var<"J">;
inline constexpr auto K = var<"K">;
inline constexpr auto L = var<"L">;

inline constexpr auto a = var<"a">;
inline constexpr auto b = var<"b">;
inline constexpr auto c = var<"c">;
inline constexpr auto d = var<"d">;
inline constexpr auto h = var<"h">;
inline constexpr auto k = var<"k">;
inline constexpr auto m = var<"m">;
inline constexpr auto n = var<"n">;
inline constexpr auto p = var<"p">;
inline constexpr auto q = var<"q">;

inline constexpr auto a0 = var<"a0">;
inline constexpr auto a1 = var<"a1">;
inline constexpr auto a2 = var<"a2">;
inline constexpr auto a3 = var<"a3">;
inline constexpr auto a4 = var<"a4">;
inline constexpr auto c1 = var<"c1">;
inline constexpr auto c2 = var<"c2">;
inline constexpr auto c3 = var<"c3">;
inline constexpr auto c4 = var<"c4">;
inline constexpr auto c5 = var<"c5">;
inline constexpr auto D1 = var<"D1">;
inline constexpr auto D2 = var<"D2">;
inline constexpr auto r0 = var<"r0">;
inline constexpr auto r1 = var<"r1">;
inline constexpr auto r2 = var<"r2">;
inline constexpr auto R = var<"R">;
inline constexpr auto S = var<"S">;

inline constexpr auto De = var<"De">;
inline constexpr auto re = var<"re">;
inline constexpr auto E0 = var<"E0">;
inline constexpr auto beta = var<"beta">;
inline constexpr auto chi = var<"chi">;
inline constexpr auto delta = var<"delta">;
inline constexpr auto epsilon = var<"epsilon">;
inline constexpr auto gamma = var<"gamma">;
inline constexpr auto lambda = var<"lambda">;
inline constexpr auto mu = var<"mu">;
inline constexpr auto omega = var<"omega">;
inline constexpr auto phi = var<"phi">;
inline constexpr auto rc = var<"rc">;
inline constexpr auto rho = var<"rho">;
inline constexpr auto sigma = var<"sigma">;
} // namespace sym

struct ParamDefault {
  double value;
  double min;
  double max;
};

template <class T>
concept CAnalyticForm = requires {
  { T::names };
  { T::param_names };
  { T::defaults };
  { T::expr };
  { T::num_params } -> std::convertible_to<std::size_t>;
};

namespace detail {

inline constexpr std::size_t kSlotR = static_cast<std::size_t>(-1);
inline constexpr std::size_t kSlotRmax = static_cast<std::size_t>(-2);

template <CAnalyticForm Form, symbolic::CEquation Eq>
inline constexpr auto symbol_map = [] {
  using Syms = typename Eq::symbols;
  std::array<std::size_t, boost::mp11::mp_size<Syms>::value> slots{};
  std::size_t k = 0;
  boost::mp11::mp_for_each<
      boost::mp11::mp_transform<boost::mp11::mp_identity, Syms>>([&](auto tag) {
    const std::string_view name = symbolic::name_of<typename decltype(tag)::type>();
    slots[k] =
        name == "r" ? kSlotR
        : name == "rmax"
            ? kSlotRmax
            : std::ranges::distance(Form::param_names.begin(),
                                    std::ranges::find(Form::param_names, name));
    ++k;
  });
  return slots;
}();

template <CAnalyticForm Form>
inline constexpr auto equation = ddx::Equation{Form::expr};
template <CAnalyticForm Form>
inline constexpr std::size_t r_slot = [] {
  constexpr std::size_t k =
      symbolic::slot_of<std::remove_cvref_t<decltype(equation<Form>)>>("r");
  return k == symbolic::npos ? std::size_t{0} : k;
}();
template <CAnalyticForm Form>
inline constexpr auto derivative =
    ddx::Equation{equation<Form>[ddx::idx<r_slot<Form> + 1>()]};

constexpr auto cosine_cutoff(auto rr, auto lo, auto hi) {
  return 0.5 +
         0.5 * cos(std::numbers::pi * (min(max(rr, lo), hi) - lo) / (hi - lo));
}

constexpr auto switching_factor(auto rr, auto r0_, auto h_) {
  return [](auto u) {
    return ((u * u) * (u * u)) / (1.0 + (u * u) * (u * u));
  }(min((rr - r0_) / h_, 0.0));
}

} // namespace detail

template <typename Derived, std::size_t N>
struct AnalyticForm : ParamSet<AnalyticForm<Derived, N>>, NoSiteCache<Derived> {
  static constexpr std::size_t num_params = N;

protected:
  std::array<Param, N> params;
  double rmin, rmax;

public:
  constexpr AnalyticForm(std::array<double, N> vals, double lo,
                         double hi) noexcept
      : rmin(lo), rmax(hi) {
    std::ranges::transform(vals, params.begin(),
                           [](double v) { return Param{v}; });
  }

  template <std::convertible_to<double>... Ds>
    requires(sizeof...(Ds) == N + 2)
  constexpr explicit AnalyticForm(Ds... args) noexcept {
    const std::array<double, N + 2> all{static_cast<double>(args)...};
    for (std::size_t i = 0; i < N; ++i) {
      params[i] = Param{all[i]};
    }
    rmin = all[N];
    rmax = all[N + 1];
  }

  constexpr std::pair<double, double> span() const { return {rmin, rmax}; }

  constexpr void set_fixed(std::size_t i, bool f) { params[i].fixed = f; }
  constexpr bool is_fixed(std::size_t i) const { return params[i].fixed; }
  constexpr void set_param(std::size_t i, double v) { params[i].value = v; }

  constexpr auto param_fields() { return std::views::all(params); }
  constexpr auto param_fields() const { return std::views::all(params); }

  constexpr void set_bounds(std::size_t i, double lo, double hi) {
    params[i].min = lo;
    params[i].max = hi;
  }

  // Bind symbols to parameters by name: ddx sorts them alphabetically, so a
  // positional point would silently read the wrong parameter.
  template <symbolic::CEquation Eq, class Use>
  FORCE_INLINE constexpr decltype(auto) at_point(const Eq &eq, double r,
                                                 Use &&use) const {
    constexpr auto slots = detail::symbol_map<Derived, std::remove_cvref_t<Eq>>;
    // param_names and the sym:: objects in expr spell the same names twice; a
    // typo in either would silently leave a symbol unbound (and index past the
    // parameter array), so it has to fail here instead.
    static_assert(std::ranges::all_of(slots,
                                      [](std::size_t i) {
                                        return i == detail::kSlotR ||
                                               i == detail::kSlotRmax || i < N;
                                      }),
                  "a symbol in this form's expression is not one of its "
                  "param_names (or the reserved r / rmax)");
    return [&]<std::size_t... K>(std::index_sequence<K...>) -> decltype(auto) {
      return use(eq, (slots[K] == detail::kSlotR ? r
                      : slots[K] == detail::kSlotRmax
                          ? rmax
                          : params[slots[K]].value)...);
    }(std::make_index_sequence<slots.size()>{});
  }

  template <symbolic::CEquation Eq>
  FORCE_INLINE constexpr double run(const Eq &eq, double r) const {
    return at_point(eq, r, [](const auto &e, auto... v) {
      return e.evaluate(v...);
    });
  }

  FORCE_INLINE constexpr double eval(double r) const {
    return run(detail::equation<Derived>, r);
  }
  FORCE_INLINE constexpr double deriv(double r) const {
    return run(detail::derivative<Derived>, r);
  }

  // Exact dphi/dtheta (jacobian) and d2phi/dr dtheta (derivative_tensor<2>) from
  // the value equation; jacobian() of the stored r-derivative is silently zero.
  static constexpr bool has_param_jacobian() noexcept { return true; }

  FORCE_INLINE void param_grad(double r, std::span<double> out) const {
    fill_free(out, all_param_grad(r));
  }
  FORCE_INLINE void dderiv_dparam(double r, std::span<double> out) const {
    fill_free(out, all_dderiv_dparam(r));
  }

  [[nodiscard]] FORCE_INLINE double deriv2(double r) const {
    return at_point(detail::equation<Derived>, r,
                    [](const auto &e, auto... v) {
                      constexpr std::size_t R = detail::r_slot<Derived>;
                      return e.template derivative_tensor<2>(v...)[R, R];
                    });
  }

private:
  template <class Row>
  FORCE_INLINE static std::array<double, N> by_param(const Row &row) {
    constexpr auto slots =
        detail::symbol_map<Derived,
                           std::remove_cvref_t<decltype(detail::equation<
                                                        Derived>)>>;
    std::array<double, N> g{};
    for (std::size_t k = 0; k < slots.size(); ++k) {
      if (slots[k] < N) {
        g[slots[k]] = row(k);
      }
    }
    return g;
  }

  FORCE_INLINE std::array<double, N> all_param_grad(double r) const {
    return at_point(detail::equation<Derived>, r,
                    [](const auto &e, auto... v) {
                      const auto j = e.jacobian(v...);
                      return by_param([&](std::size_t k) { return j[k]; });
                    });
  }

  FORCE_INLINE std::array<double, N> all_dderiv_dparam(double r) const {
    return at_point(detail::equation<Derived>, r,
                    [](const auto &e, auto... v) {
                      const auto t = e.template derivative_tensor<2>(v...);
                      return by_param([&](std::size_t k) {
                        return t[detail::r_slot<Derived>, k];
                      });
                    });
  }

  FORCE_INLINE void fill_free(std::span<double> out,
                              const std::array<double, N> &all) const {
    std::size_t o = 0;
    for (std::size_t i = 0; i < N; ++i) {
      if (!params[i].fixed) {
        out[o++] = all[i];
      }
    }
  }

public:
};

// Expressions keep the original kernels' operand association (bit-identical).

using namespace std::string_view_literals;

struct LennardJones : AnalyticForm<LennardJones, 2> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"lj"sv, "pair_lj"sv};
  static constexpr std::array param_names{"epsilon"sv, "sigma"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{0.1, 0.0, 1.0}, {2.5, 1.0, 4.0}}};
  static constexpr auto expr = [] {
    const auto sr6 = pow(sym::sigma / sym::r, 6.0);
    return 4.0 * sym::epsilon * (sr6 * sr6 - sr6);
  }();
};

struct Morse : AnalyticForm<Morse, 3> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"morse"sv};
  static constexpr std::array param_names{"De"sv, "a"sv, "re"sv};
  static constexpr std::array<ParamDefault, 3> defaults{
      {{0.1, 0.0, 1.0}, {2.0, 1.0, 5.0}, {2.5, 1.0, 5.0}}};
  static constexpr auto expr = [] {
    const auto e = exp(-sym::a * (sym::r - sym::re));
    return sym::De * (1.0 - e) * (1.0 - e) - sym::De;
  }();
};

struct Buckingham : AnalyticForm<Buckingham, 3> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"buckingham"sv, "buck"sv};
  static constexpr std::array param_names{"A"sv, "rho"sv, "C"sv};
  static constexpr std::array<ParamDefault, 3> defaults{
      {{1.0, -10.0, 10.0}, {1.0, -10.0, 10.0}, {1.0, -10.0, 10.0}}};
  static constexpr auto expr = [] {
    const auto x = (sym::rho * sym::rho) / (sym::r * sym::r);
    return sym::A * exp(-sym::r / sym::rho) - sym::C * x * x * x;
  }();
};

struct Born : AnalyticForm<Born, 5> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"born"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "C"sv, "D"sv, "E"sv};
  static constexpr std::array<ParamDefault, 5> defaults{{{0.1, 0.0, 10.0},
                                                         {2.0, 0.0, 10.0},
                                                         {3.0, 0.0, 10.0},
                                                         {2.0, 0.0, 10.0},
                                                         {2.5, 1.0, 8.0}}};
  static constexpr auto expr = [] {
    const auto r2 = sym::r * sym::r;
    const auto r6 = r2 * r2 * r2;
    return sym::A * exp((sym::C - sym::r) / sym::B) - sym::D / r6 +
           sym::E / (r6 * r2);
  }();
};

struct PowerDecay : AnalyticForm<PowerDecay, 2> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"power_decay"sv, "power"sv};
  static constexpr std::array param_names{"A"sv, "n"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{1.0, 0.1, 10.0}, {2.0, 1.0, 5.0}}};
  static constexpr auto expr = sym::A / pow(sym::r, sym::n);
};

struct ExpDecay : AnalyticForm<ExpDecay, 2> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"exp_decay"sv, "exp"sv};
  static constexpr std::array param_names{"A"sv, "B"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{4.0, 1.0, 20.0}, {1.0, 0.5, 5.0}}};
  static constexpr auto expr = sym::A * exp(-sym::B * sym::r);
};

struct MexpDecay : AnalyticForm<MexpDecay, 3> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"mexp_decay"sv, "mexp"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "r0"sv};
  static constexpr std::array<ParamDefault, 3> defaults{
      {{0.1, 0.0, 10.0}, {0.1, 0.0, 10.0}, {2.0, 0.0, 10.0}}};
  static constexpr auto expr = sym::A * exp(-sym::B * (sym::r - sym::r0));
};

struct Harmonic : AnalyticForm<Harmonic, 2> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"harmonic"sv};
  static constexpr std::array param_names{"k"sv, "r0"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{0.1, 0.0, 10.0}, {2.0, 0.0, 8.0}}};
  static constexpr auto expr = sym::k * (sym::r - sym::r0) * (sym::r - sym::r0);
};

struct Universal : AnalyticForm<Universal, 4> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"universal"sv};
  static constexpr std::array param_names{"E0"sv, "a"sv, "b"sv, "c"sv};
  static constexpr std::array<ParamDefault, 4> defaults{{{1.0, -10.0, 10.0},
                                                         {1.0, 0.0, 20.0},
                                                         {2.0, 0.0, 20.0},
                                                         {0.0, -1.0, 1.0}}};
  static constexpr auto expr =
      sym::E0 * (sym::b / (sym::b - sym::a) * pow(sym::r, sym::a) -
                 sym::a / (sym::b - sym::a) * pow(sym::r, sym::b)) +
      sym::c * sym::r;
};

struct Eopp : AnalyticForm<Eopp, 6> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"eopp"sv};
  static constexpr std::array param_names{"A"sv, "n"sv, "B"sv,
                                          "m"sv, "k"sv, "phi"sv};
  static constexpr std::array<ParamDefault, 6> defaults{{{15.0, 1.0, 10000.0},
                                                         {6.0, 1.0, 20.0},
                                                         {5.0, -100.0, 100.0},
                                                         {3.0, 1.0, 10.0},
                                                         {2.5, 0.0, 6.0},
                                                         {3.0, 0.0, 6.3}}};
  static constexpr auto expr =
      sym::A / pow(sym::r, sym::n) +
      (sym::B / pow(sym::r, sym::m)) * cos(sym::k * sym::r + sym::phi);
};

struct EoppExp : AnalyticForm<EoppExp, 6> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"eopp_exp"sv, "eopp_exp_"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "C"sv,
                                          "m"sv, "k"sv, "phi"sv};
  static constexpr std::array<ParamDefault, 6> defaults{{{15.0, 1.0, 10000.0},
                                                         {1.0, 0.5, 5.0},
                                                         {5.0, -100.0, 100.0},
                                                         {3.0, 1.0, 10.0},
                                                         {2.5, 0.0, 6.0},
                                                         {3.0, 0.0, 6.3}}};
  static constexpr auto expr =
      sym::A * exp(-sym::B * sym::r) +
      (sym::C / pow(sym::r, sym::m)) * cos(sym::k * sym::r + sym::phi);
};

struct Meopp : AnalyticForm<Meopp, 7> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"meopp"sv};
  static constexpr std::array param_names{"A"sv, "n"sv,   "B"sv, "m"sv,
                                          "k"sv, "phi"sv, "r0"sv};
  static constexpr std::array<ParamDefault, 7> defaults{{{15.0, 1.0, 10000.0},
                                                         {6.0, 1.0, 20.0},
                                                         {5.0, -100.0, 100.0},
                                                         {3.0, 1.0, 10.0},
                                                         {2.5, 0.0, 6.0},
                                                         {3.0, 0.0, 6.3},
                                                         {0.5, 0.0, 2.0}}};
  static constexpr auto expr =
      sym::A / pow(sym::r - sym::r0, sym::n) +
      (sym::B / pow(sym::r, sym::m)) * cos(sym::k * sym::r + sym::phi);
};

struct GenLJ : AnalyticForm<GenLJ, 5> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"gen_lj"sv, "genlj"sv};
  static constexpr std::array param_names{"A"sv, "n"sv, "m"sv, "r0"sv, "B"sv};
  static constexpr std::array<ParamDefault, 5> defaults{{{0.1, 0.0, 10.0},
                                                         {12.0, 1.0, 20.0},
                                                         {6.0, 1.0, 20.0},
                                                         {2.5, 1.0, 5.0},
                                                         {0.0, -10.0, 10.0}}};
  static constexpr auto expr = [] {
    const auto x = sym::r / sym::r0;
    return sym::A / (sym::m - sym::n) *
               (sym::m * pow(x, -sym::n) - sym::n * pow(x, -sym::m)) +
           sym::B;
  }();
};

struct DoubleMorse : AnalyticForm<DoubleMorse, 7> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"double_morse"sv, "dbl_morse"sv};
  static constexpr std::array param_names{"D1"sv, "a1"sv, "r1"sv, "D2"sv,
                                          "a2"sv, "r2"sv, "C"sv};
  static constexpr std::array<ParamDefault, 7> defaults{{{0.1, 0.0, 1.0},
                                                         {2.0, 1.0, 5.0},
                                                         {2.5, 1.0, 5.0},
                                                         {0.1, 0.0, 1.0},
                                                         {2.0, 1.0, 5.0},
                                                         {3.0, 1.0, 6.0},
                                                         {0.0, -1.0, 1.0}}};
  static constexpr auto expr = [] {
    const auto e1 = exp(-sym::a1 * (sym::r - sym::r1));
    const auto e2 = exp(-sym::a2 * (sym::r - sym::r2));
    return sym::D1 * ((1.0 - e1) * (1.0 - e1) - 1.0) +
           sym::D2 * ((1.0 - e2) * (1.0 - e2) - 1.0) + sym::C;
  }();
};

struct DoubleExp : AnalyticForm<DoubleExp, 5> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"double_exp"sv, "dbl_exp"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "r1"sv, "C"sv, "r2"sv};
  static constexpr std::array<ParamDefault, 5> defaults{{{0.1, 0.0, 10.0},
                                                         {1.0, 0.0, 10.0},
                                                         {2.5, 1.0, 5.0},
                                                         {1.0, 0.0, 10.0},
                                                         {2.5, 1.0, 5.0}}};
  static constexpr auto expr = [] {
    const auto dr = sym::r - sym::r1;
    return sym::A * exp(-sym::B * dr * dr) + exp(-sym::C * (sym::r - sym::r2));
  }();
};

struct Mishin : AnalyticForm<Mishin, 6> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"mishin"sv};
  static constexpr std::array param_names{"A"sv,  "B"sv, "C"sv,
                                          "r0"sv, "n"sv, "d"sv};
  static constexpr std::array<ParamDefault, 6> defaults{{{0.1, 0.0, 10.0},
                                                         {1.0, -10.0, 10.0},
                                                         {0.0, -1.0, 1.0},
                                                         {1.0, 0.0, 5.0},
                                                         {2.0, 0.0, 10.0},
                                                         {1.0, 0.0, 10.0}}};
  static constexpr auto expr = [] {
    const auto z = sym::r - sym::r0;
    const auto e = exp(-sym::d * z);
    return sym::A * pow(z, sym::n) * e * (1.0 + sym::B * e) + sym::C;
  }();
};

struct SqrtFunc : AnalyticForm<SqrtFunc, 2> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"sqrt"sv};
  static constexpr std::array param_names{"A"sv, "B"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{0.1, 0.0, 10.0}, {2.0, 0.0, 10.0}}};
  static constexpr auto expr = sym::A * sqrt(sym::r / sym::B);
};

struct ConstFunc : AnalyticForm<ConstFunc, 1> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"const"sv};
  static constexpr std::array param_names{"C"sv};
  static constexpr std::array<ParamDefault, 1> defaults{{{1.0, 0.0, 2.0}}};
  static constexpr auto expr = sym::C + 0.0 * sym::r;
};

struct Parabola : AnalyticForm<Parabola, 3> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"parabola"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "C"sv};
  static constexpr std::array<ParamDefault, 3> defaults{
      {{1.0, -10.0, 10.0}, {1.0, -10.0, 10.0}, {1.0, -10.0, 10.0}}};
  static constexpr auto expr =
      sym::A * sym::r * sym::r + sym::B * sym::r + sym::C;
};

struct Poly5 : AnalyticForm<Poly5, 5> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"poly5"sv};
  static constexpr std::array param_names{"a0"sv, "a1"sv, "a2"sv, "a3"sv,
                                          "a4"sv};
  static constexpr std::array<ParamDefault, 5> defaults{{{0.0, -10.0, 10.0},
                                                         {0.0, -10.0, 10.0},
                                                         {0.0, -10.0, 10.0},
                                                         {0.0, -10.0, 10.0},
                                                         {0.0, -10.0, 10.0}}};
  static constexpr auto expr = [] {
    const auto s = sym::r - 1.0;
    const auto s2 = s * s;
    return sym::a0 + 0.5 * sym::a1 * s2 + sym::a2 * s * s2 + sym::a3 * s2 * s2 +
           sym::a4 * s2 * s2 * s;
  }();
};

struct StiwWeb2 : AnalyticForm<StiwWeb2, 6> {
  using Base = AnalyticForm<StiwWeb2, 6>;
  using Base::Base;
  static constexpr std::array names{"stiweb_2"sv, "sw2"sv};
  static constexpr std::array param_names{"A"sv, "B"sv,     "p"sv,
                                          "q"sv, "delta"sv, "rc"sv};
  static constexpr std::array<ParamDefault, 6> defaults{{{1.0, 0.0, 100.0},
                                                         {1.0, 0.0, 100.0},
                                                         {4.0, 1.0, 10.0},
                                                         {0.0, 0.0, 10.0},
                                                         {1.0, 0.0, 10.0},
                                                         {3.0, 1.0, 10.0}}};
  static constexpr auto expr =
      (sym::A * pow(sym::r, -sym::p) - sym::B * pow(sym::r, -sym::q)) *
      exp(sym::delta / (sym::r - sym::rc));

  FORCE_INLINE bool outside(double r) const {
    const double d = r - params[5].value;
    return d >= 0.0 || params[4].value / d < -708.0;
  }
  FORCE_INLINE double eval(double r) const {
    return outside(r) ? 0.0 : Base::eval(r);
  }
  FORCE_INLINE double deriv(double r) const {
    return outside(r) ? 0.0 : Base::deriv(r);
  }
};

struct StiwWeb3 : AnalyticForm<StiwWeb3, 2> {
  using Base = AnalyticForm<StiwWeb3, 2>;
  using Base::Base;
  static constexpr std::array names{"stiweb_3"sv, "sw3"sv};
  static constexpr std::array param_names{"gamma"sv, "a"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{1.0, 0.0, 10.0}, {3.0, 1.0, 10.0}}};
  static constexpr auto expr = exp(sym::gamma / (sym::r - sym::a));

  FORCE_INLINE bool outside(double r) const {
    const double d = r - params[1].value;
    return d >= 0.0 || params[0].value / d < -708.0;
  }
  FORCE_INLINE double eval(double r) const {
    return outside(r) ? 0.0 : Base::eval(r);
  }
  FORCE_INLINE double deriv(double r) const {
    return outside(r) ? 0.0 : Base::deriv(r);
  }
};

struct TersoffPot : AnalyticForm<TersoffPot, 11> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"tersoff"sv, "tersoff_pot"sv};
  static constexpr std::array param_names{"A"sv,    "B"sv, "lambda"sv, "mu"sv,
                                          "beta"sv, "n"sv, "c"sv,      "d"sv,
                                          "h"sv,    "R"sv, "S"sv};
  static constexpr std::array<ParamDefault, 11> defaults{{{1000.0, 0.0, 1e5},
                                                          {100.0, 0.0, 1e5},
                                                          {3.0, 0.0, 10.0},
                                                          {1.5, 0.0, 10.0},
                                                          {1e-6, 0.0, 1.0},
                                                          {1.0, 0.0, 10.0},
                                                          {1e5, 0.0, 1e6},
                                                          {20.0, 0.0, 1e3},
                                                          {-0.5, -1.0, 1.0},
                                                          {2.7, 1.0, 5.0},
                                                          {3.0, 1.0, 6.0}}};
  static constexpr auto expr =
      detail::cosine_cutoff(sym::r, sym::R, sym::S) *
      (sym::A * exp(-sym::lambda * sym::r) - sym::B * exp(-sym::mu * sym::r));
};

struct TersoffMix : AnalyticForm<TersoffMix, 2> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"tersoff_mix"sv};
  static constexpr std::array param_names{"chi"sv, "omega"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{1.0, 0.0, 10.0}, {1.0, 0.0, 10.0}}};
  static constexpr auto expr = sym::chi * exp(-sym::omega * sym::r);
};

struct TersoffModPot : AnalyticForm<TersoffModPot, 16> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"tersoff_mod"sv, "tmod"sv};
  static constexpr std::array param_names{
      "A"sv, "B"sv, "lambda"sv, "mu"sv, "beta"sv, "n"sv,  "c"sv,  "d"sv,
      "h"sv, "R"sv, "S"sv,      "c1"sv, "c2"sv,   "c3"sv, "c4"sv, "c5"sv};
  static constexpr std::array<ParamDefault, 16> defaults{{{1000.0, 0.0, 1e5},
                                                          {100.0, 0.0, 1e5},
                                                          {3.0, 0.0, 10.0},
                                                          {1.5, 0.0, 10.0},
                                                          {1e-6, 0.0, 1.0},
                                                          {1.0, 0.0, 10.0},
                                                          {1e5, 0.0, 1e6},
                                                          {20.0, 0.0, 1e3},
                                                          {-0.5, -1.0, 1.0},
                                                          {2.7, 1.0, 5.0},
                                                          {3.0, 1.0, 6.0},
                                                          {0.0, -10.0, 10.0},
                                                          {0.0, -10.0, 10.0},
                                                          {0.0, -10.0, 10.0},
                                                          {0.0, -10.0, 10.0},
                                                          {0.0, -10.0, 10.0}}};
  static constexpr auto expr = [] {
    const auto r2 = sym::r * sym::r;
    const auto r3 = r2 * sym::r;
    const auto r4 = r3 * sym::r;
    const auto r5 = r4 * sym::r;
    return detail::cosine_cutoff(sym::r, sym::R, sym::S) *
           (sym::A * exp(-sym::lambda * sym::r) -
            sym::B * exp(-sym::mu * sym::r)) *
           (1.0 + sym::c1 * sym::r + sym::c2 * r2 + sym::c3 * r3 +
            sym::c4 * r4 + sym::c5 * r5);
  }();
};

struct Kawamura : AnalyticForm<Kawamura, 9> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"kawamura"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "C"sv, "D"sv, "E"sv,
                                          "F"sv, "G"sv, "H"sv, "I"sv};
  static constexpr std::array<ParamDefault, 9> defaults{{{1.0, -5.0, 5.0},
                                                         {1.0, -5.0, 5.0},
                                                         {1.0, 0.0, 100.0},
                                                         {1.0, 0.0, 5.0},
                                                         {1.0, 0.0, 5.0},
                                                         {0.1, 0.01, 2.0},
                                                         {0.1, 0.01, 2.0},
                                                         {1.0, 0.0, 100.0},
                                                         {1.0, 0.0, 100.0}}};
  static constexpr auto expr = [] {
    const auto s = sym::F + sym::G;
    const auto t = sym::D + sym::E;
    return sym::A * sym::B / sym::r + sym::C * s * exp((t - sym::r) / s) -
           sym::H * sym::I / pow(sym::r, 6.0);
  }();
};

struct KawamuraMix : AnalyticForm<KawamuraMix, 12> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"kawamura_mix"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "C"sv, "D"sv,
                                          "E"sv, "F"sv, "G"sv, "H"sv,
                                          "I"sv, "J"sv, "K"sv, "L"sv};
  static constexpr std::array<ParamDefault, 12> defaults{{{1.0, -5.0, 5.0},
                                                          {1.0, -5.0, 5.0},
                                                          {1.0, 0.0, 100.0},
                                                          {1.0, 0.0, 5.0},
                                                          {1.0, 0.0, 5.0},
                                                          {0.1, 0.01, 2.0},
                                                          {0.1, 0.01, 2.0},
                                                          {1.0, 0.0, 100.0},
                                                          {1.0, 0.0, 100.0},
                                                          {0.0, -10.0, 10.0},
                                                          {1.0, 0.0, 10.0},
                                                          {2.5, 1.0, 5.0}}};
  static constexpr auto expr = [] {
    const auto s = sym::F + sym::G;
    const auto t = sym::D + sym::E;
    const auto w = sym::r - sym::L;
    return sym::A * sym::B / sym::r + sym::C * s * exp((t - sym::r) / s) -
           sym::H * sym::I / pow(sym::r, 6.0) +
           sym::C * sym::J * (exp(-2.0 * sym::K * w) - 2.0 * exp(-sym::K * w));
  }();
};

struct Softshell : AnalyticForm<Softshell, 2> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"softshell"sv, "soft"sv};
  static constexpr std::array param_names{"A"sv, "n"sv};
  static constexpr std::array<ParamDefault, 2> defaults{
      {{1.0, 0.1, 10.0}, {2.0, 1.0, 5.0}}};
  static constexpr auto expr = pow(sym::A / sym::r, sym::n);
};

struct ExpPlus : AnalyticForm<ExpPlus, 3> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"exp_plus"sv, "expplus"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "C"sv};
  static constexpr std::array<ParamDefault, 3> defaults{
      {{4.0, 1.0, 20.0}, {1.0, 0.5, 5.0}, {0.0, -1.0, 1.0}}};
  static constexpr auto expr = sym::A * exp(-sym::B * sym::r) + sym::C;
};

struct Strmm : AnalyticForm<Strmm, 5> {
  using AnalyticForm::AnalyticForm;
  static constexpr std::array names{"strmm"sv};
  static constexpr std::array param_names{"A"sv, "B"sv, "C"sv, "D"sv, "E"sv};
  static constexpr std::array<ParamDefault, 5> defaults{{{0.1, 0.0, 10.0},
                                                         {1.0, 0.0, 10.0},
                                                         {0.1, 0.0, 10.0},
                                                         {1.0, 0.0, 10.0},
                                                         {2.5, 1.0, 5.0}}};
  static constexpr auto expr = [] {
    const auto s = sym::r - sym::E;
    return 2.0 * sym::A * exp(-sym::B / 2.0 * s) -
           sym::C * (1.0 + sym::D * s) * exp(-sym::D * s);
  }();
};

namespace detail {
template <class A, std::size_t Na, class B, std::size_t Nb>
consteval auto concat(const std::array<A, Na> &a, const std::array<B, Nb> &b) {
  std::array<A, Na + Nb> out{};
  std::ranges::copy(a, out.begin());
  std::ranges::copy(b, out.begin() + Na);
  return out;
}
} // namespace detail

template <typename Base, ddx::impl::FixedString Name,
          ddx::impl::FixedString Alias = "">
struct SmoothCutoff
    : AnalyticForm<SmoothCutoff<Base, Name, Alias>, Base::num_params + 1> {
  using Self = SmoothCutoff<Base, Name, Alias>;
  using AnalyticForm<Self, Base::num_params + 1>::AnalyticForm;

  static constexpr std::array names{Name.view(), Alias.view()};
  static constexpr auto param_names =
      detail::concat(Base::param_names, std::array{"h"sv});
  static constexpr auto defaults = detail::concat(
      Base::defaults, std::array<ParamDefault, 1>{ParamDefault{1.0, 0.5, 2.0}});
  static constexpr auto expr =
      Base::expr * detail::switching_factor(sym::r, sym::rmax, sym::h);
};

using LennardJonesSC = SmoothCutoff<LennardJones, "lj_sc", "pair_lj_sc">;
using MorseSC = SmoothCutoff<Morse, "morse_sc">;
using ExpDecaySC = SmoothCutoff<ExpDecay, "exp_decay_sc", "exp_sc">;
using EoppSC = SmoothCutoff<Eopp, "eopp_sc">;

using AnalyticForms = boost::mp11::mp_list<
    LennardJones, Morse, Buckingham, Born, PowerDecay, ExpDecay, MexpDecay,
    Harmonic, Universal, Eopp, EoppExp, Meopp, GenLJ, DoubleMorse, DoubleExp,
    Mishin, SqrtFunc, ConstFunc, Parabola, Poly5, StiwWeb2, StiwWeb3,
    TersoffPot, TersoffMix, TersoffModPot, Kawamura, KawamuraMix, Softshell,
    ExpPlus, Strmm, LennardJonesSC, MorseSC, ExpDecaySC, EoppSC>;

} // namespace forcesmith
