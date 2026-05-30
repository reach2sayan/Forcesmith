#include "potfit/core/elements.hpp"

#include <boost/leaf/error.hpp>
#include <boost/multi_index/hashed_index.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index_container.hpp>

namespace potfit::elements {

namespace {

struct BySymbol {};
struct ByZ {};

using ElementTable = boost::multi_index_container<
    Element,
    boost::multi_index::indexed_by<
        boost::multi_index::hashed_unique<
            boost::multi_index::tag<BySymbol>,
            boost::multi_index::member<Element, const std::string_view,
                                       &Element::symbol>>,
        boost::multi_index::hashed_unique<
            boost::multi_index::tag<ByZ>,
            boost::multi_index::member<Element, const int, &Element::Z>>>>;

const ElementTable &table() {
  static const ElementTable t{{
      {"Ac", 89, 227.028},  {"Ag", 47, 107.868},  {"Al", 13, 26.982},
      {"Am", 95, 243.061},  {"Ar", 18, 39.948},   {"As", 33, 74.922},
      {"At", 85, 209.987},  {"Au", 79, 196.967},  {"B", 5, 10.811},
      {"Ba", 56, 137.327},  {"Be", 4, 9.012},     {"Bh", 107, 270.133},
      {"Bi", 83, 208.980},  {"Bk", 97, 247.070},  {"Br", 35, 79.904},
      {"C", 6, 12.011},     {"Ca", 20, 40.078},   {"Cd", 48, 112.411},
      {"Ce", 58, 140.116},  {"Cf", 98, 251.080},  {"Cl", 17, 35.453},
      {"Cm", 96, 247.070},  {"Cn", 112, 285.177}, {"Co", 27, 58.933},
      {"Cr", 24, 51.996},   {"Cs", 55, 132.905},  {"Cu", 29, 63.546},
      {"Db", 105, 268.126}, {"Ds", 110, 281.165}, {"Dy", 66, 162.500},
      {"Er", 68, 167.259},  {"Es", 99, 252.083},  {"Eu", 63, 151.964},
      {"F", 9, 18.998},     {"Fe", 26, 55.845},   {"Fl", 114, 289.190},
      {"Fm", 100, 257.095}, {"Fr", 87, 223.020},  {"Ga", 31, 69.723},
      {"Gd", 64, 157.250},  {"Ge", 32, 72.630},   {"H", 1, 1.008},
      {"He", 2, 4.003},     {"Hf", 72, 178.490},  {"Hg", 80, 200.592},
      {"Ho", 67, 164.930},  {"Hs", 108, 277.154}, {"I", 53, 126.904},
      {"In", 49, 114.818},  {"Ir", 77, 192.217},  {"K", 19, 39.098},
      {"Kr", 36, 83.798},   {"La", 57, 138.905},  {"Li", 3, 6.941},
      {"Lr", 103, 262.110}, {"Lu", 71, 174.967},  {"Lv", 116, 293.204},
      {"Mc", 115, 290.196}, {"Md", 101, 258.098}, {"Mg", 12, 24.305},
      {"Mn", 25, 54.938},   {"Mo", 42, 95.960},   {"Mt", 109, 278.156},
      {"N", 7, 14.007},     {"Na", 11, 22.990},   {"Nb", 41, 92.906},
      {"Nd", 60, 144.242},  {"Ne", 10, 20.180},   {"Nh", 113, 286.182},
      {"Ni", 28, 58.693},   {"No", 102, 259.101}, {"Np", 93, 237.048},
      {"O", 8, 15.999},     {"Og", 118, 294.214}, {"Os", 76, 190.230},
      {"P", 15, 30.974},    {"Pa", 91, 231.036},  {"Pb", 82, 207.200},
      {"Pd", 46, 106.420},  {"Pm", 61, 144.913},  {"Po", 84, 208.982},
      {"Pr", 59, 140.908},  {"Pt", 78, 195.084},  {"Pu", 94, 244.064},
      {"Ra", 88, 226.025},  {"Rb", 37, 85.468},   {"Re", 75, 186.207},
      {"Rf", 104, 267.122}, {"Rg", 111, 282.169}, {"Rh", 45, 102.906},
      {"Rn", 86, 222.018},  {"Ru", 44, 101.070},  {"S", 16, 32.065},
      {"Sb", 51, 121.760},  {"Sc", 21, 44.956},   {"Se", 34, 78.960},
      {"Sg", 106, 271.134}, {"Si", 14, 28.086},   {"Sm", 62, 150.360},
      {"Sn", 50, 118.710},  {"Sr", 38, 87.620},   {"Ta", 73, 180.948},
      {"Tb", 65, 158.925},  {"Tc", 43, 97.907},   {"Te", 52, 127.600},
      {"Th", 90, 232.038},  {"Ti", 22, 47.867},   {"Tl", 81, 204.383},
      {"Tm", 69, 168.934},  {"Ts", 117, 294.211}, {"U", 92, 238.029},
      {"V", 23, 50.942},    {"W", 74, 183.840},   {"Xe", 54, 131.293},
      {"Y", 39, 88.906},    {"Yb", 70, 173.054},  {"Zn", 30, 65.380},
      {"Zr", 40, 91.224},
  }};
  return t;
}

} // namespace

boost::optional<const Element &>
find_by_symbol(std::string_view symbol) noexcept {
  const auto &idx = table().get<BySymbol>();
  const auto it = idx.find(symbol);
  if (it == idx.end())
    return boost::none;
  return *it;
}

boost::optional<const Element &> find_by_Z(int Z) noexcept {
  const auto &idx = table().get<ByZ>();
  const auto it = idx.find(Z);
  if (it == idx.end())
    return boost::none;
  return *it;
}

boost::leaf::result<Element> lookup(std::string_view symbol) {
  if (const auto e = find_by_symbol(symbol))
    return *e;
  return boost::leaf::new_error(std::string("unknown element symbol: ") +
                                std::string(symbol));
}

boost::leaf::result<int> atomic_number(std::string_view symbol) {
  BOOST_LEAF_AUTO(e, lookup(symbol));
  return e.Z;
}

boost::leaf::result<double> atomic_mass(std::string_view symbol) {
  BOOST_LEAF_AUTO(e, lookup(symbol));
  return e.mass_amu;
}

} // namespace potfit::elements
