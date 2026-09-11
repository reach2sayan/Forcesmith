#pragma once

#include "forcesmith/force/adp_force.hpp"
#include "forcesmith/force/angular_force.hpp"
#include "forcesmith/force/eam_force.hpp"
#include "forcesmith/force/pair_force.hpp"
#include "forcesmith/force/stiweb_force.hpp"
#include "forcesmith/force/tersoff_force.hpp"
#include "forcesmith/potentials/acsf.hpp"
#include "forcesmith/potentials/lmbtr.hpp"
#include "forcesmith/potentials/soap.hpp"

#include <boost/mp11/algorithm.hpp>
#include <boost/mp11/list.hpp>

#include <string_view>

namespace forcesmith {

using MLFamilies = boost::mp11::mp_list<ACSF, SoapModel, LMBTR>;

using ModelFamilies =
    boost::mp11::mp_list<PairForceCalculator, EAMForceCalculator,
                         ADPForceCalculator, AngularForceCalculator,
                         TersoffForceCalculator, StiwebForceCalculator, ACSF,
                         SoapModel, LMBTR>;

template <class T> inline constexpr std::string_view family_name = {};

template <>
inline constexpr std::string_view family_name<PairForceCalculator> = "pair";
template <>
inline constexpr std::string_view family_name<EAMForceCalculator> = "eam";
template <>
inline constexpr std::string_view family_name<ADPForceCalculator> = "adp";
template <>
inline constexpr std::string_view family_name<AngularForceCalculator> =
    "angular";
template <>
inline constexpr std::string_view family_name<TersoffForceCalculator> =
    "tersoff";
template <>
inline constexpr std::string_view family_name<StiwebForceCalculator> = "stiweb";
template <> inline constexpr std::string_view family_name<ACSF> = "acsf";
template <> inline constexpr std::string_view family_name<SoapModel> = "soap";
template <> inline constexpr std::string_view family_name<LMBTR> = "lmbtr";

template <class T>
inline constexpr bool is_ml_family =
    boost::mp11::mp_contains<MLFamilies, T>::value;

template <class T>
inline constexpr bool is_model_family =
    boost::mp11::mp_contains<ModelFamilies, T>::value;

template <class T>
concept CMLFamily = is_ml_family<T>;

template <class T>
concept CModelFamily = is_model_family<T>;

} // namespace forcesmith
