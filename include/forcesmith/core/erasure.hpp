#pragma once

#include <Eigen/Core>

#include <boost/mpl/vector.hpp>
#include <boost/type_erasure/any.hpp>
#include <boost/type_erasure/any_cast.hpp>
#include <boost/type_erasure/builtin.hpp>
#include <boost/type_erasure/free.hpp>
#include <boost/type_erasure/member.hpp>
#include <boost/type_erasure/relaxed.hpp>

#include <cstddef>
#include <utility>

namespace forcesmith::te {

namespace bte = boost::type_erasure;
using bte::_self;

BOOST_TYPE_ERASURE_MEMBER(eval)
BOOST_TYPE_ERASURE_MEMBER(deriv)
BOOST_TYPE_ERASURE_MEMBER(eval_and_deriv)
BOOST_TYPE_ERASURE_MEMBER(prepare_site)
BOOST_TYPE_ERASURE_MEMBER(eval_at)
BOOST_TYPE_ERASURE_MEMBER(deriv_at)
BOOST_TYPE_ERASURE_MEMBER(eval_and_deriv_at)
BOOST_TYPE_ERASURE_MEMBER(span)

BOOST_TYPE_ERASURE_MEMBER(param_count)
BOOST_TYPE_ERASURE_MEMBER(gather_params)
BOOST_TYPE_ERASURE_MEMBER(scatter_params)
BOOST_TYPE_ERASURE_MEMBER(gather_bounds)
BOOST_TYPE_ERASURE_MEMBER(set_param)
BOOST_TYPE_ERASURE_MEMBER(set_fixed)
BOOST_TYPE_ERASURE_MEMBER(set_bounds)

BOOST_TYPE_ERASURE_MEMBER(minimize)
BOOST_TYPE_ERASURE_MEMBER(honors_bounds)
BOOST_TYPE_ERASURE_MEMBER(one_shot)

BOOST_TYPE_ERASURE_MEMBER(energy)
BOOST_TYPE_ERASURE_MEMBER(grad)
BOOST_TYPE_ERASURE_MEMBER(constant_grad)
BOOST_TYPE_ERASURE_MEMBER(has_param_jacobian)
BOOST_TYPE_ERASURE_MEMBER(param_grad)
BOOST_TYPE_ERASURE_MEMBER(dgrad_dparam)
BOOST_TYPE_ERASURE_MEMBER(all_values)
BOOST_TYPE_ERASURE_MEMBER(set_all_values)
BOOST_TYPE_ERASURE_MEMBER(remapped)

BOOST_TYPE_ERASURE_MEMBER(eval_forces)
BOOST_TYPE_ERASURE_MEMBER(max_cutoff)
BOOST_TYPE_ERASURE_MEMBER(prepare)
BOOST_TYPE_ERASURE_MEMBER(has_cache)
BOOST_TYPE_ERASURE_MEMBER(has_standardization)
BOOST_TYPE_ERASURE_MEMBER(eval_cached)
BOOST_TYPE_ERASURE_MEMBER(eval_cached_jacobian)
BOOST_TYPE_ERASURE_MEMBER(head_param_counts)

BOOST_TYPE_ERASURE_FREE(has_ntypes_of, ntypes_of, 1)
BOOST_TYPE_ERASURE_FREE(has_curvature_count, curvature_count, 1)
BOOST_TYPE_ERASURE_FREE(has_write_curvature, write_curvature, 4)
BOOST_TYPE_ERASURE_FREE(has_model_smoothness_count, model_smoothness_count, 1)
BOOST_TYPE_ERASURE_FREE(has_model_write_smoothness, model_write_smoothness, 4)
BOOST_TYPE_ERASURE_FREE(has_build_json_from_head, build_json_from_head, 1)

struct ValueBuiltins : boost::mpl::vector<bte::copy_constructible<>,
                                          bte::constructible<_self(_self &&)>,
                                          bte::typeid_<>, bte::relaxed> {};

struct MoveOnlyBuiltins
    : boost::mpl::vector<bte::constructible<_self(_self &&)>,
                         bte::destructible<>, bte::typeid_<>, bte::relaxed> {};

struct CFittableTE
    : boost::mpl::vector<
          has_param_count<std::size_t() const>,
          has_gather_params<void(Eigen::VectorXd &, std::size_t) const>,
          has_scatter_params<void(const Eigen::VectorXd &, std::size_t)>,
          has_gather_bounds<void(Eigen::VectorXd &, Eigen::VectorXd &,
                                 std::size_t) const>> {};

} // namespace forcesmith::te
