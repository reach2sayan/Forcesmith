// Exercises the shared foundations every later layer is built on: the family
// list, the Voigt order, the describe-driven JSON and archive defaults, the
// table-field walk, strong indices, the boost::parser bridge and the shared
// finite-difference kernels.

#include "forcesmith/core/families.hpp"
#include "forcesmith/core/fields.hpp"
#include "forcesmith/core/json.hpp"
#include "forcesmith/core/serialization.hpp"
#include "forcesmith/core/site_id.hpp"
#include "forcesmith/core/strong.hpp"
#include "forcesmith/core/voigt.hpp"
#include "forcesmith/io/grammar.hpp"
#include "forcesmith/optimization/fd_jacobian.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/describe/class.hpp>
#include <boost/leaf/handle_errors.hpp>
#include <boost/mp11/algorithm.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

using namespace forcesmith;

namespace {

struct Sample {
  double a = 0.0;
  int b = 0;
  std::string c;
};
BOOST_DESCRIBE_STRUCT(Sample, (), (a, b, c))

struct SampleArchived : Serializable<SampleArchived> {
  double a = 0.0;
  int b = 0;
};
BOOST_DESCRIBE_STRUCT(SampleArchived, (), (a, b))

} // namespace

TEST(Foundations, FamilyListCoversEveryNamedFamily) {
  EXPECT_EQ(boost::mp11::mp_size<ModelFamilies>::value, 9u);
  EXPECT_EQ(boost::mp11::mp_size<MLFamilies>::value, 3u);

  std::vector<std::string_view> names;
  boost::mp11::mp_for_each<
      boost::mp11::mp_transform<boost::mp11::mp_identity, ModelFamilies>>(
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        names.push_back(family_name<T>);
        EXPECT_FALSE(family_name<T>.empty());
      });
  EXPECT_EQ(names.front(), "pair");
  EXPECT_EQ(names.back(), "lmbtr");

  static_assert(is_ml_family<ACSF>);
  static_assert(!is_ml_family<EAMForceCalculator>);
  static_assert(is_model_family<StiwebForceCalculator>);
}

TEST(Foundations, VoigtOrderMatchesResidualLayout) {
  ASSERT_EQ(kVoigt6.size(), 6u);
  const std::array<std::pair<int, int>, 6> expected{
      std::pair{0, 0}, std::pair{1, 1}, std::pair{2, 2},
      std::pair{0, 1}, std::pair{0, 2}, std::pair{1, 2}};
  EXPECT_EQ(kVoigt6, expected);
}

TEST(Foundations, DescribedAggregateRoundTripsThroughJson) {
  const Sample s{1.5, 7, "x"};
  const nlohmann::json j = s;
  EXPECT_EQ(j.at("a").get<double>(), 1.5);
  EXPECT_EQ(j.at("b").get<int>(), 7);
  EXPECT_EQ(j.at("c").get<std::string>(), "x");

  const auto back = j.get<Sample>();
  EXPECT_EQ(back.a, s.a);
  EXPECT_EQ(back.b, s.b);
  EXPECT_EQ(back.c, s.c);
}

TEST(Foundations, MissingKeyBecomesOneParseError) {
  // The error object must be created inside the handling scope for LEAF to
  // capture the ParseError payload.
  const std::string msg = boost::leaf::try_handle_all(
      []() -> boost::leaf::result<std::string> {
        BOOST_LEAF_CHECK(io::catch_json([]() -> boost::leaf::result<Sample> {
          return nlohmann::json::parse(R"({"a": 1.0, "b": 2})").get<Sample>();
        }));
        return std::string{};
      },
      [](const io::ParseError &e) { return e.message; },
      [] { return std::string("other"); });
  EXPECT_NE(msg.find('c'), std::string::npos) << msg;
}

TEST(Foundations, ParamAndTensorSerializers) {
  Param p{2.0, -1.0, 3.0};
  const nlohmann::json bare = p;
  EXPECT_TRUE(bare.is_number());
  EXPECT_EQ(bare.get<double>(), 2.0);

  const auto from_obj =
      nlohmann::json::parse(
          R"({"value": 4.0, "min": 0.0, "max": 5.0, "fixed": true})")
          .get<Param>();
  EXPECT_EQ(from_obj.value, 4.0);
  EXPECT_EQ(from_obj.min, 0.0);
  EXPECT_EQ(from_obj.max, 5.0);
  EXPECT_TRUE(from_obj.fixed);
  EXPECT_EQ(nlohmann::json::parse("6.5").get<Param>().value, 6.5);

  const Vec3 v{1.0, 2.0, 3.0};
  EXPECT_EQ(nlohmann::json(v).get<Vec3>(), v);
  Mat3 m;
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  EXPECT_EQ(nlohmann::json(m).get<Mat3>(), m);
}

TEST(Foundations, DescribedAggregateRoundTripsThroughArchive) {
  SampleArchived out;
  out.a = 3.25;
  out.b = -4;
  std::stringstream ss;
  {
    boost::archive::text_oarchive oa(ss);
    oa << out;
  }
  SampleArchived in;
  {
    boost::archive::text_iarchive ia(ss);
    ia >> in;
  }
  EXPECT_EQ(in.a, out.a);
  EXPECT_EQ(in.b, out.b);
}

TEST(Foundations, TableFieldsListEveryRadialTable) {
  EXPECT_EQ(table_count<PairForceCalculator>, 1u);
  EXPECT_EQ(table_count<EAMForceCalculator>, 3u);
  EXPECT_EQ(table_count<ADPForceCalculator>, 5u);
  EXPECT_EQ(table_count<AngularForceCalculator>, 3u);
  EXPECT_EQ(table_count<TersoffForceCalculator>, 0u);

  EAMForceCalculator eam;
  std::vector<std::string_view> keys;
  for_each_table(eam,
                 [&](auto &, std::string_view name) { keys.push_back(name); });
  EXPECT_EQ(keys,
            (std::vector<std::string_view>{"pair", "density", "embedding"}));
}

TEST(Foundations, StrongIndices) {
  EXPECT_FALSE(SiteId{}.cacheable());
  EXPECT_TRUE(SiteId{3}.cacheable());
  EXPECT_EQ(SiteId{3}.index(), 3);
  EXPECT_EQ(SiteId{3}, SiteId{3});
  EXPECT_NE(SiteId{3}, SiteId{4});
  static_assert(sizeof(SiteId) == sizeof(std::int32_t));

  const TypeIndex t{2};
  const std::size_t as_index = t; // implicit, unlike SiteId
  EXPECT_EQ(as_index, 2u);
  static_assert(!std::is_convertible_v<SiteId, std::int32_t>);
}

TEST(Foundations, ParserBridgeParsesAndReports) {
  namespace bp = boost::parser;
  // bp::char_("...") is a SET of code points, not a regex class: spell ranges
  // out with the two-argument form.
  const auto ident = +(bp::char_('a', 'z') | bp::char_('A', 'Z') |
                       bp::char_('0', '9') | bp::char_('_'));
  const auto item = (bp::uint_ >> '*' | bp::attr(1u)) >> ident;
  const auto spec = item % ',';

  const auto ok = io::parse_or_error(spec, "3*lj,morse", "--functions");
  ASSERT_TRUE(ok);
  ASSERT_EQ(ok->size(), 2u);
  EXPECT_EQ(std::get<0>((*ok)[0]), 3u);
  EXPECT_EQ(std::get<0>((*ok)[1]), 1u);

  const auto err = boost::leaf::try_handle_all(
      [&]() -> boost::leaf::result<io::ParseError> {
        BOOST_LEAF_CHECK(io::parse_or_error(spec, "3*lj,,", "--functions"));
        return io::ParseError{};
      },
      [](const io::ParseError &e) { return e; },
      [] { return io::ParseError{"other"}; });
  EXPECT_NE(err.message.find("--functions"), std::string::npos) << err.message;
  EXPECT_EQ(err.line, 1u);
  EXPECT_GT(err.column, 1u);
}

TEST(Foundations, FiniteDifferenceKernels) {
  // f(x) = [x0^2, x0 * x1]
  const auto f = [](const Eigen::VectorXd &x) {
    Eigen::VectorXd out(2);
    out << x[0] * x[0], x[0] * x[1];
    return out;
  };
  Eigen::VectorXd x(2);
  x << 2.0, 3.0;

  Eigen::MatrixXd J(2, 2);
  opt::fd_jacobian(f, x, J);
  EXPECT_NEAR(J(0, 0), 4.0, 1e-6);
  EXPECT_NEAR(J(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(J(1, 0), 3.0, 1e-6);
  EXPECT_NEAR(J(1, 1), 2.0, 1e-6);

  Eigen::VectorXd g;
  opt::fd_gradient(f, x, g);
  const Eigen::VectorXd expected = J.transpose() * f(x);
  EXPECT_NEAR(g[0], expected[0], 1e-6);
  EXPECT_NEAR(g[1], expected[1], 1e-6);

  // The buffer-reusing shape must agree with the by-value one.
  const auto f_into = [&](const Eigen::VectorXd &xv, Eigen::VectorXd &out) {
    out = f(xv);
  };
  Eigen::MatrixXd J2(2, 2);
  opt::fd_jacobian(f_into, x, J2);
  EXPECT_EQ(J2, J);
}
