#include "potfit/core/boundary_conditions.hpp"

#include <gtest/gtest.h>

using namespace potfit;

static constexpr double EPS = 1e-10;

// ── PeriodicBC cubic box ──────────────────────────────────────────────────────

TEST(PeriodicBC, Volume) {
    PeriodicBC bc(10.0 * Mat3::Identity());
    EXPECT_NEAR(bc.volume(), 1000.0, EPS);
}

TEST(PeriodicBC, WrapInsideStaysTheSame) {
    PeriodicBC bc(10.0 * Mat3::Identity());
    Vec3 r(3.0, 4.0, 5.0);
    Vec3 w = bc.wrap(r);
    EXPECT_NEAR(w[0], 3.0, EPS);
    EXPECT_NEAR(w[1], 4.0, EPS);
    EXPECT_NEAR(w[2], 5.0, EPS);
}

TEST(PeriodicBC, WrapPositiveOverflow) {
    PeriodicBC bc(10.0 * Mat3::Identity());
    Vec3 w = bc.wrap(Vec3(12.0, 0.0, 0.0));
    EXPECT_NEAR(w[0], 2.0, EPS);
}

TEST(PeriodicBC, WrapNegative) {
    PeriodicBC bc(10.0 * Mat3::Identity());
    Vec3 w = bc.wrap(Vec3(-1.0, 0.0, 0.0));
    EXPECT_NEAR(w[0], 9.0, EPS);
}

TEST(PeriodicBC, WrapIsIdempotent) {
    PeriodicBC bc(10.0 * Mat3::Identity());
    Vec3 r(12.5, -3.1, 22.7);
    Vec3 w1 = bc.wrap(r);
    Vec3 w2 = bc.wrap(w1);
    EXPECT_NEAR((w1 - w2).norm(), 0.0, EPS);
}

TEST(PeriodicBC, MinImageShortDistance) {
    PeriodicBC bc(10.0 * Mat3::Identity());
    Vec3 mi = bc.min_image(Vec3(3.0, 0.0, 0.0));
    EXPECT_NEAR(mi[0], 3.0, EPS);
}

TEST(PeriodicBC, MinImageFarWrapsToShorter) {
    // d = 8 > 10/2, so min image = 8 - 10 = -2
    PeriodicBC bc(10.0 * Mat3::Identity());
    Vec3 mi = bc.min_image(Vec3(8.0, 0.0, 0.0));
    EXPECT_NEAR(mi[0], -2.0, EPS);
}

TEST(PeriodicBC, MinImageAcrossCorner) {
    PeriodicBC bc(10.0 * Mat3::Identity());
    // Atom at 9.5 should appear at -0.5 from origin
    Vec3 mi = bc.min_image(Vec3(9.5, 0.0, 0.0));
    EXPECT_NEAR(mi[0], -0.5, EPS);
}

// ── PeriodicBC triclinic box ──────────────────────────────────────────────────

TEST(PeriodicBC, TriclinicVolumePositive) {
    Mat3 box;
    box << 10.0, 5.0, 0.0,
            0.0, 8.66, 0.0,
            0.0, 0.0, 10.0;
    PeriodicBC bc(box);
    EXPECT_GT(bc.volume(), 0.0);
}

TEST(PeriodicBC, TriclinicWrapIdempotent) {
    Mat3 box;
    box << 10.0, 5.0, 0.0,
            0.0, 8.66, 0.0,
            0.0, 0.0, 10.0;
    PeriodicBC bc(box);
    Vec3 r(11.0, 3.0, -1.0);
    Vec3 w1 = bc.wrap(r);
    Vec3 w2 = bc.wrap(w1);
    EXPECT_NEAR((w1 - w2).norm(), 0.0, EPS);
}

// ── PeriodicBC set_box ────────────────────────────────────────────────────────

TEST(PeriodicBC, SetBox) {
    PeriodicBC bc(5.0 * Mat3::Identity());
    bc.set_box(10.0 * Mat3::Identity());
    EXPECT_NEAR(bc.volume(), 1000.0, EPS);
    Vec3 w = bc.wrap(Vec3(12.0, 0.0, 0.0));
    EXPECT_NEAR(w[0], 2.0, EPS);
}

// ── InfiniteBC passthrough ────────────────────────────────────────────────────

TEST(InfiniteBC, WrapIsIdentity) {
    InfiniteBC bc(500.0);
    Vec3 r(100.0, -50.0, 300.0);
    EXPECT_NEAR((bc.wrap(r) - r).norm(), 0.0, EPS);
}

TEST(InfiniteBC, MinImageIsIdentity) {
    InfiniteBC bc;
    Vec3 d(100.0, 0.0, 0.0);
    EXPECT_NEAR((bc.min_image(d) - d).norm(), 0.0, EPS);
}

TEST(InfiniteBC, Volume) {
    InfiniteBC bc(500.0);
    EXPECT_NEAR(bc.volume(), 500.0, EPS);
}

// ── BoundaryConditions variant dispatch ──────────────────────────────────────

TEST(BoundaryConditions, VariantPeriodicWrap) {
    BoundaryConditions bc = PeriodicBC(10.0 * Mat3::Identity());
    Vec3 w = bc_wrap(bc, Vec3(12.0, 3.0, 0.0));
    EXPECT_NEAR(w[0], 2.0, EPS);
    EXPECT_NEAR(bc_volume(bc), 1000.0, EPS);
}

TEST(BoundaryConditions, VariantInfiniteMinImage) {
    BoundaryConditions bc = InfiniteBC(1.0);
    Vec3 d(42.0, -7.0, 0.0);
    EXPECT_NEAR((bc_min_image(bc, d) - d).norm(), 0.0, EPS);
}
