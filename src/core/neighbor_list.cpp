#include "forcesmith/core/neighbor_list.hpp"
#include "forcesmith/core/potential_base.hpp"

#include <boost/container_hash/hash.hpp>

#include <algorithm>
#include <cmath>
#include <execution>
#include <span>
#include <variant>
#include <vector>

namespace forcesmith {

namespace {

// Fingerprint of everything that determines the neighbour list besides rcut and
// the potential table: atom count, positions, types and the periodic box. Cheap
// (O(N)) compared to the O(N²·images) build it guards.
std::size_t geometry_signature(const Configuration &cfg) {
  std::size_t h = 0;
  boost::hash_combine(h, cfg.atoms.size());
  for (const Atom &a : cfg.atoms) {
    boost::hash_combine(h, a.pos(0));
    boost::hash_combine(h, a.pos(1));
    boost::hash_combine(h, a.pos(2));
    boost::hash_combine(h, a.type.Z);
    boost::hash_combine(h, a.type.index);
  }
  if (const auto *pbc = std::get_if<PeriodicBC>(&cfg.bc)) {
    const Mat3 &box = pbc->box();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        boost::hash_combine(h, box(r, c));
  } else {
    boost::hash_combine(h, true); // non-periodic discriminant
  }
  return h;
}

const Potential *resolve_pot(const PotentialPair *pots, const Atom &ai,
                             const Atom &aj) {
  if (pots && ai.type < pots->ntypes() && aj.type < pots->ntypes()) {
    return &(*pots)[ai.type, aj.type];
  }
  return nullptr;
}

void add_neighbor(Configuration &cfg, const PotentialPair *pots, std::size_t i,
                  std::size_t j, const Vec3 &d, double rcut2) {
  if (d.squaredNorm() > rcut2) {
    return;
  }
  NeighborEntry entry;
  entry.neighbor = &cfg.atoms[j];
  entry.pot = resolve_pot(pots, cfg.atoms[i], cfg.atoms[j]);
  entry.dist = d;
  cfg.atoms[i].neighbors.push_back(entry);
}

// Periodic build: replicate the cell out to ceil(rcut / box_height) image
// shells per lattice direction (matching forcesmith's config.c), so that for
// cells smaller than the cutoff every periodic neighbour — including an atom's
// own images — is found. Each ordered pair is stored on the owning atom; the
// equal-and-opposite mirror entry supplies the reaction on the partner, and the
// +R/−R image entries of a self-pair cancel (zero net self force) and each
// contribute the correct 0.5·phi, so no separate "self" handling is needed.
void build_periodic(Configuration &cfg, double rcut, double rcut2,
                    const PotentialPair *pots, const PeriodicBC &pbc) {
  const Mat3 &box = pbc.box();
  const Vec3 a = box.col(0), b = box.col(1), c = box.col(2);

  // Image shells needed per axis: ceil(rcut * |reciprocal lattice vector|),
  // where the reciprocal vectors are the rows of the inverse box.
  const Mat3 &inv = pbc.inv_box();
  const int sx = static_cast<int>(std::ceil(rcut * inv.row(0).norm()));
  const int sy = static_cast<int>(std::ceil(rcut * inv.row(1).norm()));
  const int sz = static_cast<int>(std::ceil(rcut * inv.row(2).norm()));

  // Wrap positions into the unit cell so the base separation is minimal and
  // the image shells above are guaranteed to cover the cutoff sphere.
  std::vector<Vec3> wpos(cfg.atoms.size());
  std::ranges::transform(cfg.atoms, wpos.begin(),
                         [&](const Atom &atom) { return pbc.wrap(atom.pos); });

  for (std::size_t i = 0; i < cfg.atoms.size(); ++i) {
    for (std::size_t j = 0; j < cfg.atoms.size(); ++j) {
      const Vec3 base = wpos[j] - wpos[i];
      for (int ix = -sx; ix <= sx; ++ix)
        for (int iy = -sy; iy <= sy; ++iy)
          for (int iz = -sz; iz <= sz; ++iz) {
            if (i == j && ix == 0 && iy == 0 && iz == 0) {
              continue; // skip an atom paired with itself in the home cell
            }
            add_neighbor(cfg, pots, i, j, base + ix * a + iy * b + iz * c,
                         rcut2);
          }
    }
  }
}

// Non-periodic (cluster) build: direct pairs only, no images.
void build_infinite(Configuration &cfg, double rcut2,
                    const PotentialPair *pots) {
  for (std::size_t i = 0; i < cfg.atoms.size(); ++i)
    for (std::size_t j = 0; j < cfg.atoms.size(); ++j) {
      if (i == j) {
        continue;
      }
      add_neighbor(cfg, pots, i, j, cfg.atoms[j].pos - cfg.atoms[i].pos, rcut2);
    }
}

void build_impl(Configuration &cfg, double rcut, const PotentialPair *pots) {
  // Cache hit: an identical list was already built for THIS configuration
  // object (parent stamp rules out copies/loads, whose neighbour pointers would
  // dangle into the source) with the same cutoff, potential table and geometry.
  const std::size_t sig = geometry_signature(cfg);
  const void *pots_key = static_cast<const void *>(pots);
  if (cfg.nl_valid && cfg.nl_rcut == rcut && cfg.nl_pots == pots_key &&
      cfg.nl_sig == sig && !cfg.atoms.empty() &&
      cfg.atoms.front().parent == &cfg) {
    return;
  }

  std::ranges::for_each(cfg.atoms, [&cfg](auto &a) {
    a.neighbors.clear();
    a.parent = &cfg; // stamp owning config (transient; mirrors neighbor ptrs)
  });
  const double rcut2 = rcut * rcut;

  if (const auto *pbc = std::get_if<PeriodicBC>(&cfg.bc)) {
    build_periodic(cfg, rcut, rcut2, pots, *pbc);
  } else {
    build_infinite(cfg, rcut2, pots);
  }

  cfg.nl_valid = true;
  cfg.nl_rcut = rcut;
  cfg.nl_pots = pots_key;
  cfg.nl_sig = sig;
}

} // anonymous namespace

void build_neighbor_list(Configuration &cfg, double rcut) {
  build_impl(cfg, rcut, nullptr);
}

void build_neighbor_list(Configuration &cfg, double rcut,
                         const PotentialPair &pots) {
  build_impl(cfg, rcut, &pots);
}

void build_all_neighbor_lists(std::span<Configuration> configs, double rcut) {
  std::for_each(std::execution::par, configs.begin(), configs.end(),
                [rcut](Configuration &cfg) { build_impl(cfg, rcut, nullptr); });
}

} // namespace forcesmith
