#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "drake/geometry/proximity/make_unit_sphere_mesh.h"
#include "drake/multibody/hydroelastics/hydroelastic_field.h"

namespace drake {
namespace multibody {
namespace hydroelastic {
namespace internal {

/// Creates a HydroelasticField model for a sphere of a given radius.
/// The input parameter `refinement_level`, ℓ ∈ ℕ₀, controls the the number of
/// vertices (and faces) generated. The resulting number of tetrahedra nₜ can
/// be predicted according to: nₜ = 8ˡ⁺¹.
/// Even though the dimensionless hydroelastic strain field is arbitrary, we
/// construct it to satisfy a number of properties:
///  1. The field ε and its gradient ∇ε are continuous in the domain of the
///     sphere.
///  2. The radial gradient at the boundary of the sphere is dε/dr = -1 / R,
///     with R the radius of the sphere, so that if the field was linear, then
///     the maximum strain ε = 1 would occur at the center of the sphere.
///
/// We then choose the simplest functional form that can satisfy these
/// requirements; a quadratic function of the radius, ε(r) = 0.5 [1 - (r / R)²].
template <typename T>
std::unique_ptr<HydroelasticField<T>> MakeUnitSphereHydroelasticField(
    int refinement_level, double sphere_radius) {
  geometry::VolumeMesh<T> unit_sphere_mesh =
      geometry::internal::MakeUnitSphereMesh<T>(refinement_level);

  // Scale the unit sphere to have the desired radius.
  std::vector<geometry::VolumeElement> tetrahedra =
      unit_sphere_mesh.tetrahedra();
  std::vector<geometry::VolumeVertex<T>> vertices;
  vertices.reserve(unit_sphere_mesh.num_vertices());
  for (const auto& v : unit_sphere_mesh.vertices()) {
    const Vector3<T> scaled_v = sphere_radius * v.r_MV();
    vertices.emplace_back(scaled_v);
  }
  auto mesh = std::make_unique<geometry::VolumeMesh<T>>(std::move(tetrahedra),
                                                        std::move(vertices));

  // Analytic pressure field and gradient.
  std::vector<T> e_mn_values(mesh->vertices().size());
  std::vector<Vector3<T>> grad_e_mn_values(mesh->vertices().size());
  const double sphere_radius_squared = sphere_radius * sphere_radius;
  for (geometry::VolumeVertexIndex v(0); v < mesh->num_vertices(); ++v) {
    const Vector3<T>& p_MV = mesh->vertex(v).r_MV();
    const T radius_squared = p_MV.squaredNorm();
    // x is the dimensionless radius squared.
    const T x = radius_squared / sphere_radius_squared;
    // The dimensionless strain field ε.
    e_mn_values[v] = 0.5 * (1.0 - x);

    // Gradient ∇ε = dε/dr⋅r̂ = -r / R².
    grad_e_mn_values[v] = -p_MV / sphere_radius_squared;
  }
  auto e_mn = std::make_unique<geometry::VolumeMeshFieldLinear<T, T>>(
      "Sphere Pressure Field", std::move(e_mn_values), mesh.get());
  auto grad_e_mn =
      std::make_unique<geometry::VolumeMeshFieldLinear<Vector3<T>, T>>(
          "Sphere Pressure Gradient Field", std::move(grad_e_mn_values),
          mesh.get());
  return std::make_unique<HydroelasticField<T>>(
      std::move(mesh), std::move(e_mn), std::move(grad_e_mn));
}

}  // namespace internal
}  // namespace hydroelastic
}  // namespace multibody
}  // namespace drake
