#pragma once

#include <memory>
#include <utility>

#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/proximity/volume_mesh_field.h"

namespace drake {
namespace multibody {
namespace hydroelastic {
namespace internal {

/// This class stores the tetrahedral mesh, scalar, and vector fields for the
/// hydroelastic model.
template <typename T>
class HydroelasticField {
 public:
  /// Constructor of a HydroelasticField. It takes ownership of its input
  /// arguments.
  /// @param[in] mesh_M
  ///   The tetrahedral mesh representation of the geometry, with position
  ///   vectors measured and expressed in the frame M of the model.
  /// @param[in] e_m
  ///   The volumetric scalar field of the hydroelastic model.
  /// @param[in] grad_e_m_M
  ///   The gradient of the scalar field e_m, expressed in the frame M of the
  ///   mesh.
  HydroelasticField(
      std::unique_ptr<geometry::VolumeMesh<T>> mesh_M,
      std::unique_ptr<geometry::VolumeMeshFieldLinear<T, T>> e_m,
      std::unique_ptr<geometry::VolumeMeshFieldLinear<Vector3<T>, T>>
          grad_e_m_M)
      : mesh_M_(std::move(mesh_M)),
        e_m_(std::move(e_m)),
        grad_e_m_M_(std::move(grad_e_m_M)) {}

  HydroelasticField(HydroelasticField&&) = default;
  HydroelasticField& operator=(HydroelasticField&&) = default;

  const geometry::VolumeMesh<T>& volume_mesh() const { return *mesh_M_; }

  const geometry::VolumeMeshFieldLinear<T, T>& scalar_field() const {
    return *e_m_;
  }

  const geometry::VolumeMeshFieldLinear<Vector3<T>, T>& gradient_field() const {
    return *grad_e_m_M_;
  }

 private:
  /** The surface mesh of the contact surface 𝕊ₘₙ between M and N. */
  std::unique_ptr<geometry::VolumeMesh<T>> mesh_M_;
  /** Represents the scalar field eₘₙ on the surface mesh. */
  std::unique_ptr<geometry::VolumeMeshFieldLinear<T, T>> e_m_;
  /** Represents the vector field ∇hₘₙ on the surface mesh, expressed in M's
    frame */
  std::unique_ptr<geometry::VolumeMeshFieldLinear<Vector3<T>, T>> grad_e_m_M_;
};

}  // namespace internal
}  // namespace hydroelastic
}  // namespace multibody
}  // namespace drake
