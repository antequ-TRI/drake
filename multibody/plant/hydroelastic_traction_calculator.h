#pragma once

#include "drake/geometry/proximity/surface_mesh.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/math/spatial_force.h"
#include "drake/multibody/math/spatial_velocity.h"

namespace drake {
namespace multibody {

namespace internal {

/**
 A class for computing the spatial forces on rigid bodies in a MultibodyPlant
 using the hydroelastic contact model, as described in:

 [Elandt, 2019]  R. Elandt, E. Drumwright, M. Sherman, and A. Ruina.
 A pressure field model for fast, robust approximation of net contact force and
 moment between nominally rigid objects. Proc. IEEE/RSJ Intl. Conf. on
 Intelligent Robots and Systems (IROS), 2019.
 */
template <typename T>
class HydroelasticTractionCalculator {
 public:
  /// Set of common quantities used through hydroelastic traction calculations.
  /// Documentation for parameter names (minus the `_in`
  /// suffixes) can be found in the corresponding member documentation.
  struct Data {
    Data(
        const math::RigidTransform<T>& X_WA_in,
        const math::RigidTransform<T>& X_WB_in,
        const SpatialVelocity<T>& V_WA_in,
        const SpatialVelocity<T>& V_WB_in,
        const math::RigidTransform<T>& X_WM_in,
        const geometry::ContactSurface<T>* surface_in) :
            X_WA(X_WA_in), X_WB(X_WB_in), V_WA(V_WA_in), V_WB(V_WB_in),
            X_WM(X_WM_in), surface(*surface_in),
            p_WC(X_WM_in * surface_in->mesh().centroid()) {
      DRAKE_DEMAND(surface_in);
    }

    /// The pose of Body A (the body that Geometry `surface.M_id()` in the
    /// contact surface is affixed to) in the world frame.
    const math::RigidTransform<T> X_WA;

    /// The pose of Body B (the body that Geometry `surface.N_id()` in the
    /// contact surface is affixed to) in the world frame.
    const math::RigidTransform<T> X_WB;

    /// The spatial velocity of Body A (the body that Geometry
    /// `surface.M_id()` in the contact surface is affixed to) at the origin of
    /// A's frame, measured and expressed in the world frame.
    const SpatialVelocity<T> V_WA;

    /// The spatial velocity of Body B (the body that Geometry
    /// `surface.N_id()` in the contact surface is affixed to) at the origin of
    /// B's frame, measured and expressed in the world frame.
    const SpatialVelocity<T> V_WB;

    /// The pose of Geometry `surface.M_id()` in the world frame.
    const math::RigidTransform<T> X_WM;

    /// A pointer to the ContactSurface that must be maintained for the life
    /// of this object.
    const geometry::ContactSurface<T>& surface;

    /// The traction computation needs a point C near the contact surface at
    /// which to accumulate forces in a numerically robust way. Our calculations
    /// define C to be the centroid of the contact surface, and measure and
    /// express this point in the world frame.
    const Vector3<T> p_WC;
  };

 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(HydroelasticTractionCalculator)

  HydroelasticTractionCalculator() {}

  /**
   Gets the regularization parameter used for friction (in m/s). The closer
   that this parameter is to zero, the closer that the regularized friction
   model will approximate Coulomb friction.
   */
  double regularization_scalar() const { return vslip_regularizer_; }

  /**
   Applies the hydroelastic model to two geometries defined in `surface`,
   resulting in a pair of spatial forces at the origins of two body frames.
   The body frames, A and B, are those to which `surface.M_id()` and
   `surface.N_id()` are affixed, respectively.
   @param data Relevant kinematic data.
   @param dissipation the nonnegative coefficient (in s/m) for dissipating
          energy along the direction of the surface normals.
   @param mu_coulomb the nonnegative coefficient for Coulomb friction.
   @param[output] F_Ao_W the spatial force on Body A, on return.
   @param[output] F_Bo_W the spatial force on Body B, on return.
   */ 
  void ComputeSpatialForcesAtBodyOriginsFromHydroelasticModel(
       const Data& data, double dissipation, double mu_coulomb,
       multibody::SpatialForce<T>* F_Ao_W,
       multibody::SpatialForce<T>* F_Bo_W) const;

 private:
  // To allow GTEST to test private functions.
  friend class MultibodyPlantHydroelasticTractionTests;

  Vector3<T> CalcTractionAtPoint(
      const Data& data, geometry::SurfaceFaceIndex face_index,
      const typename geometry::SurfaceMesh<T>::Barycentric& Q_barycentric,
      double dissipation, double mu_coulomb, Vector3<T>* p_WQ) const;

  multibody::SpatialForce<T> ComputeSpatialTractionAtAcFromTractionAtAq(
      const Data& data, const Vector3<T>& p_WQ,
      const Vector3<T>& traction_Aq_W) const;

  // The parameter (in m/s) for regularizing the Coulomb friction model.
  double vslip_regularizer_{1e-6};
};

}  // namespace internal
}  // namespace multibody
}  // namespace drake

// TODO(edrumwri) instantiate on SymbolicExpression when it no longer
// causes a linker error complaining about an unresolved symbol in SceneGraph.
DRAKE_DECLARE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class drake::multibody::internal::HydroelasticTractionCalculator)
