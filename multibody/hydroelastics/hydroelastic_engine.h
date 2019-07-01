#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "drake/common/drake_optional.h"
#include "drake/common/eigen_types.h"
#include "drake/geometry/proximity/volume_mesh.h"
#include "drake/geometry/query_results/contact_surface.h"
#include "drake/geometry/query_object.h"
#include "drake/multibody/hydroelastics/hydroelastic_field.h"
#include "drake/multibody/hydroelastics/level_set_field.h"

namespace drake {
namespace multibody {
namespace hydroelastic {
namespace internal {

/// This class provides the underlying computational representation for each
/// geometry in the model.
template <typename T>
class HydroelasticModel {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(HydroelasticModel)

  /// Creates a soft model from a given %HydroelasticField object.
  explicit HydroelasticModel(std::unique_ptr<HydroelasticField<T>> mesh_field);

  /// Constructor for a rigid model with its geometry represented by the
  /// zero-level of a level set function.
  explicit HydroelasticModel(
      std::unique_ptr<geometry::LevelSetField<T>> level_set);

  /// Returns `true` iff the underlying model represents a soft object.
  bool is_soft() const {
    // There must be only one representation.
    DRAKE_ASSERT(mesh_field_.has_value() != level_set_.has_value());
    return mesh_field_.has_value();
  }

  /// Returns a reference to the underlying HydroelasticField object.
  /// If aborts if the model is not soft.
  const HydroelasticField<T>& hydroelastic_field() const {
    DRAKE_DEMAND(is_soft());
    return *mesh_field_.value();
  }

  /// Returns the underlying geometry::LevelSetField object.
  /// It aborts if the model is not rigid.
  const geometry::LevelSetField<T>& level_set() const {
    DRAKE_DEMAND(!is_soft());
    return *level_set_.value();
  }

  /// Returns the modulus of elasticity for `this` model.
  /// If infinity, the model is considered to be rigid.
  double modulus_of_elasticity() const { return modulus_of_elasticity_; }

  /// Sets the modulus of elasticity for `this` model.
  /// If infinity, the model is considered to be rigid.
  void set_modulus_of_elasticity(double E) { modulus_of_elasticity_ = E; }

 private:
  // Model is rigid by default.
  double modulus_of_elasticity_{std::numeric_limits<double>::infinity()};
  optional<std::unique_ptr<HydroelasticField<T>>> mesh_field_;
  optional<std::unique_ptr<geometry::LevelSetField<T>>> level_set_;
};

/// The underlying engine to perform the geometric computations needed by the
/// hydroelastic model described in:
/// [Elandt, 2019]  R. Elandt, E. Drumwright, M. Sherman, and A. Ruina.
/// A pressure field model for fast, robust approximation of net contact force
/// and moment between nominally rigid objects. Proc. IEEE/RSJ Intl. Conf. on
/// Intelligent Robots and Systems (IROS), 2019.
///
/// This engine:
///  - Creates the internal representation of the geometric models as
///    HydroelasticModel instances. This creation takes place on the first
///    context-based query.
///  - Owns the HydroelasticModel instances for each geometry in the model.
///  - Provides API to perform hydroelastic model specific geometric queries.
///
/// Instantiated templates for the following kinds of T's are provided:
///
/// - double
/// - AutoDiffXd
///
/// They are already available to link against in the containing library.
/// No other values for T are currently supported.
template <typename T>
class HydroelasticEngine {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(HydroelasticEngine)

  /// Constructor for an engine to be used by a given `plant`.
  /// %HydroelasticEngine keeps a reference to `plant` that must remain valid
  /// for the lifetime of the engine.
  ///
  /// %HydroelasticEngine will create an appropriate computation model for each
  /// geometry registered with this `plant` on the first call to
  /// `ComputeContactSurfaces()`.
  ///
  /// @throws std::exception if `plant` is nullptr.
  HydroelasticEngine();

  ~HydroelasticEngine();

  /// Returns the number of underlying %HydroelasticModel models.
  int num_models() const;

  /// Returns a constant reference to the underlying HydroelasticModel for the
  /// given geometry identified by its `id`.
  const HydroelasticModel<T>& get_model(geometry::GeometryId id) const;

  /// For a given state of the associated plant (provided at construction) as
  /// stored in `plant_context`, this method computes the contact surfaces for
  /// all geometries in contact.
  /// On the first call to this method, the engine creates the underlying
  /// computational representation for each geometry in the model, which is then
  /// used in future queries.
  std::vector<geometry::ContactSurface<T>> ComputeContactSurfaces(
      const geometry::QueryObject<T>& query_object) const;

 private:
  class Impl;
  Impl* impl_{};
};

template <>
class HydroelasticEngine<symbolic::Expression> {
 public:
  //  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(HydroelasticEngine<symbolic::Expression>)

  using T = symbolic::Expression;

  HydroelasticEngine() {}

  ~HydroelasticEngine() {}

  std::vector<geometry::ContactSurface<T>> ComputeContactSurfaces(
      const geometry::QueryObject<T>& query_object) const {
    Throw("ComputeContactSurfaces");
    return std::vector<geometry::ContactSurface<T>>();
  }

 private:
  static void Throw(const char* operation_name) {
    throw std::logic_error(
        fmt::format("Cannot {} on a HydroelasticEngine<symbolic::Expression>",
                    operation_name));
  }
};

}  // namespace internal
}  // namespace hydroelastic
}  // namespace multibody
}  // namespace drake
