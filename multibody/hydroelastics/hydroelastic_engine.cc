#include "drake/multibody/hydroelastics/hydroelastic_engine.h"

#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_assert.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/hydroelastics/contact_surface_from_level_set.h"
#include "drake/multibody/hydroelastics/hydroelastic_field_sphere.h"

using drake::geometry::Box;
using drake::geometry::ContactSurface;
using drake::geometry::Convex;
using drake::geometry::Cylinder;
using drake::geometry::GeometryId;
using drake::geometry::HalfSpace;
using drake::geometry::Mesh;
using drake::geometry::QueryObject;
using drake::geometry::Shape;
using drake::geometry::Sphere;
using drake::geometry::SurfaceMesh;
using drake::math::RigidTransform;

#include <iostream>
#define PRINT_VAR(a) std::cout << #a": " << a << std::endl;


namespace drake {
namespace multibody {
namespace hydroelastic {
namespace internal {

template <typename T>
HydroelasticModel<T>::HydroelasticModel(
    std::unique_ptr<HydroelasticField<T>> mesh_field)
    : mesh_field_(std::move(mesh_field)) {}

template <typename T>
HydroelasticModel<T>::HydroelasticModel(
    std::unique_ptr<geometry::LevelSetField<T>> level_set)
    : level_set_(std::move(level_set)) {}

// The implementation class for HydroelasticEngine engine. Each of these
// functions mirrors a method on the HydroelasticEngine (unless otherwise
// indicated. See HydroelasticEngine for documentation.
template <typename T>
class HydroelasticEngine<T>::Impl final : public geometry::ShapeReifier {
 public:
  Impl() {}

  int num_models() const {
    return static_cast<int>(model_data_.geometry_id_to_model_.size());
  }

  const HydroelasticModel<T>* get_model(GeometryId id) const {
    auto it = model_data_.geometry_id_to_model_.find(id);
    if (it != model_data_.geometry_id_to_model_.end())
      return it->second.get();
    return nullptr;  
  }

  std::vector<ContactSurface<T>> ComputeContactSurfaces(
      const geometry::QueryObject<T>& query_object) const {
    // Models are built the first time this method is called.
    if (!model_data_.models_are_initialized_) {
      Impl* nonconst_this = const_cast<Impl*>(this);
      nonconst_this->MakeModels(query_object);
    }

#if 0
    const auto& query_port = plant_.get_geometry_query_input_port();
    const auto& query_object =
        query_port.template Eval<geometry::QueryObject<T>>(plant_context);
    const std::vector<SortedPair<GeometryId>>& geometry_pairs =
        query_object.FindCollisionCandidates();
#endif

  const std::vector<SortedPair<GeometryId>>& geometry_pairs =
        query_object.FindCollisionCandidates();

    std::vector<ContactSurface<T>> all_contact_surfaces;
    for (const auto& pair : geometry_pairs) {
      GeometryId id_M = pair.first();
      GeometryId id_N = pair.second();
      const HydroelasticModel<T>* model_M = get_model(id_M);
      const HydroelasticModel<T>* model_N = get_model(id_N);

      // Skip contact surface computation if these ids do not have a hydrostatic
      // model.
      if (!model_M || !model_N) continue;

      // Thus far we only support rigid vs. soft.
      if (model_M->is_soft() == model_N->is_soft()) {
        throw std::runtime_error(
            "The current implementation of the hydroelastic model only "
            "supports soft vs. rigid contact.");
      }
      const RigidTransform<T>& X_WM = query_object.X_WG(id_M);
      const RigidTransform<T>& X_WN = query_object.X_WG(id_N);

      // Pose of the soft model frame S in the rigid model frame R.
      // N.B. For a given state, SceneGraph broadphase reports are guaranteed to
      // always be in the same order that is, id_M < id_N.
      // Thereofore, even if we swap the id's below so that id_S (id_R) always
      // corresponds to the soft (rigid) geometry, the order still is guaranteed
      // to be the same on successive calls.
      const RigidTransform<T> X_NM = X_WN.inverse() * X_WM;
      const RigidTransform<T> X_RS = model_M->is_soft() ? X_NM : X_NM.inverse();
      const GeometryId id_S = model_M->is_soft() ? id_M : id_N;
      const GeometryId id_R = model_M->is_soft() ? id_N : id_M;
      const HydroelasticModel<T>& model_S =
          model_M->is_soft() ? *model_M : *model_N;
      const HydroelasticModel<T>& model_R =
          model_M->is_soft() ? *model_N : *model_M;

      optional<ContactSurface<T>> surface =
          CalcContactSurface(id_S, model_S, id_R, model_R, X_RS);
      if (surface) all_contact_surfaces.emplace_back(std::move(*surface));
    }

    return all_contact_surfaces;
  }

  // Implementation of ShapeReifier interface

  void ImplementGeometry(const Sphere& sphere, void* user_data) override {
    const GeometryImplementationData& specs =
        *reinterpret_cast<GeometryImplementationData*>(user_data);
    const double E = specs.effective_young_modulus;
    if (E == std::numeric_limits<double>::infinity()) {
      throw std::runtime_error(
          "Currently only soft spheres are supported. Provide a finite value "
          "of the modulus of elasticity.");
    }
    // We arbitrarily choose the refinement level so that we have 512
    // tetrahedron in the tessellation of the sphere. This provides a reasonable
    // tessellation of the sphere with a coarse mesh.
    // TODO(amcastro-tri): Make this a user setable parameter.
    const int refinement_level = 2;
    auto sphere_field = MakeUnitSphereHydroelasticField<T>(refinement_level,
                                                           sphere.get_radius());
    auto model =
        std::make_unique<HydroelasticModel<T>>(std::move(sphere_field));
    model->set_modulus_of_elasticity(E);

    model_data_.geometry_id_to_model_[specs.id] = std::move(model);
        
  }

  void ImplementGeometry(const Cylinder& cylinder, void* user_data) override {
    const GeometryImplementationData& specs =
        *reinterpret_cast<GeometryImplementationData*>(user_data);
    const double E = specs.effective_young_modulus;
    if (E != std::numeric_limits<double>::infinity()) {
      throw std::runtime_error(
          "Currently only rigid cylinders spaces are supported");
    }
    const double radius = cylinder.get_radius();
    const double half_length = cylinder.get_length() / 2.0;
    
    // Cylinder centered at the origin with its axis along the y axis.
    std::function<T(const Vector3<T>&)> cylinder_sdf =
        [radius, half_length](const Vector3<T>& p_WQ) {
          // from: http://mercury.sexy/hg_sdf/. fCylinder().
          const Vector2<T> p_xy(p_WQ[0], p_WQ[1]);
          const T sdf_cylinder_shell = p_xy.norm() - radius;
          using std::abs;
          using std::max;
          const T sdf_sides = abs(p_WQ[2]) - half_length;
          return max(sdf_cylinder_shell, sdf_sides);
        };

    std::function<Vector3<T>(const Vector3<T>&)> grad_cylinder_sdf =
        [radius, half_length](const Vector3<T>& p_WQ) {          
          const Vector2<T> p_xy(p_WQ[0], p_WQ[1]);
          const T sdf_cylinder_shell = p_xy.norm() - radius;
          using std::abs;
          using std::max;
          const T sdf_sides = abs(p_WQ[2]) - half_length;

          if (sdf_cylinder_shell > sdf_sides) {
            const Vector2<T> n_xy = p_xy.normalized();
            return Vector3<T>(n_xy[0], n_xy[1], 0.0);
          } else {
            if (p_WQ[2] > 0.0)
              return Vector3<T>(0., 0., 1.);
            else   
              return Vector3<T>(0., 0., -1.);
          }
          DRAKE_UNREACHABLE();
        };

    auto level_set = std::make_unique<geometry::LevelSetField<T>>(
        cylinder_sdf, grad_cylinder_sdf);
    model_data_.geometry_id_to_model_[specs.id] =
        std::make_unique<HydroelasticModel<T>>(std::move(level_set));
    // throw std::logic_error("There is no support for cylinders yet.");
  }

  void ImplementGeometry(const HalfSpace& half_space,
                         void* user_data) override {
    const GeometryImplementationData& specs =
        *reinterpret_cast<GeometryImplementationData*>(user_data);
    const double E = specs.effective_young_modulus;
    if (E != std::numeric_limits<double>::infinity()) {
      throw std::runtime_error(
          "Currently only rigid half spaces are supported");
    }
    auto level_set = std::make_unique<geometry::LevelSetField<T>>(
        [](const Vector3<T>& p) { return p[2]; },
        [](const Vector3<T>& p) { return Vector3<double>::UnitZ(); });
    model_data_.geometry_id_to_model_[specs.id] =
        std::make_unique<HydroelasticModel<T>>(std::move(level_set));
  }

  void ImplementGeometry(const Box& box, void* user_data) override {
#if 0        
    // Box: correct distance to corners
float fBox(vec3 p, vec3 b) {
	vec3 d = abs(p) - b;
	return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}
#endif
    // No-op. Enable this when  needed.
    return;

    const GeometryImplementationData& specs =
        *reinterpret_cast<GeometryImplementationData*>(user_data);
    const double E = specs.effective_young_modulus;
    if (E != std::numeric_limits<double>::infinity()) {
      throw std::runtime_error(
          "Currently only rigid boxes spaces are supported");
    }

    const Vector3<double> half_sizes = box.size() / 2.0;

    std::function<T(const Vector3<T>&)> box_sdf =
        [&half_sizes](const Vector3<T>& p_WQ) {
          const Vector3<T> slabs_sdf = p_WQ.cwiseAbs() - half_sizes;
          // SDF inside the box. It is negative inside and zero outside.
          const T inside_sdf = slabs_sdf.cwiseMin(T(0)).maxCoeff();
          // SDF outside the box. It is zero inside and positive outside.
          const T outside_sdf = slabs_sdf.cwiseMax(T(0)).norm();
          return inside_sdf + outside_sdf;
        };

    std::function<Vector3<T>(const Vector3<T>&)> grad_box_sdf =
        [&half_sizes](const Vector3<T>& p_WQ) {
          const Vector3<T> slabs_sdf = p_WQ.cwiseAbs() - half_sizes;
          // SDF inside the box. It is negative inside and zero outside.
          int max_index;
          const T inside_sdf = slabs_sdf.cwiseMin(T(0)).maxCoeff(&max_index);
          if (inside_sdf <= T(0)) {
            if (p_WQ[max_index] > 0)
              return Vector3<T>(Vector3<T>::Unit(max_index));
            else
              return Vector3<T>(-Vector3<T>::Unit(max_index));
          } else {
            // SDF outside the box. It is zero inside and positive outside.
            //const T outside_sdf = slabs_sdf.cwiseMax(T(0)).norm();
            const Vector3<T> x_plus = slabs_sdf.cwiseMax(T(0));
            const Vector3<T> grad = x_plus.normalized();
            return grad;
          }
          DRAKE_UNREACHABLE();
        };

    auto level_set =
        std::make_unique<geometry::LevelSetField<T>>(box_sdf, grad_box_sdf);
    model_data_.geometry_id_to_model_[specs.id] =
        std::make_unique<HydroelasticModel<T>>(std::move(level_set));
  }

  void ImplementGeometry(const Mesh& mesh, void* user_data) override {
    throw std::logic_error("There is no support for general meshes yet.");
  }

  void ImplementGeometry(const Convex& convex, void* user_data) override {
    throw std::logic_error("There is no support for convex geometry yet.");
  }

 private:
  // This struct stores additional data passed to ImplementGeometry() during
  // the reification proceses.
  struct GeometryImplementationData {
    GeometryId id;
    double effective_young_modulus;
  };

  // This struct holds the engines's data, created by the call to
  // MakeGeometryModels() the first time a query is issued.
  struct ModelData {
    bool models_are_initialized_{false};
    std::unordered_map<geometry::GeometryId,
                       std::unique_ptr<HydroelasticModel<T>>>
        geometry_id_to_model_;
  };

  HydroelasticModel<T>& get_mutable_model(GeometryId id) {
    return *model_data_.geometry_id_to_model_.at(id);
  }

  // This method is invoked the first time ComputeContactSurfaces() is called in
  // order to create the underlying computational representation for each
  // geometry in the model.
  void MakeModels(const geometry::QueryObject<T>& query_object) {
    const geometry::SceneGraphInspector<T>& inspector =
        query_object.inspector();

    // Only reify geometries with proximity roles.
    for (const geometry::GeometryId geometry_id :
         inspector.all_geometry_ids()) {
      if (const geometry::ProximityProperties* properties =
              inspector.GetProximityProperties(geometry_id)) {
        const Shape& shape = inspector.GetShape(geometry_id);
        const double elastic_modulus =
            properties->template GetPropertyOrDefault<double>(
                "hydroelastics", "elastic modulus",
                std::numeric_limits<double>::infinity());

        GeometryImplementationData specs{geometry_id, elastic_modulus};
        shape.Reify(this, &specs);
      }
    }

    // Mark the model data as initialized so that we don't perform this step on
    // the next call to ComputeContactSurfaces().
    model_data_.models_are_initialized_ = true;
  }

  // Helper method to comptue the contact surface betwen a soft model S and a
  // rigid model R with a relative pose X_RS.
  optional<ContactSurface<T>> CalcContactSurface(
      GeometryId id_S, const HydroelasticModel<T>& soft_model_S,
      GeometryId id_R, const HydroelasticModel<T>& rigid_model_R,
      const RigidTransform<T>& X_RS) const {
    DRAKE_DEMAND(soft_model_S.is_soft());
    DRAKE_DEMAND(!rigid_model_R.is_soft());
    const HydroelasticField<T>& soft_field_S =
        soft_model_S.hydroelastic_field();
    std::vector<T> e_s_surface;
    std::vector<Vector3<T>> grad_level_set_R_surface;
    std::unique_ptr<SurfaceMesh<T>> surface_R =
        geometry::internal::CalcZeroLevelSetInMeshDomain(
            soft_field_S.volume_mesh(), rigid_model_R.level_set(), X_RS,
            soft_field_S.scalar_field().values(), &e_s_surface,
            &grad_level_set_R_surface);
    if (surface_R->num_vertices() == 0) return nullopt;
    // Compute pressure field.
    for (T& e_s : e_s_surface) e_s *= soft_model_S.modulus_of_elasticity();

    auto e_s = std::make_unique<geometry::SurfaceMeshFieldLinear<T, T>>(
        "scalar field", std::move(e_s_surface), surface_R.get());
    auto grad_level_set_R =
        std::make_unique<geometry::SurfaceMeshFieldLinear<Vector3<T>, T>>(
            "gradient field", std::move(grad_level_set_R_surface),
            surface_R.get());
    // Surface and gradients are measured and expressed in the rigid frame R.
    // In consistency with ContactSurface's constract, the first id must belong
    // to the geometry associated with the frame in which quantities are
    // expressed, in this case id_R.
    return ContactSurface<T>(id_R, id_S, std::move(surface_R), std::move(e_s),
                             std::move(grad_level_set_R));
  }

  mutable ModelData model_data_;
};

template <typename T>
HydroelasticEngine<T>::HydroelasticEngine()
    : impl_(new Impl()) {
}

template <typename T>
HydroelasticEngine<T>::~HydroelasticEngine() {
  delete impl_;
}

template <typename T>
std::vector<ContactSurface<T>> HydroelasticEngine<T>::ComputeContactSurfaces(
    const geometry::QueryObject<T>& query_object) const {
  return impl_->ComputeContactSurfaces(query_object);
}

template <typename T>
int HydroelasticEngine<T>::num_models() const {
  return impl_->num_models();
}

template <typename T>
const HydroelasticModel<T>& HydroelasticEngine<T>::get_model(
    geometry::GeometryId id) const {
  return *impl_->get_model(id);
}

}  // namespace internal
}  // namespace hydroelastic
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::hydroelastic::internal::HydroelasticEngine)
