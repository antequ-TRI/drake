#include "drake/multibody/hydroelastics/hydroelastic_engine.h"

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/hydroelastics/contact_surface_from_level_set.h"
#include "drake/multibody/hydroelastics/hydroelastic_field_box.h"
#include "drake/multibody/hydroelastics/hydroelastic_field_cylinder.h"
#include "drake/multibody/hydroelastics/hydroelastic_field_sphere.h"
#include "drake/multibody/hydroelastics/write_meshes.h"

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
using drake::geometry::VolumeMesh;
using drake::math::RigidTransform;

#include <iostream>
#include <fstream>
//#define PRINT_VAR(a) std::cout << #a": " << a << std::endl;
//#define PRINT_VAR(a) (void) (a);

namespace drake {
namespace multibody {
namespace hydroelastics {
namespace internal {

template <typename T>
HydroelasticGeometry<T>::HydroelasticGeometry(
    std::unique_ptr<HydroelasticField<T>> mesh_field)
    : mesh_field_(std::move(mesh_field)) {}

template <typename T>
HydroelasticGeometry<T>::HydroelasticGeometry(
    std::unique_ptr<LevelSetField<T>> level_set)
    : level_set_(std::move(level_set)) {}

template <typename T>
void HydroelasticEngine<T>::MakeModels(
    const geometry::SceneGraphInspector<T>& inspector) {
  // Only reify geometries with proximity roles.
  for (const geometry::GeometryId geometry_id : inspector.all_geometry_ids()) {
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

template <typename T>
int HydroelasticEngine<T>::num_models() const {
  return static_cast<int>(model_data_.geometry_id_to_model_.size());
}

template <typename T>
const HydroelasticGeometry<T>* HydroelasticEngine<T>::get_model(
    geometry::GeometryId id) const {
  auto it = model_data_.geometry_id_to_model_.find(id);
  if (it != model_data_.geometry_id_to_model_.end()) return it->second.get();
  return nullptr;
}

template <typename T>
std::vector<ContactSurface<T>> HydroelasticEngine<T>::ComputeContactSurfaces(
    const geometry::QueryObject<T>& query_object) const {
  const std::vector<SortedPair<GeometryId>>& geometry_pairs =
      query_object.FindCollisionCandidates();

  std::vector<ContactSurface<T>> all_contact_surfaces;
  for (const auto& pair : geometry_pairs) {
    GeometryId id_M = pair.first();
    GeometryId id_N = pair.second();
    const HydroelasticGeometry<T>* model_M = get_model(id_M);
    const HydroelasticGeometry<T>* model_N = get_model(id_N);

    // Skip contact surface computation if these ids do not have a hydrostatic
    // model.
    if (!model_M || !model_N) {
      const std::string name_M = query_object.inspector().GetName(id_M);
      const std::string name_N = query_object.inspector().GetName(id_N);
      throw std::runtime_error(
          "HydroelasticEngine. Unsupported geometries possibly in contact. "
          "For the geometry pair ('" + name_M + "', '" + name_N + "')."
          "You can remove the unsupported geometries, replace them with "
          "supported geometry, or filter collisions on them.");
    }

    // Thus far we only support rigid vs. soft.
    if (model_M->is_soft() == model_N->is_soft()) {
      throw std::runtime_error(
          "HydroelasticEngine. The current implementation of the "
          "hydroelastic model only supports soft vs. rigid contact.");
    }
    const RigidTransform<T>& X_WM = query_object.X_WG(id_M);
    const RigidTransform<T>& X_WN = query_object.X_WG(id_N);

    // Pose of the soft model frame S in the rigid model frame R.
    // N.B. For a given state, SceneGraph broadphase reports are guaranteed to
    // always be in the same order that is, id_M < id_N.
    // Therefore, even if we swap the id's below so that id_S (id_R) always
    // corresponds to the soft (rigid) geometry, the order still is guaranteed
    // to be the same on successive calls.
    const RigidTransform<T> X_NM = X_WN.inverse() * X_WM;
    const RigidTransform<T> X_RS = model_M->is_soft() ? X_NM : X_NM.inverse();
    const GeometryId id_S = model_M->is_soft() ? id_M : id_N;
    const GeometryId id_R = model_M->is_soft() ? id_N : id_M;
    const HydroelasticGeometry<T>& model_S =
        model_M->is_soft() ? *model_M : *model_N;
    const HydroelasticGeometry<T>& model_R =
        model_M->is_soft() ? *model_N : *model_M;

    //PRINT_VAR(model_S.hydroelastic_field().volume_mesh().num_vertices());
    //PRINT_VAR(model_S.hydroelastic_field().volume_mesh().num_elements());

    optional<ContactSurface<T>> surface =
        CalcContactSurface(id_S, model_S, id_R, model_R, X_RS);
    //if (surface) {
    //  vtkio::write_vtk_mesh("vol_mesh_from_calc_method.vtk",
    //                model_S.hydroelastic_field().volume_mesh());
    //}

    if (surface) all_contact_surfaces.emplace_back(std::move(*surface));
  }

  return all_contact_surfaces;
}

template <typename T>
optional<ContactSurface<T>> HydroelasticEngine<T>::CalcContactSurface(
    GeometryId id_S, const HydroelasticGeometry<T>& soft_model_S,
    GeometryId id_R, const HydroelasticGeometry<T>& rigid_model_R,
    const RigidTransform<T>& X_RS) const {
  DRAKE_DEMAND(soft_model_S.is_soft());
  DRAKE_DEMAND(!rigid_model_R.is_soft());
  const HydroelasticField<T>& soft_field_S = soft_model_S.hydroelastic_field();
  std::vector<T> e_s_surface;
  std::vector<Vector3<T>> grad_level_set_R_surface;
  std::unique_ptr<SurfaceMesh<T>> surface_R = CalcZeroLevelSetInMeshDomain(
      soft_field_S.volume_mesh(), rigid_model_R.level_set(), X_RS,
      soft_field_S.scalar_field().values(), &e_s_surface,
      &grad_level_set_R_surface);
  if (surface_R->num_vertices() == 0) return nullopt;
  // Compute pressure field.
  for (T& e_s : e_s_surface) e_s *= soft_model_S.elastic_modulus();

  // ∇hₘₙ is a vector that points from N (in this case S) into M (in this case
  // R). However, the gradient of the level set function points into S (N).
  // Therefore we flip its direction.
  for (Vector3<T>& grad_level_set_R : grad_level_set_R_surface) {
    grad_level_set_R = -grad_level_set_R;
  }

  auto e_s = std::make_unique<geometry::SurfaceMeshFieldLinear<T, T>>(
      "e_MN", std::move(e_s_surface), surface_R.get());
  auto grad_level_set_R =
      std::make_unique<geometry::SurfaceMeshFieldLinear<Vector3<T>, T>>(
          "grad_h_MN_M", std::move(grad_level_set_R_surface), surface_R.get());
  // Surface and gradients are measured and expressed in the rigid frame R.
  // In consistency with ContactSurface's contract, the first id must belong
  // to the geometry associated with the frame in which quantities are
  // expressed, in this case id_R.
  return ContactSurface<T>(id_R, id_S, std::move(surface_R), std::move(e_s),
                           std::move(grad_level_set_R), X_RS);
}

template <typename T>
void HydroelasticEngine<T>::ImplementGeometry(const Sphere& sphere,
                                              void* user_data) {
  const GeometryImplementationData& specs =
      *reinterpret_cast<GeometryImplementationData*>(user_data);
  const double elastic_modulus = specs.elastic_modulus;
  if (elastic_modulus == std::numeric_limits<double>::infinity()) {
    drake::log()->warn(
        "HydroelasticEngine. The current hydroelastic model implementation "
        "does not support rigid spheres. Geometry ignored.");
  }
  // We arbitrarily choose the refinement level so that we have 512
  // tetrahedron in the tessellation of the sphere. This provides a reasonable
  // tessellation of the sphere with a coarse mesh.
  // TODO(amcastro-tri): Make this a user setable parameter.
  const int refinement_level = 2;
  auto sphere_field =
      MakeSphereHydroelasticField<T>(refinement_level, sphere.get_radius());
  auto model =
      std::make_unique<HydroelasticGeometry<T>>(std::move(sphere_field));
  model->set_elastic_modulus(elastic_modulus);

  model_data_.geometry_id_to_model_[specs.id] = std::move(model);
}

template <typename T>
void HydroelasticEngine<T>::ImplementGeometry(const HalfSpace&,
                                              void* user_data) {
  const GeometryImplementationData& specs =
      *reinterpret_cast<GeometryImplementationData*>(user_data);
  const double elastic_modulus = specs.elastic_modulus;
  if (elastic_modulus != std::numeric_limits<double>::infinity()) {
    drake::log()->warn(
        "HydroelasticEngine. The current hydroelastic model implementation "
        "does not support soft half-spaces. Geometry ignored.");
  }
  auto level_set = std::make_unique<LevelSetField<T>>(
      [](const Vector3<T>& p) { return p[2]; },
      [](const Vector3<T>&) { return Vector3<double>::UnitZ(); });
  model_data_.geometry_id_to_model_[specs.id] =
      std::make_unique<HydroelasticGeometry<T>>(std::move(level_set));
}

template <typename T>
void HydroelasticEngine<T>::ImplementGeometry(const Cylinder& cylinder,
                                              void* user_data) {
  const GeometryImplementationData& specs =
      *reinterpret_cast<GeometryImplementationData*>(user_data);
  const double E = specs.elastic_modulus;

  // Model of a soft cylinder.
  if (E != std::numeric_limits<double>::infinity()) {
    const int refinement_level = 2;
    auto cylinder_field =
        MakeCylinderHydroelasticField<T>(cylinder, refinement_level);
    //vtkio::write_vtk_mesh("vol_mesh_from_implement_geometry.vtk",
    //                cylinder_field->volume_mesh());    
    auto model =
        std::make_unique<HydroelasticGeometry<T>>(std::move(cylinder_field));
    model->set_elastic_modulus(E);
    model_data_.geometry_id_to_model_[specs.id] = std::move(model);
    std::cout << "MakeCylinderHydroelasticField() called\n";
    return;
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

  auto level_set = std::make_unique<LevelSetField<T>>(
      cylinder_sdf, grad_cylinder_sdf);
  model_data_.geometry_id_to_model_[specs.id] =
      std::make_unique<HydroelasticGeometry<T>>(std::move(level_set));
  // throw std::logic_error("There is no support for cylinders yet.");
}

template <typename T>
void HydroelasticEngine<T>::ImplementGeometry(const Box& box, void* user_data) {
#if 0        
    // Box: correct distance to corners
float fBox(vec3 p, vec3 b) {
	vec3 d = abs(p) - b;
	return length(max(d, vec3(0))) + vmax(min(d, vec3(0)));
}
#endif
  // No-op. Enable this when  needed.
  //return;

  const GeometryImplementationData& specs =
      *reinterpret_cast<GeometryImplementationData*>(user_data);
  const double elastic_modulus = specs.elastic_modulus;
  
  if (elastic_modulus != std::numeric_limits<double>::infinity()) {
      // Soft mesh field model
      const T mesh_size = box.size().maxCoeff() / 5;
      auto box_field = MakeBoxHydroelasticField<T>(box, mesh_size);
      auto model =
          std::make_unique<HydroelasticGeometry<T>>(std::move(box_field));
      model->set_elastic_modulus(elastic_modulus);
      model_data_.geometry_id_to_model_[specs.id] = std::move(model);
  } else {
    // Rigid level set model.
    const Vector3<double> half_sizes = box.size() / 2.0;

    std::function<T(const Vector3<T>&)> box_sdf =
        [half_sizes](const Vector3<T>& p_WQ) {
          const Vector3<T> slabs_sdf = p_WQ.cwiseAbs() - half_sizes;
          // SDF inside the box. It is negative inside and zero outside.
          const T inside_sdf = slabs_sdf.cwiseMin(T(0)).maxCoeff();
          // SDF outside the box. It is zero inside and positive outside.
          const T outside_sdf = slabs_sdf.cwiseMax(T(0)).norm();
          return inside_sdf + outside_sdf;
        };

    std::function<Vector3<T>(const Vector3<T>&)> grad_box_sdf =
        [half_sizes](const Vector3<T>& p_WQ) {
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
            // const T outside_sdf = slabs_sdf.cwiseMax(T(0)).norm();
            const Vector3<T> x_plus = slabs_sdf.cwiseMax(T(0));
            const Vector3<T> grad = x_plus.normalized();
            return grad;
          }
          DRAKE_UNREACHABLE();
        };

    auto level_set = std::make_unique<LevelSetField<T>>(box_sdf, grad_box_sdf);
    model_data_.geometry_id_to_model_[specs.id] =
        std::make_unique<HydroelasticGeometry<T>>(std::move(level_set));
  }
}

// The following overrides are no-ops given that currently HydroelasticEngine
// does not support these geometries.
template <typename T>
void HydroelasticEngine<T>::ImplementGeometry(const Mesh&, void*) {
  drake::log()->warn(
      "HydroelasticEngine. The current hydroelastic model implementation "
      "does not support mesh geometries. Geometry ignored.");
}

template <typename T>
void HydroelasticEngine<T>::ImplementGeometry(const Convex&, void*) {
  drake::log()->warn(
      "HydroelasticEngine. The current hydroelastic model implementation "
      "does not support convex geometries. Geometry ignored.");
}

}  // namespace internal
}  // namespace hydroelastics
}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::hydroelastics::internal::HydroelasticEngine)
