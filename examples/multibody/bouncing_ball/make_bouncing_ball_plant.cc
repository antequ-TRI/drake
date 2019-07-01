#include "drake/examples/multibody/bouncing_ball/make_bouncing_ball_plant.h"

#include "drake/multibody/tree/uniform_gravity_field_element.h"

namespace drake {
namespace examples {
namespace multibody {
namespace bouncing_ball {

using drake::multibody::CoulombFriction;
using drake::multibody::MultibodyPlant;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::UnitInertia;
using math::RigidTransformd;
using geometry::Sphere;
using geometry::HalfSpace;
using geometry::SceneGraph;

std::unique_ptr<drake::multibody::MultibodyPlant<double>>
MakeBouncingBallPlant(double radius, double mass,
                      double elastic_modulus, double dissipation,
                      const CoulombFriction<double>& surface_friction,
                      const Vector3<double>& gravity_W,
                      SceneGraph<double>* scene_graph) {
  auto plant = std::make_unique<MultibodyPlant<double>>();

  UnitInertia<double> G_Bcm = UnitInertia<double>::SolidSphere(radius);
  SpatialInertia<double> M_Bcm(mass, Vector3<double>::Zero(), G_Bcm);

  const RigidBody<double>& ball = plant->AddRigidBody("Ball", M_Bcm);

  if (scene_graph != nullptr) {
    plant->RegisterAsSourceForSceneGraph(scene_graph);

    Vector3<double> normal_W(0, 0, 1);
    Vector3<double> point_W(0, 0, 0);

    const RigidTransformd X_WG(HalfSpace::MakePose(normal_W, point_W));
    // A half-space for the ground geometry.
    plant->RegisterCollisionGeometry(plant->world_body(), X_WG, HalfSpace(),
                                     "collision", surface_friction);

    // Add visual for the ground.
    plant->RegisterVisualGeometry(plant->world_body(), X_WG, HalfSpace(),
                                  "visual");

    // Add sphere geometry for the ball.
    // Pose of sphere geometry S in body frame B.
    const RigidTransformd X_BS = RigidTransformd::Identity();
    geometry::GeometryId sphere_collision_id = plant->RegisterCollisionGeometry(
        ball, X_BS, Sphere(radius), "collision", surface_friction);

    // set modulus of elasticity.
    plant->set_elastic_modulus(sphere_collision_id, elastic_modulus);
    plant->set_hydroelastics_dissipation(sphere_collision_id, dissipation);

    // Add visual for the ball.
    const Vector4<double> orange(1.0, 0.55, 0.0, 1.0);
    plant->RegisterVisualGeometry(ball, X_BS, Sphere(radius), "visual", orange);

    const Vector4<double> purple(0.6, 0.2, 0.8, 1.0);
    const double visual_radius = 0.2 * radius;
    plant->RegisterVisualGeometry(ball, Eigen::Translation3d(0., 0., radius),
                                  Sphere(visual_radius), "sphere_z+", purple);
    plant->RegisterVisualGeometry(ball, Eigen::Translation3d(0., 0., -radius),
                                  Sphere(visual_radius), "sphere_z-", purple);
    plant->RegisterVisualGeometry(ball, Eigen::Translation3d(radius, 0., 0.),
                                  Sphere(visual_radius), "sphere_x+", purple);
    plant->RegisterVisualGeometry(ball, Eigen::Translation3d(-radius, 0., 0.),
                                  Sphere(visual_radius), "sphere_x-", purple);

    plant->RegisterVisualGeometry(ball, Eigen::Translation3d(0., radius, 0.),
                                  Sphere(visual_radius), "sphere_y+", purple);
    plant->RegisterVisualGeometry(ball, Eigen::Translation3d(0., -radius, 0.),
                                  Sphere(visual_radius), "sphere_y-", purple);
  }

  // Gravity acting in the -z direction.
  plant->mutable_gravity_field().set_gravity_vector(gravity_W);

  // We are done creating the plant.
  plant->Finalize();

  return plant;
}

}  // namespace bouncing_ball
}  // namespace multibody
}  // namespace examples
}  // namespace drake
