#include <memory>
#include <string>

#include <gflags/gflags.h>
#include <fstream>
#include <sstream>
#include "fmt/ostream.h"

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/implicit_euler_integrator.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/runge_kutta3_integrator.h"
#include "drake/systems/analysis/semi_explicit_euler_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/sine.h"

namespace drake {
namespace examples {
namespace bubble_gripper {
namespace {

using Eigen::Vector3d;
using geometry::SceneGraph;
using geometry::Sphere;
using lcm::DrakeLcm;
using math::RigidTransformd;
using math::RollPitchYawd;
using multibody::Body;
using multibody::CoulombFriction;
using multibody::ConnectContactResultsToDrakeVisualizer;
using multibody::MultibodyPlant;
using multibody::Parser;
using multibody::PrismaticJoint;
using systems::ImplicitEulerIntegrator;
using systems::RungeKutta2Integrator;
using systems::RungeKutta3Integrator;
using systems::SemiExplicitEulerIntegrator;
using systems::Sine;

// TODO(amcastro-tri): Consider moving this large set of parameters to a
// configuration file (e.g. YAML).
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

DEFINE_double(simulation_time, 10.0,
              "Desired duration of the simulation. [s].");

DEFINE_double(grip_width, 0.14,
              "The initial distance between the gripper fingers. [m].");

// Integration parameters:
DEFINE_string(integration_scheme, "implicit_euler",
              "Integration scheme to be used. Available options are: "
              "'semi_explicit_euler','runge_kutta2','runge_kutta3',"
              "'implicit_euler'");
DEFINE_double(max_time_step, 1.0e-3,
              "Maximum time step used for the integrators. [s]. "
              "If negative, a value based on parameter penetration_allowance "
              "is used.");
DEFINE_double(accuracy, 1.0e-2, "Sets the simulation accuracy for variable step"
              "size integrators with error control.");
DEFINE_bool(time_stepping, true, "If 'true', the plant is modeled as a "
    "discrete system with periodic updates of period 'max_time_step'."
    "If 'false', the plant is modeled as a continuous system.");

// Contact parameters
DEFINE_double(penetration_allowance, 1.0e-2,
              "Penetration allowance. [m]. "
              "See MultibodyPlant::set_penetration_allowance().");
DEFINE_double(v_stiction_tolerance, 1.0e-2,
              "The maximum slipping speed allowed during stiction. [m/s]");

// Pads parameters
DEFINE_int32(ring_samples, 8,
             "The number of spheres used to sample the pad ring");
DEFINE_double(ring_orient, 0, "Rotation of the pads around x-axis. [degrees]");
DEFINE_double(ring_static_friction, 0.3, "The coefficient of static friction "
              "for the ring pad.");
DEFINE_double(ring_dynamic_friction, 0.2, "The coefficient of dynamic friction "
              "for the ring pad.");

// Parameters for rotating the mug.
DEFINE_double(rx, 0, "The x-rotation of the mug around its origin - the center "
              "of its bottom. [degrees]. Extrinsic rotation order: X, Y, Z");
DEFINE_double(ry, 0, "The y-rotation of the mug around its origin - the center "
              "of its bottom. [degrees]. Extrinsic rotation order: X, Y, Z");
DEFINE_double(rz, 0, "The z-rotation of the mug around its origin - the center "
              "of its bottom. [degrees]. Extrinsic rotation order: X, Y, Z");


DEFINE_double(boxz, 0, "The z-coordinate of the box");

// Gripping force.
DEFINE_double(gripper_force, 5, "The force to be applied by the gripper. [N]. "
              "A value of 0 indicates a fixed grip width as set with option "
              "grip_width.");

// Parameters for shaking the mug.
DEFINE_double(amplitude, 0.0, "The amplitude of the harmonic oscillations "
              "carried out by the gripper. [m].");
DEFINE_double(frequency, 2.0, "The frequency of the harmonic oscillations "
              "carried out by the gripper. [Hz].");


// default assumes that timescale is 0.5 s and box char length is 1-2 cm
// must square costs. 
DEFINE_double(state_box_rot_cost, 0.0004,"LQR cost for box rotation." );
DEFINE_double(state_box_transl_cost, 1.0,"LQR cost for box translation." );

DEFINE_double(state_grip_width_cost, 1.0, "LQR cost for gripper width." );
DEFINE_double(state_z_transl_cost, 1.0, "LQR cost for z translation." );

DEFINE_double(state_box_angvel_cost, 0.0016, "LQR cost for box ang velocity." );
DEFINE_double(state_box_vel_cost, 4.0, "LQR cost for box translational velocity." );
DEFINE_double(state_x_vel_cost, 4.0, "LQR cost for gripper width velocity." );
DEFINE_double(state_z_vel_cost, 4.0, "LQR cost for z velocity." );


DEFINE_double(input_z_force_cost, 16.0,"LQR cost for z gripper force." );
DEFINE_double(input_x_force_cost, 16.0,"LQR cost for x gripper clench force." );


// isosphere has vertices that are 0.14628121 units apart so 
const double kSphereScaledRadius = 0.048760403;

std::vector<std::tuple<double, double, double>> read_obj_v(std::string filename)
{
  std::ifstream infile(filename);
  std::vector<std::tuple<double, double, double>> vertices;
  std::string line;
  DRAKE_DEMAND(infile.is_open());
  while (std::getline(infile, line))
  {
    if (line.at(0) == 'v' && line.length() > 2)
    {
      std::istringstream coordinates(line.substr(2));
      double v1, v2, v3;
      coordinates >> v1 >> v2 >> v3;
      vertices.push_back(std::make_tuple(v1,v2,v3));
    }
  }
  infile.close();
  return vertices;


}
// This uses the parameters of the ring to add collision geometries to a
// rigid body for a finger. The collision geometries, consisting of a set of
// small spheres, approximates a torus attached to the finger.
//
// @param[in] plant the MultiBodyPlant in which to add the pads.
// @param[in] pad_offset the ring offset along the x-axis in the finger
// coordinate frame, i.e., how far the ring protrudes from the center of the
// finger.
// @param[in] finger the Body representing the finger
void AddGripperPads(MultibodyPlant<double>* plant,
                    const double bubble_radius, const double x_offset, const Body<double>& bubble,
                    const std::vector<std::tuple<double, double, double>>& vertices, bool incl_left,
                    bool incl_right) {
  const int sample_count = vertices.size();
  

  Vector3d p_FSo;  // Position of the sphere frame S in the finger frame F.
  // The finger frame is defined in simpler_gripper.sdf so that:
  //  - x axis pointing to the right of the gripper.
  //  - y axis pointing forward in the direction of the fingers.
  //  - z axis points up.
  //  - It's origin Fo is right at the geometric center of the finger.
  for (int i = 0; i < sample_count; ++i) {
    const auto& vertex = vertices.at(i);
    double x_coord = std::get<0>(vertex);
    if( (x_coord >= 0 && incl_right) || (x_coord <= 0 && incl_left))
    {

      // The y-offset of the center of the torus in the finger frame F.
      const double torus_center_y_position_F = 0.00;
      p_FSo(0) = std::get<0>(vertex) * bubble_radius + x_offset;
      p_FSo(1) = std::get<1>(vertex) * bubble_radius +
              torus_center_y_position_F;
      p_FSo(2) = std::get<2>(vertex) * bubble_radius;

      // Pose of the sphere frame S in the finger frame F.
      const RigidTransformd X_FS(p_FSo);

      CoulombFriction<double> friction(
          FLAGS_ring_static_friction, FLAGS_ring_static_friction);

      plant->RegisterCollisionGeometry(bubble, X_FS, Sphere(kSphereScaledRadius*bubble_radius),
                                      "collision" + std::to_string(i), friction);

      // don't need fully saturated red
      const Vector4<double> red(0.8, 0.2, 0.2, 1.0);
      plant->RegisterVisualGeometry(bubble, X_FS, Sphere(kSphereScaledRadius*bubble_radius),
                                    "visual" + std::to_string(i), red);
    }
  }
}

int do_main() {
   // TODO Ante : split subroutine up!!

   /* PHASE 1: RUN SIMULATOR TO FIXED POINT */
  systems::DiagramBuilder<double> builder;

  SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
  scene_graph.set_name("scene_graph");

  DRAKE_DEMAND(FLAGS_max_time_step > 0);
  
  MultibodyPlant<double>& plant =
      FLAGS_time_stepping ?
      *builder.AddSystem<MultibodyPlant>(FLAGS_max_time_step) :
      *builder.AddSystem<MultibodyPlant>();
  plant.RegisterAsSourceForSceneGraph(&scene_graph);
  Parser parser(&plant);
  std::string full_name =
      FindResourceOrThrow("drake/examples/bubble_gripper/bubble_gripper.sdf");
  parser.AddModelFromFile(full_name);

  full_name =
      FindResourceOrThrow("drake/examples/bubble_gripper/simple_box.sdf");
  parser.AddModelFromFile(full_name);

  std::string icospherename = FindResourceOrThrow("drake/examples/bubble_gripper/icosphere.obj");
  auto vert = drake::examples::bubble_gripper::read_obj_v(icospherename);
  // Obtain the "translate_joint" axis so that we know the direction of the
  // forced motions. We do not apply gravity if motions are forced in the
  // vertical direction so that the gripper doesn't start free falling. See note
  // below on how we apply these motions. A better strategy would be using
  // constraints but we keep it simple for this demo.
  const PrismaticJoint<double>& translate_joint =
      plant.GetJointByName<PrismaticJoint>("z_translate_joint");
  const Vector3d axis = translate_joint.translation_axis();
  if (axis.isApprox(Vector3d::UnitZ())) {
    fmt::print("Gripper motions forced in the vertical direction.\n");
    plant.mutable_gravity_field().set_gravity_vector(Vector3d::Zero());
  } else if (axis.isApprox(Vector3d::UnitX())) {
    fmt::print("Gripper motions forced in the horizontal direction.\n");
  } else {
    throw std::runtime_error(
        "Only horizontal or vertical motions of the gripper are supported for "
        "this example. The joint axis in the SDF file must either be the "
        "x-axis or the z-axis");
  }

  // Add the pads.
  const Body<double>& left_bubble = plant.GetBodyByName("left_bubble");
  const Body<double>& right_bubble = plant.GetBodyByName("right_bubble");

  // Pads offset from the center of a finger. pad_offset = 0 means the center of
  // the spheres is located right at the center of the finger.
  const double bubble_radius = 0.065+0.001; // this should be gripper radius + 0.0011
  if (FLAGS_gripper_force == 0) {
    // We then fix everything to the right finger and leave the left finger
    // "free" with no applied forces (thus we see it not moving).
    // ANTE TODO: change bubble width to that in the file
    const double bubble_width = 0.007;  // From the visual in the SDF file.
    AddGripperPads(&plant, bubble_radius,0.0 /*xoffset */, right_bubble, vert, 
                    true /* incl_left */, false /* incl_right */);
    AddGripperPads(&plant,
                    bubble_radius, -(FLAGS_grip_width + bubble_width),
                   right_bubble, vert, 
                    false /* incl_left */, true /* incl_right */);
  } else {
    AddGripperPads(&plant, bubble_radius, 0.0 /*xoffset */, right_bubble, vert,
                    true /* incl_left */, false /* incl_right */);
    AddGripperPads(&plant, bubble_radius, 0.0 /*xoffset */, left_bubble, vert,
                    false /* incl_left */, true /* incl_right */);
  }

  // Now the model is complete.
  plant.Finalize();

  // Set how much penetration (in meters) we are willing to accept.
  plant.set_penetration_allowance(FLAGS_penetration_allowance);
  plant.set_stiction_tolerance(FLAGS_v_stiction_tolerance);

  // If the user specifies a time step, we use that, otherwise estimate a
  // maximum time step based on the compliance of the contact model.
  // The maximum time step is estimated to resolve this time scale with at
  // least 30 time steps. Usually this is a good starting point for fixed step
  // size integrators to be stable.
  const double max_time_step =
      FLAGS_max_time_step > 0 ? FLAGS_max_time_step :
      plant.get_contact_penalty_method_time_scale() / 30;

  // Print maximum time step and the time scale introduced by the compliance in
  // the contact model as a reference to the user.
  fmt::print("Maximum time step = {:10.6f} s\n", max_time_step);
  fmt::print("Compliance time scale = {:10.6f} s\n",
             plant.get_contact_penalty_method_time_scale());

  // from simple_griper.sdf, there are two actuators. One actuator on the
  // prismatic joint named "finger_sliding_joint" to actuate the left finger and
  // a second actuator on the prismatic joint named "translate_joint" to impose
  // motions of the gripper.
  DRAKE_DEMAND(plant.num_actuators() == 2);
  DRAKE_DEMAND(plant.num_actuated_dofs() == 2);

  // Sanity check on the availability of the optional source id before using it.
  DRAKE_DEMAND(!!plant.get_source_id());

  builder.Connect(scene_graph.get_query_output_port(),
                  plant.get_geometry_query_input_port());

  DrakeLcm lcm;
  geometry::ConnectDrakeVisualizer(&builder, scene_graph, &lcm);
  builder.Connect(
      plant.get_geometry_poses_output_port(),
      scene_graph.get_source_pose_port(plant.get_source_id().value()));

  // Publish contact results for visualization.
  // (Currently only available when time stepping.)
  if (FLAGS_time_stepping)
    ConnectContactResultsToDrakeVisualizer(&builder, plant, &lcm);

  // Sinusoidal force input. We want the gripper to follow a trajectory of the
  // form x(t) = X0 * sin(ω⋅t). By differentiating once, we can compute the
  // velocity initial condition, and by differentiating twice, we get the input
  // force we need to apply.
  // The mass of the mug is ignored.
  // TODO(amcastro-tri): add a PD controller to precisely control the
  // trajectory of the gripper. Even better, add a motion constraint when MBP
  // supports it.

  // The mass of the gripper in simple_gripper.sdf.
  // TODO(amcastro-tri): we should call MultibodyPlant::CalcMass() here.
  // TODO ANTE: figure out how to set these forces based on new mass
  const double mass = 0.2;  // kg.
  const double omega = 2 * M_PI * FLAGS_frequency;  // rad/s.
  const double x0 = FLAGS_amplitude;  // meters.
  const double v0 = -x0 * omega;  // Velocity amplitude, initial velocity, m/s.
  const double a0 = omega * omega * x0;  // Acceleration amplitude, m/s².
  const double f0 = mass * a0;  // Force amplitude, Newton.
  fmt::print("Acceleration amplitude = {:8.4f} m/s²\n", a0);

  // START WITH a0 = 0 for fixed point simulation.

  // Notice we are using the same Sine source to:
  //   1. Generate a harmonic forcing of the gripper with amplitude f0 and
  //      angular frequency omega.
  //   2. Impose a constant force to the left finger. That is, a harmonic
  //      forcing with "zero" frequency.
  const Vector2<double> amplitudes(0.0, FLAGS_gripper_force);
  const Vector2<double> frequencies(omega, 0.0);
  const Vector2<double> phases(0.0, M_PI_2);
  const auto& harmonic_force = *builder.AddSystem<Sine>(
      amplitudes, frequencies, phases);

  builder.Connect(harmonic_force.get_output_port(0),
                  plant.get_actuation_input_port());

  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram_context->EnableCaching();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // Get joints so that we can set initial conditions.
  const PrismaticJoint<double>& bubble_slider =
      plant.GetJointByName<PrismaticJoint>("bubble_sliding_joint");

  // Set initial position of the left bubble.
  bubble_slider.set_translation(&plant_context, -FLAGS_grip_width);

  // Get mug body so we can set its initial pose.
  const Body<double>& box = plant.GetBodyByName("wooden_box");

  // Initialize the box pose to be right in the middle between the bubble.
  const Vector3d& p_WBr = plant.EvalBodyPoseInWorld(
      plant_context, right_bubble).translation();
  const Vector3d& p_WBl = plant.EvalBodyPoseInWorld(
      plant_context, left_bubble).translation();
  const double box_x_W = (p_WBr(0) + p_WBl(0)) / 2.0;

  RigidTransformd X_WM(
      RollPitchYawd(FLAGS_rx * M_PI / 180, FLAGS_ry * M_PI / 180,
                    (FLAGS_rz * M_PI / 180) + M_PI),
      Vector3d(box_x_W, 0.0 , FLAGS_boxz));
  plant.SetFreeBodyPose(&plant_context, box, X_WM);

  // Set the initial height of the gripper and its initial velocity so that with
  // the applied harmonic forces it continues to move in a harmonic oscillation
  // around this initial position.
  translate_joint.set_translation(&plant_context, 0.0);
  translate_joint.set_translation_rate(&plant_context, v0);

  // Set up simulator.
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  systems::IntegratorBase<double>* integrator{nullptr};

  if (FLAGS_integration_scheme == "implicit_euler") {
    integrator =
        simulator.reset_integrator<ImplicitEulerIntegrator<double>>(
            *diagram, &simulator.get_mutable_context());
  } else if (FLAGS_integration_scheme == "runge_kutta2") {
    integrator =
        simulator.reset_integrator<RungeKutta2Integrator<double>>(
            *diagram, max_time_step, &simulator.get_mutable_context());
  } else if (FLAGS_integration_scheme == "runge_kutta3") {
    integrator =
        simulator.reset_integrator<RungeKutta3Integrator<double>>(
            *diagram, &simulator.get_mutable_context());
  } else if (FLAGS_integration_scheme == "semi_explicit_euler") {
    integrator =
        simulator.reset_integrator<SemiExplicitEulerIntegrator<double>>(
            *diagram, max_time_step, &simulator.get_mutable_context());
  } else {
    throw std::runtime_error(
        "Integration scheme '" + FLAGS_integration_scheme +
            "' not supported for this example.");
  }
  integrator->set_maximum_step_size(max_time_step);
  if (!integrator->get_fixed_step_mode())
    integrator->set_target_accuracy(FLAGS_accuracy);

  // The error controlled integrators might need to take very small time steps
  // to compute a solution to the desired accuracy. Therefore, to visualize
  // these very short transients, we publish every time step.
  simulator.set_publish_every_time_step(true);
  simulator.set_target_realtime_rate(FLAGS_target_realtime_rate);
  simulator.Initialize();
  simulator.AdvanceTo(FLAGS_simulation_time);

  if (FLAGS_time_stepping) {
    fmt::print("Used time stepping with dt={}\n", FLAGS_max_time_step);
    fmt::print("Number of time steps taken = {:d}\n",
               simulator.get_num_steps_taken());
  } else {
    fmt::print("Stats for integrator {}:\n", FLAGS_integration_scheme);
    fmt::print("Number of time steps taken = {:d}\n",
               integrator->get_num_steps_taken());
    if (!integrator->get_fixed_step_mode()) {
      fmt::print("Initial time step taken = {:10.6g} s\n",
                 integrator->get_actual_initial_step_size_taken());
      fmt::print("Largest time step taken = {:10.6g} s\n",
                 integrator->get_largest_step_size_taken());
      fmt::print("Smallest adapted step size = {:10.6g} s\n",
                 integrator->get_smallest_adapted_step_size_taken());
      fmt::print("Number of steps shrunk due to error control = {:d}\n",
                 integrator->get_num_step_shrinkages_from_error_control());
    }
  }
  /* PHASE 2: get state vectors. Set Q and R matrices */
  auto plantState = plant.get_state_output_port().Eval(plant_context);
  drake::VectorX<double> s1rot = box.EvalPoseInWorld(plant_context).rotation().ToQuaternionAsVector4();
  drake::VectorX<double> s2trans = box.EvalPoseInWorld(plant_context).translation();
  double s3gripwidth = bubble_slider.get_translation(plant_context);
  double s4ztrans = translate_joint.get_translation(plant_context);
  /* cost function for LQR state */
  Eigen::MatrixXd Q(17,17);
  Q = Eigen::MatrixXd::Identity(17,17);
  Q(0,0)   = FLAGS_state_box_rot_cost;
  Q(1,1)   = FLAGS_state_box_rot_cost;
  Q(2,2)   = FLAGS_state_box_rot_cost;
  Q(3,3)   = FLAGS_state_box_rot_cost;
  Q(4,4)   = FLAGS_state_box_transl_cost;
  Q(5,5)   = FLAGS_state_box_transl_cost;
  Q(6,6)   = FLAGS_state_box_transl_cost;
  Q(7,7)   = FLAGS_state_z_transl_cost;
  Q(8,8)   = FLAGS_state_grip_width_cost;
  Q(9,9)   = FLAGS_state_box_angvel_cost;
  Q(10,10) = FLAGS_state_box_angvel_cost;
  Q(11,11) = FLAGS_state_box_angvel_cost;
  Q(12,12) = FLAGS_state_box_vel_cost;
  Q(13,13) = FLAGS_state_box_vel_cost;
  Q(14,14) = FLAGS_state_box_vel_cost;
  Q(15,15) = FLAGS_state_z_vel_cost;
  Q(16,16) = FLAGS_state_x_vel_cost;


  /* cost function for LQR input force */
  Eigen::MatrixXd R(2,2);
  R << FLAGS_input_z_force_cost, 0, 0, FLAGS_input_x_force_cost;
  int in_port_index = plant.get_actuation_input_port().get_index();

  systems::DiagramBuilder<double> finalBuilder;
  // can't do this: auto* newplant = finalBuilder.AddSystem(std::move(&plant));
  
  std::cout << "\nBox Rotation: \n" << box.EvalPoseInWorld(plant_context).rotation().ToQuaternionAsVector4() << std::endl;
  std::cout << "\nBox Translation: \n" << box.EvalPoseInWorld(plant_context).translation() << " m" << std::endl;
  std::cout << "\nGripper Width: \n" << bubble_slider.get_translation(plant_context) << " m" << std::endl;
  std::cout << "\nZ Translation: \n" << translate_joint.get_translation(plant_context) << " m" << std::endl;
  
  std::cout << "\nPosition states: " << std::endl<< plant.GetPositions(plant_context) << std::endl;
  
  
  std::cout << "\nBox Ang Vel: \n" << box.EvalSpatialVelocityInWorld(plant_context).rotational() << " Hz" << std::endl;
  std::cout << "\nBox Velocity: \n" << box.EvalSpatialVelocityInWorld(plant_context).translational() << " m/s" << std::endl;
  std::cout << "\nGripper Width Velocity: \n" << bubble_slider.get_translation_rate(plant_context) << " m/s" << std::endl;
  
  std::cout << "\nZ Velocity: \n" << translate_joint.get_translation_rate(plant_context) << " m/s" << std::endl;
  std::cout << "\nVelocity states: " << std::endl<< plant.GetVelocities(plant_context) << std::endl;
  
  std::cout << "\nState vector: \n" <<  plant.GetPositionsAndVelocities(plant_context) << std::endl;
  
  return 0;
}

}  // namespace
}  // namespace bubble_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "Demo used to exercise MultibodyPlant's contact modeling in a gripping "
      "scenario. SceneGraph is used for both visualization and contact "
      "handling. "
      "Launch drake-visualizer before running this example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::HandleSpdlogGflags();
  return drake::examples::bubble_gripper::do_main();
}
