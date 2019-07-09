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
#include "drake/examples/bubble_gripper/bubble_gripper_common.h"

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
DEFINE_double(init_target_realtime_rate, 0.0,
              "Desired rate relative to real time, for initialization sim.");

DEFINE_double(init_simulation_time, 1.0,
              "Desired duration of the initialization simulation. [s].");

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
DEFINE_double(static_friction, 1.0, "The coefficient of static friction");

// Parameters for rotating the mug.
DEFINE_double(rx, 0, "The x-rotation of the mug around its origin - the center "
              "of its bottom. [degrees]. Extrinsic rotation order: X, Y, Z");
DEFINE_double(ry, 0, "The y-rotation of the mug around its origin - the center "
              "of its bottom. [degrees]. Extrinsic rotation order: X, Y, Z");
DEFINE_double(rz, 0, "The z-rotation of the mug around its origin - the center "
              "of its bottom. [degrees]. Extrinsic rotation order: X, Y, Z");


DEFINE_double(fixed_boxz, 0, "The z-coordinate of the box when finding the fixed point"); 
DEFINE_double(boxz, 0, "The z-coordinate of the box in simulation"); 
/* use -0.08 with pads */
/* use -0.12 otherwise */

// Gripping force.
DEFINE_double(gripper_force, 10, "The force to be applied by the gripper. [N]. "
              "A value of 0 indicates a fixed grip width as set with option "
              "grip_width.");

// Parameters for shaking the mug.
DEFINE_double(amplitude, 0, "The amplitude of the harmonic oscillations "
              "carried out by the gripper. [m].");
DEFINE_double(frequency, 2.0, "The frequency of the harmonic oscillations "
              "carried out by the gripper. [Hz].");

DEFINE_string(contact_model, "point",
              "Contact model. Options are: 'point', 'hydroelastic', 'pads'.");
DEFINE_double(elastic_modulus, 5.0e4, "Desired Accuracy. ");
DEFINE_double(dissipation, 5.0, "Desired Accuracy. ");
// isosphere has vertices that are 0.14628121 units apart so 
// const double kSphereScaledRadius = 0.048760403;

/* external force */
DEFINE_bool(repeat_force, false, "Whether to repeatedly apply the external force.");
DEFINE_double(ext_force_start_time, 1.2, "External force start time. ");
DEFINE_double(ext_force_stop_time, 1.5, "External force stop time. ");
DEFINE_double(ext_force_wx, 0.0, "External torque x. ");
DEFINE_double(ext_force_wy, 0.0, "External torque y. ");
DEFINE_double(ext_force_wz, 0.0, "External torque z. ");
DEFINE_double(ext_force_x,  0.0, "External force  x. ");
DEFINE_double(ext_force_y,  0.0, "External force  y. ");
DEFINE_double(ext_force_z,  0.5, "External force  z. ");

/* controller */
DEFINE_bool(use_controller, false, "Whether to add a controller.");
DEFINE_double(pd_kbz,  100., "Controller k on box z position ");
DEFINE_double(pd_kgz,  100., "Controller k on gripper z position ");
DEFINE_double(pd_kgw,  100., "Controller k on gripper width ");
DEFINE_double(pd_kbz_dot,  25.3, "Controller k on box z velocity ");
DEFINE_double(pd_kgz_dot,  25.3, "Controller k on gripper z velocity ");
DEFINE_double(pd_kgw_dot,  11., "Controller k on gripper width velocity ");
DEFINE_double(pd_k_scale,  1., "Scales all controller position ks.");
DEFINE_double(pd_kdot_scale,  1., "Scales all controller velocity ks.");
DEFINE_double(pd_all_scale,  1., "Scales all controller ks.");


int do_main() {
  drake::VectorX<double> plant_state(17);
  drake::VectorX<double> plant_input(2);
  SimFlags flags;
  //std::unique_ptr<systems::AffineSystem<double>> controller;
  Eigen::MatrixXd Q(17,17);
  Eigen::MatrixXd R(2,2);

   /* PHASE 1: RUN SIMULATOR TO FIXED POINT */
  { 
    flags.FLAGS_static_friction = FLAGS_static_friction;
    flags.FLAGS_elastic_modulus = FLAGS_elastic_modulus;
    flags.FLAGS_dissipation = FLAGS_dissipation;
		flags.FLAGS_time_stepping = FLAGS_time_stepping;
		flags.FLAGS_max_time_step = FLAGS_max_time_step;
		flags.FLAGS_contact_model = FLAGS_contact_model;
		flags.FLAGS_gripper_force = FLAGS_gripper_force;
		flags.FLAGS_penetration_allowance = FLAGS_penetration_allowance;
		flags.FLAGS_v_stiction_tolerance = FLAGS_v_stiction_tolerance;
		flags.FLAGS_frequency = FLAGS_frequency;
		flags.FLAGS_amplitude = FLAGS_amplitude;
		flags.FLAGS_grip_width = FLAGS_grip_width;
		flags.FLAGS_rx = FLAGS_rx; 
    flags.FLAGS_ry = FLAGS_ry; 
    flags.FLAGS_rz = FLAGS_rz;
		flags.FLAGS_boxz = FLAGS_boxz;
    flags.FLAGS_integration_scheme = FLAGS_integration_scheme;
    flags.FLAGS_accuracy = FLAGS_accuracy;
    flags.FLAGS_target_realtime_rate = FLAGS_init_target_realtime_rate;
    flags.FLAGS_simulation_time = FLAGS_init_simulation_time;

    DrakeLcm lcm;
    MultibodyPlant<double>* plant_ptr = nullptr;
    double v0;
    auto diagram = BubbleGripperCommon::make_diagram(lcm, plant_ptr, v0, true /* lqr fixed point */, flags);
    MultibodyPlant<double>& plant = *plant_ptr;
    // Create a context for this system:
    std::unique_ptr<systems::Context<double>> diagram_context =
        diagram->CreateDefaultContext();
    diagram_context->EnableCaching();
    diagram->SetDefaultContext(diagram_context.get());
    systems::Context<double>& plant_context =
        diagram->GetMutableSubsystemContext(plant, diagram_context.get());
    BubbleGripperCommon::init_context_poses(plant_context, plant, v0, flags);

    // Set up simulator.
    systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

    BubbleGripperCommon::simulate_bubbles(simulator, plant, diagram.get(), flags );
    BubbleGripperCommon::print_states(plant, plant_context, flags);
    plant_state = plant.get_state_output_port().Eval(plant_context);
    plant_input = plant.get_actuation_input_port().Eval(plant_context);

  }
  /* Phase 2: Build controller and add external forces */
  {
    std::cout << "Making ext force and controller." << std::endl;
    flags.FLAGS_target_realtime_rate = FLAGS_target_realtime_rate;
    flags.FLAGS_simulation_time = FLAGS_simulation_time;
    ExtForceFlags force_flags;
    force_flags.FLAGS_repeat_force = FLAGS_repeat_force;
    force_flags.FLAGS_ext_force_start_time = FLAGS_ext_force_start_time;
    force_flags.FLAGS_ext_force_stop_time = FLAGS_ext_force_stop_time;
    force_flags.FLAGS_ext_force_wx = FLAGS_ext_force_wx;
    force_flags.FLAGS_ext_force_wy = FLAGS_ext_force_wy;
    force_flags.FLAGS_ext_force_wz = FLAGS_ext_force_wz;
    force_flags.FLAGS_ext_force_x = FLAGS_ext_force_x;
    force_flags.FLAGS_ext_force_y = FLAGS_ext_force_y;
    force_flags.FLAGS_ext_force_z = FLAGS_ext_force_z;
    PDControllerParams params;
    params.FLAGS_pd_kbz = FLAGS_pd_kbz * FLAGS_pd_kdot_scale * FLAGS_pd_all_scale;
    params.FLAGS_pd_kgz = FLAGS_pd_kgz * FLAGS_pd_kdot_scale * FLAGS_pd_all_scale;
    params.FLAGS_pd_kgw = FLAGS_pd_kgw * FLAGS_pd_kdot_scale * FLAGS_pd_all_scale;
    params.FLAGS_pd_kbz_dot = FLAGS_pd_kbz_dot * FLAGS_pd_kdot_scale * FLAGS_pd_all_scale;
    params.FLAGS_pd_kgz_dot = FLAGS_pd_kgz_dot * FLAGS_pd_kdot_scale * FLAGS_pd_all_scale;
    params.FLAGS_pd_kgw_dot = FLAGS_pd_kgw_dot * FLAGS_pd_kdot_scale * FLAGS_pd_all_scale;
    params.desired_x = plant_state;
    DrakeLcm lcm;
    MultibodyPlant<double>* plant_ptr = nullptr;
    double v0;
    auto diagram = BubbleGripperCommon::make_diagram_wextforces(lcm, plant_ptr, v0, flags, force_flags, FLAGS_use_controller, params);
    MultibodyPlant<double>& plant = *plant_ptr;
    std::unique_ptr<systems::Context<double>> diagram_context =
        diagram->CreateDefaultContext();
    diagram_context->EnableCaching();
    diagram->SetDefaultContext(diagram_context.get());
    systems::Context<double>& plant_context =
        diagram->GetMutableSubsystemContext(plant, diagram_context.get());
    BubbleGripperCommon::init_context_poses(plant_context, plant, v0, flags);
    if (plant_ptr->is_discrete())
    {
      plant_context.get_mutable_discrete_state_vector().SetFromVector(plant_state);
    }
    else
    {
      plant_context.get_mutable_continuous_state_vector().SetFromVector(plant_state);
    }
    systems::Simulator<double> simulator(*diagram, std::move(diagram_context));

    BubbleGripperCommon::simulate_bubbles(simulator, plant, diagram.get(), flags );
    //BubbleGripperCommon::print_states(plant, plant_context, flags);


    
    
  }


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
