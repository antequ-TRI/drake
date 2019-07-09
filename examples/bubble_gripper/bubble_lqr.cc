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
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/controllers/linear_quadratic_regulator.h"
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
using systems::Sine;

// TODO(amcastro-tri): Consider moving this large set of parameters to a
// configuration file (e.g. YAML).
DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");

DEFINE_double(simulation_time, 0.1,
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

DEFINE_double(fixed_boxy, 0.0, "The y-coordinate of the box when finding the fixed point"); 
DEFINE_double(fixed_boxz, 0, "The z-coordinate of the box when finding the fixed point"); 
DEFINE_double(boxy, 0.0, "The y-coordinate of the box"); 
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
                    bool incl_right, const SimFlags& ) {
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
          FLAGS_static_friction, FLAGS_static_friction);

      plant->RegisterCollisionGeometry(bubble, X_FS, Sphere(kSphereScaledRadius*bubble_radius),
                                      "collision" + bubble.name() + std::to_string(i), friction);

      // don't need fully saturated red
      const Vector4<double> red(0.8, 0.2, 0.2, 1.0);
      plant->RegisterVisualGeometry(bubble, X_FS, Sphere(kSphereScaledRadius*bubble_radius),
                                    "visual" + bubble.name() + std::to_string(i), red);
    }
  }
      plant->set_elastic_modulus(bubble, FLAGS_elastic_modulus);
      plant->set_hydroelastics_dissipation(bubble, FLAGS_dissipation);
}

void AddCollisionGeom(MultibodyPlant<double>* plant, const double bubble_radius, const Body<double>& bubble, const SimFlags& )
{
  CoulombFriction<double> friction(
          FLAGS_static_friction, FLAGS_static_friction);
  plant->RegisterCollisionGeometry(bubble, RigidTransformd(), Sphere(bubble_radius),
                                      "collision" + bubble.name(), friction);
  plant->set_elastic_modulus(bubble, FLAGS_elastic_modulus);
  plant->set_hydroelastics_dissipation(bubble, FLAGS_dissipation);

}
#if 0
/* this is really bad drake style. but just know these are passed by reference! */
std::unique_ptr<systems::Diagram<double>> make_diagram(DrakeLcm& lcm, MultibodyPlant<double>*& plant_ptr, double& v0, bool lqr) 
{
    systems::DiagramBuilder<double> builder;
    SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
    scene_graph.set_name("scene_graph");

    DRAKE_DEMAND(FLAGS_max_time_step > 0);
    plant_ptr = FLAGS_time_stepping ?
        builder.AddSystem<MultibodyPlant>(FLAGS_max_time_step) :
        builder.AddSystem<MultibodyPlant>();
    MultibodyPlant<double>& plant = *plant_ptr;

    if (FLAGS_contact_model == "hydroelastic" ) {
      plant.use_hydroelastic_model(true);    
    } else if (FLAGS_contact_model == "point" || FLAGS_contact_model == "pads") {
        plant.use_hydroelastic_model(false);
    } else {
      throw std::runtime_error("Invalid contact model: '" + FLAGS_contact_model +
                              "'.");
    }
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
    
    if (FLAGS_gripper_force == 0) 
    {
      throw std::runtime_error("Gripper force must be nonzero!");
      #ifdef POINT_CONTACT
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
      #endif
    }
    else 
    {
      if(FLAGS_contact_model == "pads")
      {

        AddGripperPads(&plant, bubble_radius - 0.001, 0.0 /*xoffset */, right_bubble, vert,
                        true /* incl_left */, false /* incl_right */);
        AddGripperPads(&plant, bubble_radius- 0.001, 0.0 /*xoffset */, left_bubble, vert,
                        false /* incl_left */, true /* incl_right */);
      }
      else
      {
        AddCollisionGeom(&plant, bubble_radius, right_bubble);
        AddCollisionGeom(&plant, bubble_radius, left_bubble);
        
      }
    }
    
  
    // Now the model is complete.
    plant.Finalize();

    // Set how much penetration (in meters) we are willing to accept.
    plant.set_penetration_allowance(FLAGS_penetration_allowance);
    plant.set_stiction_tolerance(FLAGS_v_stiction_tolerance);

    // from bubble_gripper.sdf, there are two actuators. One actuator on the
    // prismatic joint named "bubble_sliding_joint" to actuate the left finger and
    // a second actuator on the prismatic joint named "z_translate_joint" to impose
    // motions of the gripper.
    DRAKE_DEMAND(plant.num_actuators() == 2);
    DRAKE_DEMAND(plant.num_actuated_dofs() == 2);

    // Sanity check on the availability of the optional source id before using it.
    DRAKE_DEMAND(!!plant.get_source_id());

    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port());


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
    const double mass = 0.6;  // kg.
    const double omega = 2 * M_PI * FLAGS_frequency;  // rad/s.
    const double x0 = lqr ? 0.0 : FLAGS_amplitude ;  // meters.
    v0 = -x0 * omega;  // Velocity amplitude, initial velocity, m/s.
    const double a0 = omega * omega * x0;  // Acceleration amplitude, m/s².
    const double f0 = mass * a0;  // Force amplitude, Newton.
    fmt::print("Acceleration amplitude = {:8.4f} m/s²\n", a0);

    // START WITH a0 = 0 for fixed point simulation.

    // Notice we are using the same Sine source to:
    //   1. Generate a harmonic forcing of the gripper with amplitude f0 and
    //      angular frequency omega.
    //   2. Impose a constant force to the left finger. That is, a harmonic
    //      forcing with "zero" frequency.
    const Vector2<double> amplitudes(f0, FLAGS_gripper_force);
    const Vector2<double> frequencies(omega, 0.0);
    const Vector2<double> phases(0.0, M_PI_2);
    const auto& harmonic_force = *builder.AddSystem<Sine>(
        amplitudes, frequencies, phases);

    builder.Connect(harmonic_force.get_output_port(0),
                    plant.get_actuation_input_port());
    return builder.Build();
}

void init_context_poses(systems::Context<double>& plant_context, MultibodyPlant<double>& plant, double v0)
{

    // Get joints so that we can set initial conditions.
    const PrismaticJoint<double>& bubble_slider =
        plant.GetJointByName<PrismaticJoint>("bubble_sliding_joint");
    const PrismaticJoint<double>& translate_joint =
        plant.GetJointByName<PrismaticJoint>("z_translate_joint");


    // Get box body so we can set its initial pose.
    const Body<double>& box = plant.GetBodyByName("wooden_box");


    const Body<double>& left_bubble = plant.GetBodyByName("left_bubble");
    const Body<double>& right_bubble = plant.GetBodyByName("right_bubble");


    // Set initial position of the left bubble.
    bubble_slider.set_translation(&plant_context, -FLAGS_grip_width);
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
}

#endif


int do_main() {
   // TODO Ante : split subroutine up!!
  drake::VectorX<double> plant_state(17);
  drake::VectorX<double> plant_input(2);
  int in_port_index;
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
    flags.FLAGS_boxy = FLAGS_fixed_boxy;
		flags.FLAGS_boxz = FLAGS_fixed_boxz;
    flags.FLAGS_integration_scheme = FLAGS_integration_scheme;
    flags.FLAGS_accuracy = FLAGS_accuracy;
    flags.FLAGS_target_realtime_rate = FLAGS_target_realtime_rate;
    flags.FLAGS_simulation_time = FLAGS_simulation_time;

    DrakeLcm lcm;
    MultibodyPlant<double>* plant_ptr = nullptr;
    double v0;
    auto diagram = BubbleGripperCommon::make_diagram(lcm, plant_ptr, v0, true /* lqr fixed point */, flags);
    MultibodyPlant<double>& plant = *plant_ptr;
    DRAKE_DEMAND(plant.is_discrete());
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
    //std::cout << "\nStates from the plant subcontext:\n" << plantState << std::endl;
    //std::cout << "\nDisc states from the plant subcontext:\n" << plant_context.get_mutable_discrete_state_vector().get_mutable_value() << std::endl;
    std::cout << "\nNumber of input ports: " << plant.num_input_ports() << std::endl;
    std::cout << "\nActuation input port index: " << plant.get_actuation_input_port().get_index() << std::endl;
    std::cout << "\nActuation input vector: \n" << plant.get_actuation_input_port().Eval(plant_context) << std::endl;
    /* PHASE 2: get state vectors. Set Q and R matrices */
    plant_state = plant.get_state_output_port().Eval(plant_context);
    plant_input = plant.get_actuation_input_port().Eval(plant_context);
    in_port_index = plant.get_actuation_input_port().get_index();

    /* cost function for LQR state */

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

    R << FLAGS_input_z_force_cost, 0, 0, FLAGS_input_x_force_cost;
    //plant.get_actuation_input_port().FixValue(&plant_context, plant_input);
    /* phase 3 - make controller */
    //controller = systems::controllers::LinearQuadraticRegulator( plant, plant_context, Q, R, Eigen::Matrix<double,0,0>::Zero() /* N */, in_port_index);
    //controller->set_name("LQR controller");
  }

  // set up new system
  systems::DiagramBuilder<double> builder;
  DrakeLcm lcm;
  MultibodyPlant<double>* plant_ptr = nullptr;
  double v0;
  flags.FLAGS_boxy = FLAGS_boxy;
  flags.FLAGS_boxz = FLAGS_boxz;
  {
   
      unused(v0);
      unused(in_port_index);
      unused(lcm);
      if(false)
      {
        DRAKE_UNREACHABLE();
        std::string icospherename = FindResourceOrThrow("drake/examples/bubble_gripper/icosphere.obj");
        auto vert = drake::examples::bubble_gripper::read_obj_v(icospherename);
        AddGripperPads(plant_ptr, 0.0, 0.0, *(static_cast<multibody::Body<double>*>(nullptr)),vert, true, false, flags);
        AddCollisionGeom(plant_ptr, 0.0, *(static_cast<multibody::Body<double>*>(nullptr)) ,flags);
      }
      SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
      scene_graph.set_name("scene_graph");

      // Load and parse double pendulum SDF from file into a tree.
      DRAKE_DEMAND(flags.FLAGS_max_time_step > 0);
      DRAKE_DEMAND(flags.FLAGS_time_stepping);
      plant_ptr = flags.FLAGS_time_stepping ?
          builder.AddSystem<MultibodyPlant>(flags.FLAGS_max_time_step) :
          builder.AddSystem<MultibodyPlant>();
      MultibodyPlant<double>& plant = *plant_ptr;
      plant.set_name("bubble_gripper");
      plant.RegisterAsSourceForSceneGraph(&scene_graph);
      /* hydroelastics */
      {
        if (flags.FLAGS_contact_model == "hydroelastic" ) {
          plant.use_hydroelastic_model(true);    
        } else if (flags.FLAGS_contact_model == "point" || flags.FLAGS_contact_model == "pads") {
            plant.use_hydroelastic_model(false);
        } else {
          throw std::runtime_error("Invalid contact model: '" + flags.FLAGS_contact_model +
                                  "'.");
        }
      }
      Parser parser(plant_ptr);
      std::string full_name =
          FindResourceOrThrow("drake/examples/bubble_gripper/bubble_gripper.sdf");
      parser.AddModelFromFile(full_name);
      
      full_name =
          FindResourceOrThrow("drake/examples/bubble_gripper/simple_box.sdf");
      parser.AddModelFromFile(full_name);
      /* todo: instance index stuff? */
      
      /* turn off gravity */
      {
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
      }

      // Add the bubble collision geometry TODO: consider removing
      //if(false)
      {
          const Body<double>& left_bubble = plant.GetBodyByName("left_bubble");
          const Body<double>& right_bubble = plant.GetBodyByName("right_bubble");
          const double bubble_radius = 0.065+0.001;
          if(flags.FLAGS_contact_model == "pads")
          {
            std::string icospherename = FindResourceOrThrow("drake/examples/bubble_gripper/icosphere.obj");
            auto vert = drake::examples::bubble_gripper::read_obj_v(icospherename);
            AddGripperPads(&plant, bubble_radius - 0.001, 0.0 /*xoffset */, right_bubble, vert,
                            true /* incl_left */, false /* incl_right */, flags);
            AddGripperPads(&plant, bubble_radius- 0.001, 0.0 /*xoffset */, left_bubble, vert,
                            false /* incl_left */, true /* incl_right */, flags);
          }
          else
          {
            AddCollisionGeom(&plant, bubble_radius, right_bubble, flags);
            AddCollisionGeom(&plant, bubble_radius, left_bubble, flags);
            
          }

      }
      plant.Finalize();

      // Set how much penetration (in meters) we are willing to accept.
      plant.set_penetration_allowance(flags.FLAGS_penetration_allowance);
      plant.set_stiction_tolerance(flags.FLAGS_v_stiction_tolerance);

     
      // verification
      {
          // from bubble_gripper.sdf, there are two actuators. One actuator on the
          // prismatic joint named "bubble_sliding_joint" to actuate the left finger and
          // a second actuator on the prismatic joint named "z_translate_joint" to impose
          // motions of the gripper.
          DRAKE_DEMAND(plant.num_actuators() == 2);
          DRAKE_DEMAND(plant.num_actuated_dofs() == 2);
          DRAKE_DEMAND(!!plant.get_source_id());
      }
      builder.Connect(plant.get_geometry_poses_output_port(),
          scene_graph.get_source_pose_port(plant.get_source_id().value()));
      builder.Connect(scene_graph.get_query_output_port(), 
          plant.get_geometry_query_input_port());
      geometry::ConnectDrakeVisualizer(&builder, scene_graph);

      builder.ExportInput(plant.get_actuation_input_port());
      builder.ExportOutput(plant.get_state_output_port());


  }
  
  systems::DiagramBuilder<double> finalBuilder;
  auto subdiagram = finalBuilder.AddSystem<systems::Diagram<double>>(builder.Build());

      int plant_actuation_port = plant_ptr->get_actuation_input_port().get_index();
 // Create and initialize the  context for this system
  std::unique_ptr<systems::Context<double>> subdiagram_context =
      subdiagram->CreateDefaultContext();
  MultibodyPlant<double>& plant = *plant_ptr;
  systems::Context<double>& plant_context =
      subdiagram->GetMutableSubsystemContext(plant, subdiagram_context.get());
      unused(plant_actuation_port);
      // set nominal torque
      subdiagram_context->FixInputPort(0, plant_input);
      //plant_context.FixInputPort(plant_actuation_port, plant_input);
      // set nominal state
      if( plant.is_discrete() )
      {
        plant_context.SetDiscreteState(plant_state);
      }
      else
      {
        throw new std::runtime_error("continuous LQR not supported yet!");
      }
       
      std::cout << "Constructing LQR ..." << std::endl;
      auto lqr = finalBuilder.AddSystem(systems::controllers::LinearQuadraticRegulator(
          *subdiagram, *subdiagram_context, Q, R));
      std::cout << "LQR constructed!" << std::endl;
      finalBuilder.Connect(plant.get_state_output_port(), lqr->get_input_port());
      finalBuilder.Connect(lqr->get_output_port(), plant.get_actuation_input_port());

  auto diagram = finalBuilder.Build();
  std::unique_ptr<systems::Context<double>> diagram_context = diagram->CreateDefaultContext();
  diagram_context->EnableCaching();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& final_subdiagram_context =
      diagram->GetMutableSubsystemContext(*subdiagram, diagram_context.get());
  systems::Context<double>& final_plant_context =
      subdiagram->GetMutableSubsystemContext(plant, &final_subdiagram_context);
      if( plant.is_discrete() )
      {
        final_plant_context.SetDiscreteState(plant_state);
      }
      else
      {
        final_plant_context.SetContinuousState(plant_state);
      }


  // Set up simulator.
  systems::Simulator<double> simulator(*diagram, std::move(diagram_context));
  

  BubbleGripperCommon::simulate_bubbles(simulator, plant, diagram.get(), flags );
  BubbleGripperCommon::print_states(plant, final_plant_context, flags);

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
