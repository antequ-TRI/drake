#include <memory>
#include <string>

#include <fstream>
#include <sstream>
#include "fmt/ostream.h"

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
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
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/primitives/sine.h"
#include "drake/examples/bubble_gripper/bubble_gripper_common.h"
#include "drake/systems/analysis/implicit_euler_integrator.h"
#include "drake/systems/analysis/runge_kutta2_integrator.h"
#include "drake/systems/analysis/runge_kutta3_integrator.h"
#include "drake/systems/analysis/semi_explicit_euler_integrator.h"

namespace drake {
namespace examples {
namespace bubble_gripper {
//namespace {

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
using systems::ImplicitEulerIntegrator;
using systems::RungeKutta2Integrator;
using systems::RungeKutta3Integrator;
using systems::SemiExplicitEulerIntegrator;



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
                    bool incl_right, const SimFlags& flags) {
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
          flags.FLAGS_static_friction, flags.FLAGS_static_friction);

      plant->RegisterCollisionGeometry(bubble, X_FS, Sphere(kSphereScaledRadius*bubble_radius),
                                      "collision" + bubble.name() + std::to_string(i), friction);

      // don't need fully saturated red
      const Vector4<double> red(0.8, 0.2, 0.2, 1.0);
      plant->RegisterVisualGeometry(bubble, X_FS, Sphere(kSphereScaledRadius*bubble_radius),
                                    "visual" + bubble.name() + std::to_string(i), red);
    }
  }
      plant->set_elastic_modulus(bubble, flags.FLAGS_elastic_modulus);
      plant->set_hydroelastics_dissipation(bubble, flags.FLAGS_dissipation);
}

void AddCollisionGeom(MultibodyPlant<double>* plant, const double bubble_radius,
                      const Body<double>& bubble, const SimFlags& flags)
{
  CoulombFriction<double> friction(
          flags.FLAGS_static_friction, flags.FLAGS_static_friction);
  plant->RegisterCollisionGeometry(bubble, RigidTransformd(), Sphere(bubble_radius),
                                      "collision" + bubble.name(), friction);
  plant->set_elastic_modulus(bubble, flags.FLAGS_elastic_modulus);
  plant->set_hydroelastics_dissipation(bubble, flags.FLAGS_dissipation);

}

void BubbleGripperCommon::make_bubbles_mbp_setup(systems::DiagramBuilder<double>& builder, DrakeLcm& lcm, 
        MultibodyPlant<double>*& plant_ptr, double& v0, bool lqr_fixed, const SimFlags& flags)
        //SceneGraph<double>*& scene_graph_ptr, ) 
{
    SceneGraph<double>& scene_graph = *builder.AddSystem<SceneGraph>();
    scene_graph.set_name("scene_graph");

    DRAKE_DEMAND(flags.FLAGS_max_time_step > 0);
    plant_ptr = flags.FLAGS_time_stepping ?
        builder.AddSystem<MultibodyPlant>(flags.FLAGS_max_time_step) :
        builder.AddSystem<MultibodyPlant>();
    MultibodyPlant<double>& plant = *plant_ptr;

    if (flags.FLAGS_contact_model == "hydroelastic" ) {
      plant.use_hydroelastic_model(true);    
    } else if (flags.FLAGS_contact_model == "point" || flags.FLAGS_contact_model == "pads") {
        plant.use_hydroelastic_model(false);
    } else {
      throw std::runtime_error("Invalid contact model: '" + flags.FLAGS_contact_model +
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
    
    if (flags.FLAGS_gripper_force == 0) 
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
    
  
    // Now the model is complete.
    plant.Finalize();

    // Set how much penetration (in meters) we are willing to accept.
    plant.set_penetration_allowance(flags.FLAGS_penetration_allowance);
    plant.set_stiction_tolerance(flags.FLAGS_v_stiction_tolerance);

    // from bubble_gripper.sdf, there are two actuators. One actuator on the
    // prismatic joint named "bubble_sliding_joint" to actuate the left finger and
    // a second actuator on the prismatic joint named "z_translate_joint" to impose
    // motions of the gripper.
    DRAKE_DEMAND(plant.num_actuators() == 2);
    DRAKE_DEMAND(plant.num_actuated_dofs() == 2);

    // Sanity check on the availability of the optional source id before using it.
    DRAKE_DEMAND(!!plant.get_source_id());



    geometry::ConnectDrakeVisualizer(&builder, scene_graph, &lcm);
    builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id().value()));

    // Publish contact results for visualization.
    // (Currently only available when time stepping.)
    if (flags.FLAGS_time_stepping)
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
    const double omega = 2 * M_PI * flags.FLAGS_frequency;  // rad/s.
    const double x0 = lqr_fixed ? 0.0 : flags.FLAGS_amplitude ;  // meters.
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
    const Vector2<double> amplitudes(f0, flags.FLAGS_gripper_force);
    const Vector2<double> frequencies(omega, 0.0);
    const Vector2<double> phases(0.0, M_PI_2);
    const auto& harmonic_force = *builder.AddSystem<Sine>(
        amplitudes, frequencies, phases);

    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port());
    builder.Connect(harmonic_force.get_output_port(0),
                    plant.get_actuation_input_port());
}
/* this is really bad drake style. but just know these are passed by reference! */
std::unique_ptr<systems::Diagram<double>> BubbleGripperCommon::make_diagram(DrakeLcm& lcm, 
        MultibodyPlant<double>*& plant_ptr, double& v0, bool lqr_fixed, const SimFlags& flags) 
{
    systems::DiagramBuilder<double> builder;
    BubbleGripperCommon::make_bubbles_mbp_setup(builder, lcm, plant_ptr, v0, lqr_fixed, flags);
    return builder.Build();
}

void BubbleGripperCommon::init_context_poses(systems::Context<double>& plant_context,
    MultibodyPlant<double>& plant, double v0, const SimFlags& flags)
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
    bubble_slider.set_translation(&plant_context, -flags.FLAGS_grip_width);
    // Initialize the box pose to be right in the middle between the bubble.
    const Vector3d& p_WBr = plant.EvalBodyPoseInWorld(
        plant_context, right_bubble).translation();
    const Vector3d& p_WBl = plant.EvalBodyPoseInWorld(
        plant_context, left_bubble).translation();
    const double box_x_W = (p_WBr(0) + p_WBl(0)) / 2.0;

    RigidTransformd X_WM(
        RollPitchYawd(flags.FLAGS_rx * M_PI / 180, flags.FLAGS_ry * M_PI / 180,
                      (flags.FLAGS_rz * M_PI / 180) + M_PI),
        Vector3d(box_x_W, 0.0 , flags.FLAGS_boxz));
    plant.SetFreeBodyPose(&plant_context, box, X_WM);

    // Set the initial height of the gripper and its initial velocity so that with
    // the applied harmonic forces it continues to move in a harmonic oscillation
    // around this initial position.
    translate_joint.set_translation(&plant_context, 0.0);
    translate_joint.set_translation_rate(&plant_context, v0);
}


void BubbleGripperCommon::simulate_bubbles(systems::Simulator<double>& simulator, const MultibodyPlant<double>& plant,
                      systems::Diagram<double>* diagram, const SimFlags& flags)
{


    // If the user specifies a time step, we use that, otherwise estimate a
    // maximum time step based on the compliance of the contact model.
    // The maximum time step is estimated to resolve this time scale with at
    // least 30 time steps. Usually this is a good starting point for fixed step
    // size integrators to be stable.
    const double max_time_step =
        flags.FLAGS_max_time_step > 0 ? flags.FLAGS_max_time_step :
        plant.get_contact_penalty_method_time_scale() / 30;

    // Print maximum time step and the time scale introduced by the compliance in
    // the contact model as a reference to the user.
    fmt::print("Maximum time step = {:10.6f} s\n", max_time_step);
    fmt::print("Compliance time scale = {:10.6f} s\n",
              plant.get_contact_penalty_method_time_scale());


    systems::IntegratorBase<double>* integrator{nullptr};

    if (flags.FLAGS_integration_scheme == "implicit_euler") {
      integrator =
          simulator.reset_integrator<ImplicitEulerIntegrator<double>>(
              *diagram, &simulator.get_mutable_context());
    } else if (flags.FLAGS_integration_scheme == "runge_kutta2") {
      integrator =
          simulator.reset_integrator<RungeKutta2Integrator<double>>(
              *diagram, max_time_step, &simulator.get_mutable_context());
    } else if (flags.FLAGS_integration_scheme == "runge_kutta3") {
      integrator =
          simulator.reset_integrator<RungeKutta3Integrator<double>>(
              *diagram, &simulator.get_mutable_context());
    } else if (flags.FLAGS_integration_scheme == "semi_explicit_euler") {
      integrator =
          simulator.reset_integrator<SemiExplicitEulerIntegrator<double>>(
              *diagram, max_time_step, &simulator.get_mutable_context());
    } else {
      throw std::runtime_error(
          "Integration scheme '" + flags.FLAGS_integration_scheme +
              "' not supported for this example.");
    }
    integrator->set_maximum_step_size(max_time_step);
    if (!integrator->get_fixed_step_mode())
      integrator->set_target_accuracy(flags.FLAGS_accuracy);

    // The error controlled integrators might need to take very small time steps
    // to compute a solution to the desired accuracy. Therefore, to visualize
    // these very short transients, we publish every time step.
    simulator.set_publish_every_time_step(true);
    simulator.set_target_realtime_rate(flags.FLAGS_target_realtime_rate);
    simulator.Initialize();
    simulator.AdvanceTo(flags.FLAGS_simulation_time);

    if (flags.FLAGS_time_stepping) {
      fmt::print("Used time stepping with dt={}\n", flags.FLAGS_max_time_step);
      fmt::print("Number of time steps taken = {:d}\n",
                simulator.get_num_steps_taken());
    }
    else
    {
      fmt::print("Stats for integrator {}:\n", flags.FLAGS_integration_scheme);
      fmt::print("Number of time steps taken = {:d}\n",
                integrator->get_num_steps_taken());
      if (!integrator->get_fixed_step_mode())
      {
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
}

void BubbleGripperCommon::print_states(const MultibodyPlant<double>& plant, const systems::Context<double>& plant_context, const SimFlags& )
{

    // Get joints so that we can get poses.
    const PrismaticJoint<double>& bubble_slider =
        plant.GetJointByName<PrismaticJoint>("bubble_sliding_joint");
    const PrismaticJoint<double>& translate_joint =
        plant.GetJointByName<PrismaticJoint>("z_translate_joint");


    // Get box body so that we can get poses.
    const Body<double>& box = plant.GetBodyByName("wooden_box");
    std::cout << "\nBox Rotation: \n" << box.EvalPoseInWorld(plant_context).rotation().ToQuaternionAsVector4() << std::endl;
    std::cout << "\nBox Translation: \n" << box.EvalPoseInWorld(plant_context).translation() << " m" << std::endl;
    std::cout << "\nZ Translation: \n" << translate_joint.get_translation(plant_context) << " m" << std::endl;
    std::cout << "\nGripper Width: \n" << bubble_slider.get_translation(plant_context) << " m" << std::endl;
    
    
    std::cout << "\nPosition states: " << std::endl<< plant.GetPositions(plant_context) << std::endl;
    
    
    std::cout << "\nBox Ang Vel: \n" << box.EvalSpatialVelocityInWorld(plant_context).rotational() << " Hz" << std::endl;
    std::cout << "\nBox Velocity: \n" << box.EvalSpatialVelocityInWorld(plant_context).translational() << " m/s" << std::endl;
    std::cout << "\nZ Velocity: \n" << translate_joint.get_translation_rate(plant_context) << " m/s" << std::endl;
    std::cout << "\nGripper Width Velocity: \n" << bubble_slider.get_translation_rate(plant_context) << " m/s" << std::endl;
    
    
    std::cout << "\nVelocity states: " << std::endl<< plant.GetVelocities(plant_context) << std::endl;
    
    std::cout << "\nState vector: \n" <<  plant.GetPositionsAndVelocities(plant_context) << std::endl;
    
}

//}  // namespace
}  // namespace bubble_gripper
}  // namespace examples
}  // namespace drake
