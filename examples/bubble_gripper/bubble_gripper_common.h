#pragma once

#include <memory>
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/analysis/simulator.h"

namespace drake {
namespace examples {
namespace bubble_gripper {
//namespace {
	using lcm::DrakeLcm;
	using multibody::MultibodyPlant;

	struct SimFlags
	{
		double FLAGS_static_friction, FLAGS_elastic_modulus, FLAGS_dissipation;
		bool FLAGS_time_stepping;
		double FLAGS_max_time_step;
		std::string FLAGS_contact_model;
		double FLAGS_gripper_force;
		double FLAGS_penetration_allowance;
		double FLAGS_v_stiction_tolerance;
		double FLAGS_frequency;
		double FLAGS_amplitude;
		double FLAGS_grip_width;
		double FLAGS_rx, FLAGS_ry, FLAGS_rz;
		double FLAGS_boxz;
		std::string FLAGS_integration_scheme;
		double FLAGS_accuracy;
		double FLAGS_target_realtime_rate;
		double FLAGS_simulation_time;
	};

    class BubbleGripperCommon final 
	{
		public:
		BubbleGripperCommon()
		{
			DRAKE_UNREACHABLE();
			DrakeLcm lcm;
			MultibodyPlant<double>* plantp;
			double v0;
			bool lqr;
			SimFlags flags;
			auto diagram = make_diagram(lcm, plantp, v0, lqr, flags);
			systems::Context<double>* context;
			init_context_poses(*context, *plantp, v0, flags);
		}
		DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BubbleGripperCommon);
		/* this is really bad drake style. but just know these are passed by reference! */
		static std::unique_ptr<systems::Diagram<double>> make_diagram(DrakeLcm& lcm,
	                                    MultibodyPlant<double>*& plant_ptr, double& v0, bool lqr_fixed, const SimFlags& flags);

		static void make_bubbles_mbp_setup(systems::DiagramBuilder<double>& builder, DrakeLcm& lcm, 
        								MultibodyPlant<double>*& plant_ptr, double& v0, bool lqr_fixed, const SimFlags& flags);
		static void init_context_poses(systems::Context<double>& plant_context, 
	                                    MultibodyPlant<double>& plant, double v0, const SimFlags& flags);
		static void simulate_bubbles(systems::Simulator<double>& simulator, const MultibodyPlant<double>& plant,
                      systems::Diagram<double>* diagram, const SimFlags& flags);
		static void print_states(const MultibodyPlant<double>& plant, const systems::Context<double>& plant_context, const SimFlags& );
		
    };


//} // namespace 
} // namespace bubble_gripper
} // namespace examples
} // namespace drake

