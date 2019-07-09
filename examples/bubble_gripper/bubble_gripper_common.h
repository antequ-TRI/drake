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
		double FLAGS_boxy, FLAGS_boxz;
		std::string FLAGS_integration_scheme;
		double FLAGS_accuracy;
		double FLAGS_target_realtime_rate;
		double FLAGS_simulation_time;
		double FLAGS_y_gravity;
	};

	struct ExtForceFlags
	{
		bool   FLAGS_repeat_force;
		double FLAGS_ext_force_start_time;
		double FLAGS_ext_force_stop_time;
		double FLAGS_ext_force_wx;
		double FLAGS_ext_force_wy;
		double FLAGS_ext_force_wz;
		double FLAGS_ext_force_x;
		double FLAGS_ext_force_y;
		double FLAGS_ext_force_z;
	};

	struct PDControllerParams
	{
		double FLAGS_pd_kbz;
		double FLAGS_pd_kgz;
		double FLAGS_pd_kgw;
		double FLAGS_pd_kbwx;
		double FLAGS_pd_kbz_dot;
		double FLAGS_pd_kgz_dot;
		double FLAGS_pd_kgw_dot;
		double FLAGS_pd_kbwx_dot;
		Eigen::VectorXd desired_x{ 17 };
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
			ExtForceFlags eflags;
			PDControllerParams params;
			auto diagram = make_diagram(lcm, plantp, v0, lqr, flags);
			systems::Context<double>* context;
			init_context_poses(*context, *plantp, v0, flags);
			auto diagram2 = make_diagram_wextforces(lcm, plantp, v0, flags, eflags, false, params);
		}
		DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(BubbleGripperCommon);
		/* this is really bad drake style. but just know these are passed by reference! */
		static std::unique_ptr<systems::Diagram<double>> make_diagram(DrakeLcm& lcm,
	                                    MultibodyPlant<double>*& plant_ptr, double& v0, bool lqr_fixed, const SimFlags& flags);

		static void make_bubbles_mbp_setup(systems::DiagramBuilder<double>& builder, DrakeLcm& lcm, 
        								MultibodyPlant<double>*& plant_ptr, double& v0, bool lqr_fixed, bool actuation_source, const SimFlags& flags);
	    
		static std::unique_ptr<systems::Diagram<double>> make_diagram_wextforces(DrakeLcm& lcm, 
        								MultibodyPlant<double>*& plant_ptr, double& v0, const SimFlags& flags, 
										const ExtForceFlags& force_flags, bool use_controller, const PDControllerParams& c_params );
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

