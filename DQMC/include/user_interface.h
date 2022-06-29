#pragma once
#ifndef USER_INTERFACE_H
#define USER_INTERFACE_H


//#include "../include/plog/Log.h"
//#include "../include/plog/Initializers/RollingFileInitializer.h"
#include "../src/UserInterface/ui.h"
#include "../include/hubbard_dqmc_qr.h"


// -------------------------------------------------------- HUBBARD USER INTERFACE --------------------------------------------------------

namespace hubbard {
	// -------------------------------------------------------- MAP OF DEFAULTS FOR HUBBARD


	std::unordered_map <std::string, std::string> const default_params = {
		{"m","300"},
		{"d","2"},
		{"l","0"},
		{"t","1"},
		{"a","50"},
		{"c","1"},
		{"m0","10"},
		{"dt","0.1"},
		{"dtn","1"},
		{"dts","0"},
		{"lx","4"},
		{"lxs","0"},
		{"lxn","1"},
		{"ly","4"},
		{"lys","0"},
		{"lyn","1"},
		{"lz","1"},
		{"lzs","0"},
		{"lzn","1"},
		{"b","6"},
		{"bs","0"},
		{"bn","1"},
		{"u","2"},
		{"us","0"},
		{"un","1"},
		{"mu","0"},
		{"mus","0"},
		{"mun","1"},
		{"th","1"},
		{"ti","1"},
		{"q","0"},
		{"qr","1" },
		{"cg","0"},
		{"ct","0"},
		{"sf","0"},
		{"sfn","1"}
	};

	// -------------------------------------------------------- CLASS
	class ui : public user_interface {
	private:
		v_1d<double> t;																						// hopping coefficients
		int lattice_type; 																					// for non_numeric data
		double t_fill;
		int inner_threads, outer_threads;																	// thread parameters
		int sf, sfn;																						// self learning parameters
		bool quiet, save_conf, cal_times, useHirsh;															// bool flags
		int dim, lx, ly, lz, lx_step, ly_step, lz_step, lx_num, ly_num, lz_num;								// real space proprties
		double beta, beta_step, U, U_step, mu, mu_step, dtau, dtau_step;									// physical params
		int U_num, mu_num, dtau_num, beta_num;
		int M_0, p, M, mcSteps, avsNum, corrTime;															// time properties

		// -------------------------------------------------------- HELPER FUNCTIONS
		void collectAvs(const HubbardParams& params);
		void collectRealSpace(std::string name_times, std::string name, const HubbardParams& params, std::shared_ptr<averages_par> avs);
		void collectFouriers(std::string name_times, std::string name, const HubbardParams& params, std::shared_ptr<averages_par> avs);
	public:
		// ----------------------- CONSTRUCTORS
		ui() = default;
		ui(int argc, char** argv);
		// ----------------------- PARSER FOR HELP
		void exit_with_help() override;
		// ----------------------- REAL PARSER
		void parseModel(int argc, const v_1d<std::string>& argv) override;									// the function to parse the command line
		// ----------------------- HELPERS
		void set_default() override;																		// set default parameters
		// ----------------------- SIMULATION
		void make_simulation() override;
	};
}

#endif // !UI_H
