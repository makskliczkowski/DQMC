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
		void collectRealSpace(std::string name_times, std::string name, const hubbard::HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat);
		void collectFouriers(std::string name_times, std::string name, const hubbard::HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat);
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

inline void hubbard::ui::collectAvs(const hubbard::HubbardParams& params)
{
	using namespace std;
	auto start = chrono::high_resolution_clock::now();
	const auto prec = 10;
	auto& [dim, beta, mu, U, Lx, Ly, Lz, M, M0, p, dtau] = params;
	// parameters and constants
	std::shared_ptr<averages_par> avs;
	// model
	std::shared_ptr<Lattice> lat;
	// ------------------------------- set lattice --------------------------------
	switch (this->lattice_type) {
	case 0:
		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
		break;
	case 1:
		lat = std::make_shared<HexagonalLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
		break;
	default:
		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
		break;
	}
	// ------------------------------- set model --------------------------------
	std::unique_ptr<hubbard::HubbardModel> model = std::make_unique<hubbard::HubbardQR>(this->t, params, lat, this->inner_threads);

	std::ofstream fileLog, fileGup, fileGdown, fileSignLog;

	auto dirs = model->get_directories(this->saving_dir);																				// take all the directories needed

	// RELAX
	if (sf == 0)																														// without using machine learning to self learn
	{
		model->relaxation(impDef::algMC::heat_bath, this->mcSteps, this->save_conf, this->quiet);										// this can also handle saving configurations
		if (!this->save_conf) {
			// FILES
			openFile(fileLog, this->saving_dir + "HubbardLog.csv", std::ios::in | std::ios::app);
			openFile(fileSignLog, this->saving_dir + "HubbardSignLog_" + dirs->LxLyLz + ",U=" + str_p(U, 2) + \
				",beta=" + str_p(beta, 2) + ",dtau=" + str_p(dtau, 4) + \
				".dat", std::ios::in | std::ios::app);

			// REST
			model->average(impDef::algMC::heat_bath, this->corrTime, this->avsNum, 1, this->quiet);
			avs = model->get_avs();

			// SAVING TO STRING
			printSeparatedP(fileLog, ',', 20, true, 4, lat->get_type(), this->mcSteps, this->avsNum,
				this->corrTime, M, M_0, dtau,
				Lx, Ly, Lz, beta, U,
				mu, avs->av_occupation, avs->sd_occupation,
				avs->av_sign, avs->sd_sign,
				avs->av_Ek, avs->sd_Ek,
				avs->av_M2z, avs->sd_M2z,
				avs->av_M2x, tim_s(start));
			printSeparatedP(fileSignLog, '\t', 12, true, prec, avs->av_occupation, avs->av_sign, mu);
#pragma omp critical
			printSeparatedP(stout, '\t', 15, true, 3, VEQP(avs->av_occupation, 3), VEQP(avs->av_sign, 3), VEQP(avs->av_M2z, 3));
#pragma omp critical
			fileLog.close();
#pragma omp critical
			fileSignLog.close();
			this->collectRealSpace(dirs->nameNormalTime, dirs->nameNormal, params, avs, lat);
			this->collectFouriers(dirs->nameFouriersTime, dirs->nameFouriers, params, avs, lat);
		}
	}
	std::cout << "FINISHED EVERYTHING - Time taken: " << tim_s(start) << " seconds" << endl;
}



#endif // !UI_H
