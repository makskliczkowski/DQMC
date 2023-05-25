#pragma once
#include "../source/src/UserInterface/ui.h"

#ifdef DEBUG
//#define DEBUG_BINARY
#else
//#define OMP_NUM_THREADS 16;
	#include <thread>
	#include <mutex>
#endif

// ####################### MODELS ###########################
#ifndef HUBBARD_H										 // #
#include "Models/hubbard.h"								 // #
#endif // !HUBBARD_H									 // #
// ##########################################################

// ###################### LATTICES ##########################
#ifndef SQUARE_H										 // #
#include "../source/src/Lattices/square.h"				 // #
#endif													 // #
#ifndef HEXAGONAL_H										 // #
#include "../source/src/Lattices/hexagonal.h"			 // #
#endif													 // #
// ##########################################################

namespace UI_PARAMS
{
	/*
	* @brief Defines parameters used later for the models
	*/
	struct ModP {
		// ############### TYPE ################
		UI_PARAM_CREATE_DEFAULT(modTyp		, MY_MODELS	, MY_MODELS::HUBBARD_M);

		// ############### Hubbard ###############
		v_1d<double>	t_;
		UI_PARAM_CREATE_DEFAULTD(M		, double, 4.0);
		UI_PARAM_CREATE_DEFAULTD(U		, double, 2.0);
		UI_PARAM_CREATE_DEFAULTD(beta	, double, 2.0);
		UI_PARAM_CREATE_DEFAULTD(T		, double, 0.5);
		UI_PARAM_CREATE_DEFAULTD(mu		, double, 0.0);
		UI_PARAM_CREATE_DEFAULTD(dtau	, double, 0.1);
		UI_PARAM_CREATE_DEFAULTD(M0		, double, 1.0);
		UI_PARAM_CREATE_DEFAULTD(Ns		, double, 1.0);

		void setDefault() {
			UI_PARAM_SET_DEFAULT(modTyp);
			// Hubbard
			this->t_	=	v_1d<double>(Ns_, 1.0);
			UI_PARAM_SET_DEFAULT(U);
			UI_PARAM_SET_DEFAULT(T);
			UI_PARAM_SET_DEFAULT(beta);
			UI_PARAM_SET_DEFAULT(mu);
			UI_PARAM_SET_DEFAULT(dtau);
			UI_PARAM_SET_DEFAULT(M0);
			UI_PARAM_SET_DEFAULT(Ns);
		};
	};

	/*
	* @brief Defines lattice used later for the models
	*/
	struct LatP {
		UI_PARAM_CREATE_DEFAULT(bc	, BoundaryConditions, BoundaryConditions::PBC	);
		UI_PARAM_CREATE_DEFAULT(typ	, LatticeTypes		, LatticeTypes::SQ			);
		UI_PARAM_CREATE_DEFAULT(Lx	, uint				, 2							);
		UI_PARAM_CREATE_DEFAULT(Ly	, uint				, 1							);
		UI_PARAM_CREATE_DEFAULT(Lz	, uint				, 1							);
		UI_PARAM_CREATE_DEFAULT(dim	, uint				, 1							);

		std::shared_ptr<Lattice> lat;
		
		void setDefault() {
			UI_PARAM_SET_DEFAULT(typ);
			UI_PARAM_SET_DEFAULT(bc);
			UI_PARAM_SET_DEFAULT(Lx);
			UI_PARAM_SET_DEFAULT(Ly);
			UI_PARAM_SET_DEFAULT(Lz);
			UI_PARAM_SET_DEFAULT(dim);
		};
	};
}

class UI : public UserInterface {
protected:

	// LATTICE params
	UI_PARAMS::LatP latP;

	// MODELS params
	UI_PARAMS::ModP modP;

	// define basic models
	std::shared_ptr<DQMC<2>> mod_s2_;


	void setDefaultMap()					final override {
		this->defaultParams = {
			{			"f"			, std::make_tuple(""	, FHANDLE_PARAM_DEFAULT)		},			// file to read from directory
			// ---------------- lattice parameters ----------------
			UI_OTHER_MAP(d			, this->latP._dim		, FHANDLE_PARAM_BETWEEN(1., 3.)	),	
			UI_OTHER_MAP(bc			, this->latP._bc		, FHANDLE_PARAM_BETWEEN(0., 3.)	),
			UI_OTHER_MAP(l			, this->latP._typ		, FHANDLE_PARAM_BETWEEN(0., 1.)	),
			UI_OTHER_MAP(lx			, this->latP._Lx		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(ly			, this->latP._Ly		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(lz			, this->latP._Lz		, FHANDLE_PARAM_HIGHER0			),
			// ---------------- model parameters ----------------
			UI_OTHER_MAP(mod		, this->modP._modTyp	, FHANDLE_PARAM_BETWEEN(0., 2.)	),
			// -------- Hubbard
			UI_PARAM_MAP(M			, this->modP._M			, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(U			, this->modP._U			, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(beta		, this->modP._beta		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(T			, this->modP._T			, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(mu			, this->modP._mu		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(dtau		, this->modP._dtau		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(M0			, this->modP._M0		, FHANDLE_PARAM_DEFAULT			),
			// ---------------- other ----------------
			UI_OTHER_MAP(fun		, -1.					, FHANDLE_PARAM_HIGHERV(-1.0)	),			// choice of the function to be calculated
			UI_OTHER_MAP(th			, 1.0					, FHANDLE_PARAM_HIGHER0			),			// number of threads
			UI_OTHER_MAP(q			, 0.0					, FHANDLE_PARAM_DEFAULT			),			// quiet?
			UI_OTHER_MAP(dir		, "DEFALUT"				, FHANDLE_PARAM_DEFAULT			),
		};
	};
private:
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% I N N E R    M E T H O D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	void makeSim();
	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D E F I N I T I O N S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	bool defineModels(bool _createLat = true);
	bool defineModel();

	public:
	// -----------------------------------------------        CONSTRUCTORS  		-------------------------------------------
	~UI()													= default;
	UI()													= default;
	UI(int argc, char** argv)
	{
		this->setDefaultMap();
		this->init(argc, argv);
	};

	// -----------------------------------------------   	 PARSER FOR HELP  		-------------------------------------------
	void exitWithHelp() override {
		printf(
			"Usage: name [options] outputDir \n"
			"options:\n"
			" The input can be both introduced with [options] described below or with giving the input directory \n"
			" (which also is the flag in the options) \n"
			" options:\n"
			"-f input file for all of the options : (default none) \n"
			"-m monte carlo steps : bigger than 0 (default 300) \n"
			"-d dimension : set dimension (default 2) \n"
			"	1 -- 1D \n"
			"	2 -- 2D \n"
			"	3 -- 3D -> NOT IMPLEMENTED YET \n"
			"-l lattice type : (default square) -> CHANGE NOT IMPLEMENTED YET \n"
			"   square \n"
			"-t exchange constant : set hopping (default 1) \n"
			"   is numeric? - constant value \n"
			"   'r' - uniform random on each (NOT IMPLEMENTED YET) \n"
			"-a averages number : set tolerance for statistics, bigger than 0 (default 50) \n"
			"-c correlation time : whether to wait to collect observables, bigger than 0 (default 1) \n"
			"-M0 all Trotter slices sections : this sets how many slices in division our Trotter times number has, bigger than 0 (default 10)\n"
			"-dt dtau : Trotter step (default 0.1)\n"
			"-dts dtau : Trotter step step (default 0)\n"
			"-dtn dtau : Trotter steps number(default 1)\n"
			// SIMULATIONS STEPS
			"-lx x-length : bigger than 0(default 4)\n"
			"-lxs x-length step : integer, bigger than 0 (default 1)\n"
			"-lxn x-length steps number : integer bigger than 0(default 1)\n"
			"\n"
			"-ly y-length : bigger than 0(default 4)\n"
			"-lys y-length step : integer, bigger than 0 (default 1)\n"
			"-lyn y-length steps number : integer bigger than 0(default 1)\n"
			"\n"
			"-lz z-length : bigger than 0(default 1)\n"
			"-lzs z-length step : integer, bigger than 0 (default 1)\n"
			"-lzn z-length steps number : integer bigger than 0(default 1)\n"
			"\n"
			"-b inversed temperature :bigger than 0 (default 6)\n"
			"-bs inversed temperature step : bigger than 0 (default 1)\n"
			"-bn inversed temperature number : integer bigger than 0(default 1)\n"
			"\n"
			"-u interaction U : (default 2) -> CURRENTLy ONLY U>0\n"
			"-us interaction step : bigger than 0, start from smaller (default 1)\n"
			"-un interaction number : integer bigger than 0(default 1)\n"
			"\n"
			"-mu chemical potential mu : (default 0 -> half filling)\n"
			"-mus chemical potential mu step : bigger than 0, start from smaller (default 1)\n"
			"-mun chemical potential mu number : integer bigger than 0(default 1)\n"
			"\n"
			"-th outer threads : number of outer threads (default 1)\n"
			"-ti inner threads : number of inner threads (default 1)\n"
			"-q : 0 or 1 -> quiet mode (no outputs) (default false)\n"
			"\n"
			"-cg save-config : for machine learning -> we will save all Trotter times at the same time(default false)\n"
			"-ct collect time averages - slower \n"
			"-hs use hirsh for collecting time averages \n"
			"\n"
			"-sf use self-learning : 0(default) - do not use, 1 (just train parameters and exit), 2 (just make simulation from the existing network)\n"
			"-sfn - the number of configurations saved to train(default 50)\n"
			"-h - help\n"
		);
		std::exit(1);
	}
	
	// -----------------------------------------------    	   REAL PARSER          -------------------------------------------
	// the function to parse the command line
	void funChoice()						final override;
	void parseModel(int argc, cmdArg& argv) final override;

	// ----------------------------------------------- 			HELPERS  			-------------------------------------------
	void setDefault()						final override;

	// -----------------------------------------------  	   SIMULATION 
};

////
////namespace hubbard {
////	// -------------------------------------------------------- CLASS
////	class ui : public user_interface 
////	{
////	private:
////		v_1d<double> t;																						// hopping coefficients
////		int lattice_type; 																					// for non_numeric data
////		double t_fill;
////		int inner_threads, outer_threads;																	// thread parameters
////		int sf, sfn;																						// self learning parameters
////		bool quiet, save_conf, cal_times, useHirsh;															// bool flags
////		int dim, lx, ly, lz, lx_step, ly_step, lz_step, lx_num, ly_num, lz_num;								// real space proprties
////		double beta, beta_step, U, U_step, mu, mu_step, dtau, dtau_step;									// physical params
////		int U_num, mu_num, dtau_num, beta_num;
////		int M_0, p, M, mcSteps, avsNum, corrTime;															// time properties
////
////		// -------------------------------------------------------- HELPER FUNCTIONS
////		void collectAvs(const HubbardParams& params);
////		void collectRealSpace(std::string name_times, std::string name, const hubbard::HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat);
////		void collectFouriers(std::string name_times, std::string name, const hubbard::HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat);
////	public:
////		// ----------------------- CONSTRUCTORS
////		~ui() = default;
////		ui() = default;
////		ui(int argc, char** argv);
//		// ----------------------- PARSER FOR HELP
//		//void exit_with_help() override;
//		//// ----------------------- REAL PARSER
//		//void parseModel(int argc, const v_1d<std::string>& argv) override;									// the function to parse the command line
//		//// ----------------------- HELPERS
//		//void set_default() override;																		// set default parameters
//		//void functionChoice() override {};
//		//// ----------------------- SIMULATION
//		//void make_simulation() override;
//	};
//}
//
//inline void hubbard::ui::collectAvs(const hubbard::HubbardParams& params)
//{
//	using namespace std;
//	auto start = chrono::high_resolution_clock::now();
//	const auto prec = 10;
//	auto& [dim, beta, mu, U, Lx, Ly, Lz, M, M0, p, dtau] = params;
//	// parameters and constants
//	std::shared_ptr<averages_par> avs;
//	// model
//	std::shared_ptr<Lattice> lat;
//	// ------------------------------- set lattice --------------------------------
//	switch (this->lattice_type) {
//	case 0:
//		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
//		break;
//	case 1:
//		lat = std::make_shared<HexagonalLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
//		break;
//	default:
//		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
//		break;
//	}
//	// ------------------------------- set model --------------------------------
//	std::unique_ptr<hubbard::HubbardModel> model = std::make_unique<hubbard::HubbardQR>(this->t, params, lat, this->inner_threads);
//
//	std::ofstream fileLog, fileGup, fileGdown, fileSignLog;
//
//	auto dirs = model->get_directories(this->saving_dir);																				// take all the directories needed
//
//	// RELAX
//	if (sf == 0)																														// without using machine learning to self learn
//	{
//		model->relaxation(impDef::algMC::heat_bath, this->mcSteps, this->save_conf, this->quiet);										// this can also handle saving configurations
//		if (!this->save_conf) {
//			// FILES
//			openFile(fileLog, this->saving_dir + "HubbardLog.csv", std::ios::in | std::ios::app);
//			openFile(fileSignLog, this->saving_dir + "HubbardSignLog_" + dirs->LxLyLz + ",U=" + str_p(U, 2) + \
//				",beta=" + str_p(beta, 2) + ",dtau=" + str_p(dtau, 4) + \
//				".dat", std::ios::in | std::ios::app);
//
//			// REST
//			model->average(impDef::algMC::heat_bath, this->corrTime, this->avsNum, 1, this->quiet);
//			avs = model->get_avs();
//
//			// SAVING TO STRING
//			printSeparatedP(fileLog, ',', 20, true, 4, lat->get_type(), this->mcSteps, this->avsNum,
//				this->corrTime, M, M_0, dtau,
//				Lx, Ly, Lz, beta, U,
//				mu, avs->av_occupation, avs->sd_occupation,
//				avs->av_sign, avs->sd_sign,
//				avs->av_Ek, avs->sd_Ek,
//				avs->av_M2z, avs->sd_M2z,
//				avs->av_M2x, tim_s(start), dirs->token);
//			printSeparatedP(fileSignLog, '\t', 12, true, prec, avs->av_occupation, avs->av_sign, mu);
//#pragma omp critical
//			printSeparatedP(stout, '\t', 15, true, 3, VEQP(avs->av_occupation, 3), VEQP(avs->av_sign, 3), VEQP(avs->av_M2z, 3));
//#pragma omp critical
//			fileLog.close();
//#pragma omp critical
//			fileSignLog.close();
//			this->collectRealSpace(dirs->nameNormalTime, dirs->nameNormal, params, avs, lat);
//			this->collectFouriers(dirs->nameFouriersTime, dirs->nameFouriers, params, avs, lat);
//		}
//	}
//	std::cout << "FINISHED EVERYTHING - Time taken: " << tim_s(start) << " seconds" << endl;
//}
//
//
//
//#endif // !UI_H
#pragma once
//#ifndef USER_INTERFACE_H
//#define USER_INTERFACE_H
//
//
////#include "../include/plog/Log.h"
////#include "../include/plog/Initializers/RollingFileInitializer.h"
////#include "../source/src/UserInterface/ui.h"
////#include "../include/hubbard_dqmc_qr.h"
//
//
//// -------------------------------------------------------- HUBBARD USER INTERFACE --------------------------------------------------------
//
//// -------------------------------------------------------- MAP OF DEFAULTS FOR HUBBARD
////std::unordered_map <std::string, std::string> const default_params = {
////	{"m","300"},
////	{"d","2"},
////	{"l","0"},
////	{"t","1"},
////	{"a","50"},
////	{"c","1"},
////	{"m0","10"},
////	{"dt","0.1"},
////	{"dtn","1"},
////	{"dts","0"},
////	{"lx","4"},
////	{"lxs","0"},
////	{"lxn","1"},
////	{"ly","4"},
////	{"lys","0"},
////	{"lyn","1"},
////	{"lz","1"},
////	{"lzs","0"},
////	{"lzn","1"},
////	{"b","6"},
////	{"bs","0"},
////	{"bn","1"},
////	{"u","2"},
////	{"us","0"},
////	{"un","1"},
////	{"mu","0"},
////	{"mus","0"},
////	{"mun","1"},
////	{"th","1"},
////	{"ti","1"},
////	{"q","0"},
////	{"qr","1" },
////	{"cg","0"},
////	{"ct","0"},
////	{"sf","0"},
////	{"sfn","1"}
////};
////
////namespace hubbard {
////	// -------------------------------------------------------- CLASS
////	class ui : public user_interface 
////	{
////	private:
////		v_1d<double> t;																						// hopping coefficients
////		int lattice_type; 																					// for non_numeric data
////		double t_fill;
////		int inner_threads, outer_threads;																	// thread parameters
////		int sf, sfn;																						// self learning parameters
////		bool quiet, save_conf, cal_times, useHirsh;															// bool flags
////		int dim, lx, ly, lz, lx_step, ly_step, lz_step, lx_num, ly_num, lz_num;								// real space proprties
////		double beta, beta_step, U, U_step, mu, mu_step, dtau, dtau_step;									// physical params
////		int U_num, mu_num, dtau_num, beta_num;
////		int M_0, p, M, mcSteps, avsNum, corrTime;															// time properties
////
////		// -------------------------------------------------------- HELPER FUNCTIONS
////		void collectAvs(const HubbardParams& params);
////		void collectRealSpace(std::string name_times, std::string name, const hubbard::HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat);
////		void collectFouriers(std::string name_times, std::string name, const hubbard::HubbardParams& params, std::shared_ptr<averages_par> avs, std::shared_ptr<Lattice> lat);
////	public:
////		// ----------------------- CONSTRUCTORS
////		~ui() = default;
////		ui() = default;
////		ui(int argc, char** argv);
//		// ----------------------- PARSER FOR HELP
//		//void exit_with_help() override;
//		//// ----------------------- REAL PARSER
//		//void parseModel(int argc, const v_1d<std::string>& argv) override;									// the function to parse the command line
//		//// ----------------------- HELPERS
//		//void set_default() override;																		// set default parameters
//		//void functionChoice() override {};
//		//// ----------------------- SIMULATION
//		//void make_simulation() override;
//	};
//}
//
//inline void hubbard::ui::collectAvs(const hubbard::HubbardParams& params)
//{
//	using namespace std;
//	auto start = chrono::high_resolution_clock::now();
//	const auto prec = 10;
//	auto& [dim, beta, mu, U, Lx, Ly, Lz, M, M0, p, dtau] = params;
//	// parameters and constants
//	std::shared_ptr<averages_par> avs;
//	// model
//	std::shared_ptr<Lattice> lat;
//	// ------------------------------- set lattice --------------------------------
//	switch (this->lattice_type) {
//	case 0:
//		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
//		break;
//	case 1:
//		lat = std::make_shared<HexagonalLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
//		break;
//	default:
//		lat = std::make_shared<SquareLattice>(Lx, Ly, Lz, params.dim, this->boundary_conditions);
//		break;
//	}
//	// ------------------------------- set model --------------------------------
//	std::unique_ptr<hubbard::HubbardModel> model = std::make_unique<hubbard::HubbardQR>(this->t, params, lat, this->inner_threads);
//
//	std::ofstream fileLog, fileGup, fileGdown, fileSignLog;
//
//	auto dirs = model->get_directories(this->saving_dir);																				// take all the directories needed
//
//	// RELAX
//	if (sf == 0)																														// without using machine learning to self learn
//	{
//		model->relaxation(impDef::algMC::heat_bath, this->mcSteps, this->save_conf, this->quiet);										// this can also handle saving configurations
//		if (!this->save_conf) {
//			// FILES
//			openFile(fileLog, this->saving_dir + "HubbardLog.csv", std::ios::in | std::ios::app);
//			openFile(fileSignLog, this->saving_dir + "HubbardSignLog_" + dirs->LxLyLz + ",U=" + str_p(U, 2) + \
//				",beta=" + str_p(beta, 2) + ",dtau=" + str_p(dtau, 4) + \
//				".dat", std::ios::in | std::ios::app);
//
//			// REST
//			model->average(impDef::algMC::heat_bath, this->corrTime, this->avsNum, 1, this->quiet);
//			avs = model->get_avs();
//
//			// SAVING TO STRING
//			printSeparatedP(fileLog, ',', 20, true, 4, lat->get_type(), this->mcSteps, this->avsNum,
//				this->corrTime, M, M_0, dtau,
//				Lx, Ly, Lz, beta, U,
//				mu, avs->av_occupation, avs->sd_occupation,
//				avs->av_sign, avs->sd_sign,
//				avs->av_Ek, avs->sd_Ek,
//				avs->av_M2z, avs->sd_M2z,
//				avs->av_M2x, tim_s(start), dirs->token);
//			printSeparatedP(fileSignLog, '\t', 12, true, prec, avs->av_occupation, avs->av_sign, mu);
//#pragma omp critical
//			printSeparatedP(stout, '\t', 15, true, 3, VEQP(avs->av_occupation, 3), VEQP(avs->av_sign, 3), VEQP(avs->av_M2z, 3));
//#pragma omp critical
//			fileLog.close();
//#pragma omp critical
//			fileSignLog.close();
//			this->collectRealSpace(dirs->nameNormalTime, dirs->nameNormal, params, avs, lat);
//			this->collectFouriers(dirs->nameFouriersTime, dirs->nameFouriers, params, avs, lat);
//		}
//	}
//	std::cout << "FINISHED EVERYTHING - Time taken: " << tim_s(start) << " seconds" << endl;
//}
//
//
//
//#endif // !UI_H
