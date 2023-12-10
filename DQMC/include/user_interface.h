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
	struct ModP 
	{
		// ############################### TYPE ################################
		UI_PARAM_CREATE_DEFAULT(modTyp		, MY_MODELS,	MY_MODELS::HUBBARD_M);

		// ############### Hubbard ###############
		v_1d<double>	t_;
		v_1d<double>	tt_;
		// Hubbard U on various lattice sizes
		v_1d<double>	U_;										
		v_1d<double>	dU_;
		UI_PARAM_CREATE_DEFAULTD(Un, uint, 1.0);
		// chemical potential on various lattice sizes
		v_1d<double>	mu_;									
		v_1d<double>	dmu_;									
		UI_PARAM_CREATE_DEFAULTD(mun, uint, 1.0);

		UI_PARAM_CREATE_DEFAULTD(M			, double,		4.0);
		UI_PARAM_CREATE_DEFAULTD(beta		, double,		2.0);
		UI_PARAM_CREATE_DEFAULTD(T			, double,		0.5);
		UI_PARAM_CREATE_DEFAULTD(dtau		, double,		0.1);
		UI_PARAM_CREATE_DEFAULTD(M0			, double,		1.0);
		UI_PARAM_CREATE_DEFAULTD(Ns			, double,		1.0);
		UI_PARAM_CREATE_DEFAULTD(Nband		, double,		1.0);

		void setDefault() 
		{
			UI_PARAM_SET_DEFAULT(modTyp);
			// ------ Hubbard ------
			this->t_	=	v_1d<double>(Ns_ * Nband_, 1.0);
			this->tt_	=	v_1d<double>(Ns_ * Nband_, 0.0);

			this->U_	=	v_1d<double>(Ns_ * Nband_, 1.0);
			this->dU_	=	v_1d<double>(Ns_ * Nband_, 0.0);

			this->mu_	=	v_1d<double>(Ns_ * Nband_, 0.0);
			this->dmu_	=	v_1d<double>(Ns_ * Nband_, 0.0);

			// ------ General ------
			UI_PARAM_SET_DEFAULT(T);
			UI_PARAM_SET_DEFAULT(beta);
			UI_PARAM_SET_DEFAULT(dtau);
			UI_PARAM_SET_DEFAULT(M0);
			UI_PARAM_SET_DEFAULT(Ns);
			UI_PARAM_SET_DEFAULT(Nband);
		};
	};
	
	/*
	* @brief Defines parameters used later for the simulation
	*/
	struct SimP 
	{

		UI_PARAM_CREATE_DEFAULTD(mcS, int, 100);
		UI_PARAM_CREATE_DEFAULTD(mcC, int, 1);
		UI_PARAM_CREATE_DEFAULTD(mcB, int, 100); 
		UI_PARAM_CREATE_DEFAULTD(mcA, int, 100); 
		UI_PARAM_CREATE_DEFAULTD(mcCheckLoad, std::string, "");
		UI_PARAM_CREATE_DEFAULTD(mcCheckSave, std::string, "");

		void setDefault() {
			UI_PARAM_SET_DEFAULT(mcS);
			UI_PARAM_SET_DEFAULT(mcC);
			UI_PARAM_SET_DEFAULT(mcB);
			UI_PARAM_SET_DEFAULT(mcA);
			UI_PARAM_SET_DEFAULT(mcCheckLoad);
			UI_PARAM_SET_DEFAULT(mcCheckLoad);
		};
	};

	/*
	* @brief Defines lattice used later for the models
	*/
	struct LatP 
	{
		UI_PARAM_CREATE_DEFAULT(bc	, BoundaryConditions, BoundaryConditions::PBC	);
		UI_PARAM_CREATE_DEFAULT(typ	, LatticeTypes		, LatticeTypes::SQ			);

		UI_PARAM_CREATE_DEFAULT(Lx	, uint				, 2							);
		UI_PARAM_CREATE_DEFAULT(dLx	, uint				, 0							);
		UI_PARAM_CREATE_DEFAULT(LxN	, uint				, 1							);

		UI_PARAM_CREATE_DEFAULT(Ly	, uint				, 1							);
		UI_PARAM_CREATE_DEFAULT(dLy	, uint				, 0							);
		UI_PARAM_CREATE_DEFAULT(LyN	, uint				, 1							);

		UI_PARAM_CREATE_DEFAULT(Lz	, uint				, 1							);
		UI_PARAM_CREATE_DEFAULT(dLz	, uint				, 0							);
		UI_PARAM_CREATE_DEFAULT(LzN	, uint				, 1							);

		UI_PARAM_CREATE_DEFAULT(dim	, uint				, 1							);

		std::shared_ptr<Lattice> lat;
		
		void setDefault() {
			UI_PARAM_SET_DEFAULT(typ);
			UI_PARAM_SET_DEFAULT(bc);
			UI_PARAM_SET_DEFAULT(Lx);
			UI_PARAM_SET_DEFAULT(Lx);
			UI_PARAM_SET_DEFAULT(Lx);

			UI_PARAM_SET_DEFAULT(Ly);
			UI_PARAM_SET_DEFAULT(dLy);
			UI_PARAM_SET_DEFAULT(LyN);

			UI_PARAM_SET_DEFAULT(Lz);
			UI_PARAM_SET_DEFAULT(dLz);
			UI_PARAM_SET_DEFAULT(LzN);
			UI_PARAM_SET_DEFAULT(dim);
		};
	};
}

class UI : public UserInterface 
{
protected:

	// LATTICE params
	UI_PARAMS::LatP latP;

	// MODELS params
	UI_PARAMS::ModP modP;

	// SIMULATION params
	UI_PARAMS::SimP simP;

	// define basic models
	std::shared_ptr<DQMC<2>> mod_s2_;

	void setDefaultMap() final override 
	{
		this->defaultParams = {
			{			"f"			, std::make_tuple(""	, FHANDLE_PARAM_DEFAULT)		},			// file to read from directory
			// ---------------- lattice parameters ----------------
			UI_OTHER_MAP(d			, this->latP._dim		, FHANDLE_PARAM_BETWEEN(1., 3.)	),	
			UI_OTHER_MAP(bc			, this->latP._bc		, FHANDLE_PARAM_BETWEEN(0., 3.)	),
			UI_OTHER_MAP(l			, this->latP._typ		, FHANDLE_PARAM_BETWEEN(0., 1.)	),
			// Lx
			UI_OTHER_MAP(lx			, this->latP._Lx		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(dlx		, this->latP._dLx		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(lxn		, this->latP._LxN		, FHANDLE_PARAM_HIGHERV(1)		),
			// Ly
			UI_OTHER_MAP(ly			, this->latP._Ly		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(dly		, this->latP._dLy		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(lyn		, this->latP._LyN		, FHANDLE_PARAM_HIGHERV(1)		),
			// Lz
			UI_OTHER_MAP(lz			, this->latP._Lz		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(dlz		, this->latP._dLz		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(lzn		, this->latP._LzN		, FHANDLE_PARAM_HIGHERV(1)		),
			// ------------------- MC parameters ------------------
			UI_OTHER_MAP(mcS		, this->simP._mcS		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(mcC		, this->simP._mcC		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(mcB		, this->simP._mcB		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(mcA		, this->simP._mcA		, FHANDLE_PARAM_HIGHER0			),
			UI_OTHER_MAP(mcCPL		, this->simP._mcCheckLoad, FHANDLE_PARAM_DEFAULT		),
			UI_OTHER_MAP(mcCPS		, this->simP._mcCheckSave, FHANDLE_PARAM_DEFAULT		),
			// ----------------- model parameters -----------------
			UI_OTHER_MAP(mod		, this->modP._modTyp	, FHANDLE_PARAM_BETWEEN(0., 2.)	),
			// -------- Hubbard
			UI_PARAM_MAP(M			, this->modP._M			, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(beta		, this->modP._beta		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(T			, this->modP._T			, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(mun		, this->modP._mun		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(Un			, this->modP._Un		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(dtau		, this->modP._dtau		, FHANDLE_PARAM_DEFAULT			),
			UI_PARAM_MAP(M0			, this->modP._M0		, FHANDLE_PARAM_DEFAULT			),
			// ----------------------- other ----------------------
			UI_OTHER_MAP(fun		, -1.					, FHANDLE_PARAM_HIGHERV(-1.0)	),			// choice of the function to be calculated
			UI_OTHER_MAP(th			, 1.0					, FHANDLE_PARAM_HIGHER0			),			// number of threads
			UI_OTHER_MAP(ith		, 1.0					, FHANDLE_PARAM_HIGHER0			),			// number of threads for inner simulation
			UI_OTHER_MAP(q			, 0.0					, FHANDLE_PARAM_DEFAULT			),			// quiet?
			UI_OTHER_MAP(dir		, "DEFALUT"				, FHANDLE_PARAM_DEFAULT			),
		};
	};
private:
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% I N N E R    M E T H O D S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	void makeSim();
	void makeSimSweep();
	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% D E F I N I T I O N S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	bool defineLattice();
	bool defineModel();
	bool defineModel(DQMC<2>* _model);
	bool defineModels(bool _createLat = true);
	bool defineModels(bool _createLat, DQMC<2>* _model);

public:

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% C O N S T R U C T O R S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	~UI()													= default;
	UI()													= default;
	UI(int argc, char** argv)
	{
		this->setDefaultMap();
		this->init(argc, argv);
	};

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% P A R S E R  F O R   H E L P %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	void exitWithHelp() override
	{
		UserInterface::exitWithHelp();
		printf(
			" ------------------------------------------------- General DQMC parser for C++ ------------------------------------------------ \n"
			"Usage of the DQMC library:\n"
			"options:\n"
			"-m monte carlo steps	: bigger than 0 (default 300) \n"
			"-d dimension			: set dimension (default 2) \n"
			"	1 -- 1D \n"
			"	2 -- 2D \n"
			"	3 -- 3D \n"
			"-l lattice type		: (default square) -> \n"
			"   square \n"
			"-t exchange constant	: set hopping (default 1) [VECTOR VALUE] \n"
			"-a averages number		: set tolerance for statistics, bigger than 0 (default 50) \n"
			"-c correlation time	: whether to wait to collect observables, bigger than 0 (default 1) \n"
			"-M0 Trotter slices		: this sets how many slices in division our Trotter times number has, bigger than 0 (default 10)\n"
			"-dt dtau				: Trotter step (default 0.1)\n"
			"-dts dtau				: Trotter step step (default 0)\n"
			"-dtn dtau				: Trotter steps number (default 1)\n"
			// SIMULATIONS STEPS
			"-lx  x-length			: bigger than 0(default 4)\n"
			"-lxs x-length step		: integer, bigger than 0 (default 1)\n"
			"-lxn x-length steps #	: integer bigger than 0 (default 1)\n"
			"\n"
			"-ly  y-length			: bigger than 0(default 4)\n"
			"-lys y-length step		: integer, bigger than 0 (default 1)\n"
			"-lyn y-length steps #	: integer bigger than 0 (default 1)\n"
			"\n"
			"-lz  z-length			: bigger than 0(default 1)\n"
			"-lzs z-length step		: integer, bigger than 0 (default 1)\n"
			"-lzn z-length steps #	: integer bigger than 0 (default 1)\n"
			"\n"
			"-b  inversed temp		: bigger than 0 (default 6)\n"
			"-bs inversed temp step : bigger than 0 (default 1)\n"
			"-bn inversed temp #	: integer bigger than 0 (default 1)\n"
			"\n"
			"-u  interaction U		: (default 2)\n"
			"	-[VECTOR VALUE]\n"
			"-us interaction step	: bigger than 0, start from smaller (default 1)\n"
			"-un interaction number : integer bigger than 0 (default 1)\n"
			"\n"
			"-mu  chemical potential: (default 0.0 -> half filling)\n"
			"-mus mu step			: bigger than 0, start from smaller (default 1)\n"
			"-mun mu number			: integer bigger than 0 (default 1)\n"
			"\n"
			"-th outer threads		: number of outer threads (default 1)\n"
			"-ti inner threads		: number of inner threads (default 1)\n"
			"\n"
			"-cg save-config		: for machine learning -> we will save all Trotter times at the same time (default false)\n"
			"-ct collect non-equal time averages \n"
			"-hs use hirsh for collecting time averages \n"
			"\n"
			"-sf use self-learning	: ->\n"
			"	0 (default - do not use),\n"
			"	1 (just train) [NOT-IMPLEMENTED],\n"
			"	2 (make simulation from existing network) [NOT-IMPLEMENTED]\n"
			" ------------------------------------------ Copyright : Maksymilian Kliczkowski, 2023 ------------------------------------------ "
		);
		std::exit(1);
	}
	
	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% R E A L   P A R S E R %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	void funChoice()						final override;
	void parseModel(int argc, cmdArg& argv) final override;

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% H E L P E R S %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	void setDefault()						final override;

	// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% S I M U L A T I O N %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
};