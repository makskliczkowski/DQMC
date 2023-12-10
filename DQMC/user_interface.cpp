#include "include/user_interface.h"
int LASTLVL = 0;

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief model parser
* @param argc number of line arguments
* @param argv line arguments
*/
void UI::parseModel(int argc, cmdArg& argv)
{
	// ------------------ HELP -------------------
	if (std::string option = this->getCmdOption(argv, "-h"); option != "")
		this->exitWithHelp();

	// set default at first
	this->setDefault();

	std::string choosen_option = "";
	// ---------- SIMULATION PARAMETERS ----------
	{
		SETOPTION(		simP, mcS					);	// relaxation size
		SETOPTION(		simP, mcA					);	// averages size
		SETOPTION(		simP, mcB					);	// buckets size
		SETOPTION(		simP, mcC					);	// correlation size
		SETOPTIONV(		simP, mcCheckLoad, "mcCPL"	);	// should I use the checkpoint coefficients?
		SETOPTIONV(		simP, mcCheckSave, "mcCPS"	);	// should I use the checkpoint coefficients?
	}
	// ----------------- LATTICE -----------------
	{
		SETOPTIONV(		latP, typ, "l"		);
		SETOPTIONV(		latP, dim, "d"		);
		SETOPTION(		latP, Lx			);
		SETOPTION(		latP, dLx			);
		SETOPTION(		latP, LxN			);
		SETOPTION(		latP, Ly			);
		SETOPTION(		latP, dLy			);
		SETOPTION(		latP, LyN			);
		SETOPTION(		latP, Lz			);
		SETOPTION(		latP, dLz			);
		SETOPTION(		latP, LzN			);
		SETOPTION(		latP, bc			);
		if (!this->defineLattice())
			throw std::runtime_error("Couldn't create a lattice\n");
	}
	// ------------------ MODEL ------------------
	{
		SETOPTIONV(		modP, modTyp, "mod"	);	// model type

		SETOPTION(		modP, dtau			);
		SETOPTION(		modP, beta			);
		SETOPTION(		modP, M0			);
		SETOPTION(		modP, Nband			);
		// set parameters
		this->setOption(this->modP.Ns_, this->latP.lat->get_Ns());
		this->setOption(this->modP.T_, 1.0 / this->modP.beta_);
		this->setOption(this->modP.M_, this->modP.beta_ / this->modP.dtau_);

		// ############### Hubbard ###############
		// resize
		this->modP.t_.resize(this->latP.lat->get_Ns() * this->modP.Nband_);
		this->modP.tt_.resize(this->latP.lat->get_Ns() * this->modP.Nband_);
		this->modP.U_.resize(this->latP.lat->get_Ns() * this->modP.Nband_);
		this->modP.dU_.resize(this->latP.lat->get_Ns() * this->modP.Nband_);
		this->modP.mu_.resize(this->latP.lat->get_Ns() * this->modP.Nband_);
		this->modP.dmu_.resize(this->latP.lat->get_Ns() * this->modP.Nband_);
		SETOPTION(		modP, t				);

		SETOPTION(		modP, U				);
		SETOPTION(		modP, Un			);
		SETOPTIOND(		modP, dU, 0.0		);

		SETOPTION(		modP, mu			);
		SETOPTION(		modP, mun			);
		SETOPTIOND(		modP, dmu, 0.0		);

		SETOPTIOND(		modP, tt, 0.0		);

		// ############### Other M ###############
	}

	// ------------------ OTHERS ------------------
	{
		this->setOption(this->quiet			, argv, "q"		);
		this->setOption(this->threadNum		, argv, "th"	);
		this->setOption(this->threadNumIn	, argv, "ith"	);

		// later function choice
		this->setOption(this->chosenFun		, argv, "fun"	);
	}
	// ---------------- DIRECTORY -----------------
	bool setDir		[[maybe_unused]] =	this->setOption(this->mainDir, argv, "dir");
	this->mainDir	=	makeDirsC(fs::current_path().string(), "DATA", this->mainDir);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Setting parameters to default.
*/
void UI::setDefault()
{
	// lattice stuff
	this->latP.setDefault();

	// define basic model
	this->modP.setDefault();

	// others 
	this->threadNum = 1;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief chooses the method to be used later based on input -fun argument
*/
void UI::funChoice()
{
	LOGINFO_CH_LVL(0);
	LOGINFO("USING #THREADS=" + STR(this->threadNum), LOG_TYPES::CHOICE, 1);
	this->_timer.reset();
	switch (this->chosenFun)
	{
	case -1:
		// default case of showing the help
		this->exitWithHelp();
		break;
	case 11:
		// this option utilizes the Hubbard
		LOGINFO("SIMULATION: HAMILTONIAN WITH DQMC QR", LOG_TYPES::CHOICE, 1);
		this->makeSim();
		break;
	case 12:
		// this option utilizes the Hubbard Sweep
		LOGINFO("SIMULATION: HAMILTONIAN WITH DQMC QR - SWEEP", LOG_TYPES::CHOICE, 1);
		this->makeSimSweep();
		break;
	default:
		// default case of showing the help
		this->exitWithHelp();
		break;
	}
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Depending on the lattice type, define it!
*/
bool UI::defineLattice()
{
	switch (this->latP.typ_)
	{
	case LatticeTypes::SQ:
		this->latP.lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
			this->latP.dim_, this->latP.bc_);
		break;
	case LatticeTypes::HEX:
		this->latP.lat = std::make_shared<HexagonalLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
			this->latP.dim_, this->latP.bc_);
		break;
	default:
		this->latP.lat = std::make_shared<SquareLattice>(this->latP.Lx_, this->latP.Ly_, this->latP.Lz_,
			this->latP.dim_, this->latP.bc_);
		break;
	};
	return true;
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief defines the models based on the input parameters - interacting. Also, defines the lattice
* @param _model model to be set
* @returns correctness of the setting
*/
bool UI::defineModels(bool _createLat, DQMC<2>* _model)
{
	// create lattice
	if (_createLat && !this->latP.lat)
		this->defineLattice();
	return this->defineModel(_model);
}

bool UI::defineModels(bool _createLat)
{
	// create lattice
	if (_createLat && !this->latP.lat)
		this->defineLattice();
	return this->defineModel();
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief defines the models based on the input parameters - interacting
*/
bool UI::defineModel(DQMC<2>* _model)
{
	switch (this->modP.modTyp_)
	{
	case MY_MODELS::HUBBARD_M:
		_model			= new Hubbard(this->modP.T_, this->latP.lat, this->modP.M_, this->modP.M0_,
													 this->modP.t_, this->modP.U_, this->modP.mu_, this->modP.dtau_,
													 this->modP.Nband_, this->modP.tt_);
		break;
	default:
		_model			= new Hubbard(this->modP.T_, this->latP.lat, this->modP.M_, this->modP.M0_,
													 this->modP.t_, this->modP.U_, this->modP.mu_, this->modP.dtau_,
													 this->modP.Nband_, this->modP.tt_);
		break;
	}
	return true;
}

bool UI::defineModel()
{
	switch (this->modP.modTyp_)
	{
	case MY_MODELS::HUBBARD_M:
		this->mod_s2_	= std::make_shared<Hubbard>(this->modP.T_, this->latP.lat, this->modP.M_, this->modP.M0_,
													this->modP.t_, this->modP.U_, this->modP.mu_, this->modP.dtau_,
													this->modP.Nband_, this->modP.tt_);
		break;
	default:
		this->mod_s2_	= std::make_shared<Hubbard>(this->modP.T_, this->latP.lat, this->modP.M_, this->modP.M0_,
													this->modP.t_, this->modP.U_, this->modP.mu_, this->modP.dtau_,
													this->modP.Nband_, this->modP.tt_);
		break;
	}
	return true;
}
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief Makes the simulation for a single Hamiltonian - without the outer loop
*/
void UI::makeSim()
{
	_timer.reset();
	BEGIN_CATCH_HANDLER
	{
		// define models
		if (!this->defineModels(true))
			return;
		// check inner threads
		this->mod_s2_->setThread(this->threadNumIn);
		
		// set directories
		this->mod_s2_->setDir(this->mainDir);
		
		// set the values from the checkpoint
		try 
		{
			if (!this->simP.mcCheckLoad_.empty())
				this->mod_s2_->setHS(this->simP.mcCheckLoad_);
		}
		catch (std::exception& e)
		{
			LOGINFO(LOG_TYPES::ERROR, "Couldn't setup the auxiliary fields from the standard path", 2);
			LOGINFO(LOG_TYPES::ERROR, e.what(), 2);
		}

		// start the relaxation
		BEGIN_CATCH_HANDLER
		{
			_timer.checkpoint("relaxation");
			this->mod_s2_->relaxes(this->simP.mcS_, this->quiet, _timer.point("relaxation"));
			LOGINFO(_timer.point("relaxation"), "DQMC: relaxation ", 0);
		}
		END_CATCH_HANDLER("Failed to perform the saving... ", ;);

		// save the configuration
		BEGIN_CATCH_HANDLER
		{
			if (this->simP.mcCheckSave_ == "date")
			{
				std::string time = prettyTime();
				this->mod_s2_->saveCheckPoint(this->mod_s2_->dir_->mainDir, "HS_" + time + ".h5");
			}
			else if (!this->simP.mcCheckSave_.empty())
				this->mod_s2_->saveCheckPoint(this->mod_s2_->dir_->mainDir, this->simP.mcCheckSave_);
			else
				this->mod_s2_->saveCheckPoint(this->mod_s2_->dir_->mainDir, "HS.h5");
		}
		catch (std::exception& e)
		{
			LOGINFO(LOG_TYPES::ERROR, "Couldn't setup the auxiliary fields to the standard path", 2);
			LOGINFO(LOG_TYPES::ERROR, e.what(), 2);
		}

		// start the averaging
		BEGIN_CATCH_HANDLER
		{
			_timer.checkpoint("average");
			this->mod_s2_->average(this->simP.mcS_, this->simP.mcC_, 
									this->simP.mcA_, this->simP.mcB_, this->quiet, _timer.point("average"));
			LOGINFO(_timer.point("average"), "DQMC: average ", 0);
		}
		END_CATCH_HANDLER("Failed to perform the averages... ", return;);

		// start the saving
		//BEGIN_CATCH_HANDLER
		//{
		//	_timer.checkpoint("save");
		//	this->mod_s2_->saveAverages();
		//	LOGINFO(_timer.point("save"), "DQMC: save ", 0);
		//}
		//END_CATCH_HANDLER("Failed to perform the saving... ", ;);
	}	
	END_CATCH_HANDLER(std::string(__FUNCTION__), ;);
}

/*
* @brief Sweep the parameters within the Monte Carlo simulation, taking into account the sweep constants
*/
void UI::makeSimSweep()
{
	_timer.reset();
	BEGIN_CATCH_HANDLER
	{
		this->defineLattice();
		
		// set the parameters to be run
		v_1d<std::tuple<uint, uint>> _params;
		for (auto Ui = 0; Ui < this->modP.Un_; Ui++)
			for (auto mui = 0; mui < this->modP.mun_; mui++)
				_params.push_back(std::make_tuple(Ui, mui));

		// iterate parameters
#ifndef _DEBUG
#pragma omp parallel for num_threads(this->threadNum)
#endif
		for (const auto& [Ui, mui] : _params)
		{
			LOGINFO("Doing " + VEQ(Ui), LOG_TYPES::TRACE, 2);
			auto U			=	VEC::addVecR(this->modP.U_, VEC::mulVecR(this->modP.dU_, (double)Ui));
			LOGINFO("Doing " + VEQ(mui), LOG_TYPES::TRACE, 2);
			auto mu			=	VEC::addVecR(this->modP.mu_, VEC::mulVecR(this->modP.dmu_, (double)mui));
			// define model
			DQMC2* _mod		=	new Hubbard(this->modP.T_, this->latP.lat, this->modP.M_, this->modP.M0_,
											this->modP.t_, U, mu, this->modP.dtau_,
											this->modP.Nband_, this->modP.tt_);
			// check inner threads
			_mod->setThread(this->threadNumIn);	
			// set directories
			_mod->setDir(this->mainDir);
		
			// set the values from the checkpoint
			BEGIN_CATCH_HANDLER	
			{
				if (!this->simP.mcCheckLoad_.empty())
					_mod->setHS(this->simP.mcCheckLoad_);
			}
			catch (std::exception& e)
			{
				LOGINFO(LOG_TYPES::ERROR, "Couldn't setup the auxiliary fields from the standard path", 2);
				LOGINFO(LOG_TYPES::ERROR, e.what(), 2);
				if (_mod) delete _mod; continue;
			}

			// start the relaxation
			BEGIN_CATCH_HANDLER
			{
				_timer.checkpoint("relaxation_" + STR(Ui) + "_" + STR(mui));
				_mod->relaxes(this->simP.mcS_, this->quiet, _timer.point("relaxation_" + STR(Ui) + "_" + STR(mui)));
				LOGINFO(_timer.point("relaxation_" + STR(Ui) + "_" + STR(mui)), "DQMC: relaxation ", 0);
			}
			END_CATCH_HANDLER("Failed to perform the saving... ", if (_mod) delete _mod; continue;);

			// save the configuration
			BEGIN_CATCH_HANDLER
			{
				if (this->simP.mcCheckSave_ == "date")
				{
					std::string time = prettyTime();
					_mod->saveCheckPoint(_mod->dir_->mainDir, "HS_" + time + ".h5");
				}
				else if (!this->simP.mcCheckSave_.empty())
					_mod->saveCheckPoint(_mod->dir_->mainDir, this->simP.mcCheckSave_);
				else
					_mod->saveCheckPoint(_mod->dir_->mainDir, "HS.h5");
			}
			catch (std::exception& e)
			{
				LOGINFO(LOG_TYPES::ERROR, "Couldn't setup the auxiliary fields to the standard path", 2);
				LOGINFO(LOG_TYPES::ERROR, e.what(), 2);
				if (_mod) delete _mod; continue;
			}

			// start the averaging
			BEGIN_CATCH_HANDLER
			{
				_timer.checkpoint("average_" + STR(Ui) + "_" + STR(mui));
				_mod->average(this->simP.mcS_, this->simP.mcC_, this->simP.mcA_, this->simP.mcB_,
							  this->quiet, _timer.point("average_" + STR(Ui) + "_" + STR(mui)));
				LOGINFO(_timer.point("average_" + STR(Ui) + "_" + STR(mui)), "DQMC: average ", 0);
			}
			END_CATCH_HANDLER("Failed to perform the averages... ", if (_mod) delete _mod; continue;);

			// clean
			if(_mod) delete _mod;
		}
	}
	END_CATCH_HANDLER(std::string(__FUNCTION__), ;);
}