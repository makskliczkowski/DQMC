/***************************************
* Defines the general DQMC class.
* It is a base for further more 
* complicated Hamiltonians
* development in a finite temperature.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#pragma once

#ifndef DQMC_AV_H
	#include "dqmc_av.h"
#endif

#ifndef DQMC_H
#define DQMC_H


// ######################### EXISTING MODELS ############################
enum MY_MODELS {													 // #
	HUBBARD_M														 // #
};																	 // #
BEGIN_ENUM(MY_MODELS)												 // #
{																	 // #
	DECL_ENUM_ELEMENT(HUBBARD_M)									 // #
}																	 // #
END_ENUM(MY_MODELS)                                                  // #	
// ######################################################################

// ######################### SAVING EXTENSIONS ##########################
enum HAM_SAVE_EXT {													 // #
	dat, h5															 // #
};																	 // #
BEGIN_ENUM(HAM_SAVE_EXT)											 // #
{																	 // #
	DECL_ENUM_ELEMENT(dat), DECL_ENUM_ELEMENT(h5)					 // #
}																	 // #
END_ENUM(HAM_SAVE_EXT)                                               // #	
// ######################################################################


// #################################################################################
struct DQMCdir
{
	std::string mainDir;
	std::string equalTimeDir;
	std::string unequalTimeDir;

	// Green's functions
	std::string equalGDir;
	std::string unequalGDir;
	std::string randomSampleStr;

	// correlations
	std::string equalCorrDir;
	std::string uneqalCorrDir;

	void setup(const std::string& _mainDir)
	{
		mainDir = _mainDir;
	}

	void createDQMCDirs(randomGen& _ran) {
		fs::create_directories(mainDir);
		fs::create_directories(equalTimeDir);
		fs::create_directories(unequalTimeDir);

		fs::create_directories(equalGDir);
		fs::create_directories(unequalGDir);

		fs::create_directories(equalCorrDir);
		fs::create_directories(uneqalCorrDir);
		
		const auto token	= clk::now().time_since_epoch().count();
		randomSampleStr		= STR(token % _ran.randomInt<int>(1, 1e4));
	}
};
// #################################################################################

/*
* @brief General Determinant Quantum Monte Carlo class
*/
template <size_t spinNum_>
class DQMC 
{
public:
	// initial HS fields configurations
	enum HS_CONF_TYPES 
	{
		HIGH_T,
		LOW_T
	};

	const static size_t spinNumber_		=		spinNum_;
	using spinTuple_					=		typename std::array<double, spinNum_>;
protected:
	using MutexType						=		std::shared_timed_mutex;
	using ReadLock						=		std::shared_lock<MutexType>;
	using WriteLock						=		std::unique_lock<MutexType>;
	mutable MutexType Mutex;
	uint threadNum_						=		1;
	uint Ns_							=		1;
	u64 posNum_							=		0;
	u64 negNum_							=		0;

	// ################ C U R R E N T   P R O P E R T I E S ################
	uint tau_							=		0;							// current Trotter time
	int currentSign_					=		1;							// current sign of the HS configuration probability
	int currentWarmups_					=		1;
	int currentAverages_				=		1;

	std::shared_ptr<Lattice> lat_;
	pBar pBar_;																// for printing out the progress
	// ######################### A V E R A G E S ###########################
	std::shared_ptr<DQMCavs<spinNum_, double>> avs_;

	// ############### P H Y S I C A L   P R O P E R T I E S ###############
	double T_							=		1;
	double beta_						=		1;
	arma::mat TExp_;														// hopping exponential

	// ############# S I M U L A T I O N   P R O P E R T I E S #############

	spinTuple_ currentProba_;
	double proba_						=		0.0;
	int configSign_						=		1;
	uint fromScratchNum_				=		1;
	v_1d<int> configSigns_;													// keeps track of the configuration signs

public:
	std::shared_ptr<DQMCdir> dir_;											// directories used in the simulation
	randomGen ran_;															// consistent quick random number generator
	std::string info_;														// information about the model

	DQMC()								=		default;
	DQMC(double _T, std::shared_ptr<Lattice> _lat, uint _threadNum = 1) 
		: threadNum_(_threadNum), Ns_(_lat->get_Ns()), lat_(_lat), T_(_T), beta_(1.0 / _T)
	{
		this->dir_						=		std::make_shared<DQMCdir>();
		LOGINFO("Base DQMC class is constructed.", LOG_TYPES::TRACE, 1);
		this->posNum_					=		0;
		this->negNum_					=		0;
	}

	virtual ~DQMC()					
	{ 
		LOGINFO("Base DQMC is destroyed...", LOG_TYPES::INFO, 1); 
	};

	virtual void init()														= 0;

	// ###################### C A L C U L A T O R S ########################
	virtual void relaxes(uint MCs, bool _quiet = false);
	virtual void average(uint MCs, uint corrTime, uint avNum, uint bootStraps, bool _quiet = false);

	// ########################## G E T T E R S ############################
	auto getInvTemperature()			const -> double						{ return this->beta_; };
	auto getTemperature()				const -> double						{ return this->T_; };
	auto getInfo()						const -> std::string				{ return this->info_; };
	auto getLat()						const -> std::shared_ptr<Lattice>	{ return this->lat_; };
	auto getDim()						const -> uint						{ return this->lat_->get_Dim(); };
	auto getNs()						const -> uint						{ return this->lat_->get_Ns(); };
	auto getBC()						const -> BoundaryConditions			{ return this->lat_->get_BC(); };
	auto getAverages()					const -> std::shared_ptr<DQMCavs<spinNumber_, double>>	
																			{ return this->avs_; };

	// ########################## S E T T E R S ############################
	virtual auto setHS(HS_CONF_TYPES _t)-> void								= 0;
	virtual auto setDir(std::string _m)	-> void								= 0;
	virtual auto setInfo()				-> void								= 0;
	// ###################### C A L C U L A T O R S ########################
protected:
	virtual void calProba(uint _site)										= 0;
	virtual void calQuadratic()												= 0;
	virtual void calInteracts()												= 0;
	virtual void calPropagatB(uint _tau)									= 0;
	virtual void calPropagatBC(uint _sec)									= 0;
	virtual void calPropagatB()												= 0;
	// GREENS
	virtual void compareGreen(uint _tau, double _toll, bool _print)			= 0;
	virtual void compareGreen()												= 0;
	virtual void calGreensFun(uint _tau)									= 0;
	virtual void calGreensFunC(uint _sec)									= 0;

#ifdef DQMC_CAL_TIMES
	virtual void calGreensFunT()											= 0;
	virtual void calGreensFunTHirsh()										= 0;
	virtual void calGreensFunTHirshC()										= 0;
#endif

	virtual void avSingleStep(int _currI, int _sign)						= 0;
	virtual void avSingleStepUneq(int, int, int, int _i, int _j, int _s)	= 0;
	virtual int eqSingleStep(int _site)										= 0;
	// ######################### E V O L U T I O N #########################
	virtual int sweepLattice();
	virtual double sweepForward()											= 0;
	//virtual double sweepBackward()										= 0;

	virtual void equalibrate(uint MCs, bool _quiet = false)					= 0;
	virtual void averaging(uint MCs, uint corrTime, uint avNum,
						 uint bootStraps, bool _quiet = false)				= 0;

	// ########################## U P D A T E R S ##########################
	virtual void updPropagatB(uint _site, uint _t)							= 0;
	virtual void updPropagatB(uint _site)									{ this->updPropagatB(_site, this->tau_); };
	virtual void updInteracts(uint _site, uint _t)							= 0;
	virtual void updInteracts(uint _site)									{ this->updInteracts(_site, this->tau_); };

	// GREENS
	virtual void updEqlGreens(uint _site, const spinTuple_& p)				= 0;
	virtual void updNextGreen(uint _t)										= 0;
	virtual void updPrevGreen(uint _t)										= 0;
	virtual void updGreenStep(uint _t)										= 0;
	
	// ############################ S A V E R S ############################
public:
	virtual void saveAverages()												= 0;
	virtual void saveCorrelations()											= 0;
	virtual void saveGreensT(uint _step)									= 0;
	virtual void saveGreens(uint _step)										= 0;
};

// ################################################## C A L C U L A T O R S #######################################################

/*
* @brief A function to sweep all the auxliary Ising fields for a given time configuration in the model
* @returns sign of the configuration
*/
template<size_t spinNum_>
inline int DQMC<spinNum_>::sweepLattice()
{
	int _sign = 1;
	for (int _site = 0; _site < this->Ns_; _site++)
	{
		auto sign = this->eqSingleStep(_site);
		if (sign < 0)
			_sign = -1;
	}
	return _sign;
}

// #################################################### E V O L U T O R S #########################################################

template<size_t spinNum_>
inline void DQMC<spinNum_>::relaxes(uint MCs, bool _quiet)
{
	auto start				=		std::chrono::high_resolution_clock::now();											// starting timer for averages
	this->currentWarmups_	=		MCs;
	this->equalibrate(MCs, _quiet);
	if (!_quiet && MCs != 1) {
#pragma omp critical
		LOGINFO("For " + this->getInfo() + " relaxation taken: " + TMS(start) + ". With sign: " + STRP(double(posNum_ - negNum_) / double(posNum_ + negNum_), 4), LOG_TYPES::TIME, 2);
		LOGINFO(LOG_TYPES::TRACE, 2);
	}
}

template<size_t spinNum_>
inline void DQMC<spinNum_>::average(uint MCs, uint corrTime, uint avNum, uint bootStraps, bool _quiet)
{
	auto start = std::chrono::high_resolution_clock::now();											// starting timer for averages
	this->currentAverages_	=		MCs;
	this->averaging(MCs, corrTime, avNum, bootStraps, _quiet);

	if (!_quiet) {
#pragma omp critical
		LOGINFO("For " + this->getInfo() + " averages taken: " + TMS(start) + ". With sign: " + STRP(double(posNum_ - negNum_) / double(posNum_ + negNum_), 4), LOG_TYPES::TIME, 2);
		LOGINFO("Average Onsite Occupation = " + STRP(this->avs_->av_Occupation, 4)	, LOG_TYPES::TIME, 3);
		LOGINFO("Average Onsite Magnetization = " + STRP(this->avs_->av_Mz2, 4)		, LOG_TYPES::TIME, 3);
		LOGINFO(LOG_TYPES::TRACE, 2);
	}
}

#endif