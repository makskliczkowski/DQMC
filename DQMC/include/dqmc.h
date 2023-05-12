#pragma once

/***************************************
* Defines the general DQMC class.
* It is a base for further Hamiltonian
* developments in finite temperature.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#ifndef LATTICE_H
	#include "../source/src/lattices.h"
#endif

#ifndef BINARY_H
	#include "../source/src/binary.h"
#endif

#include <mutex>
#include <shared_mutex>
#include <complex>

// ##########################################################################################################################################

#ifndef DQMC_H
#define DQMC_H

// #################################################################################

#define SINGLE_PARTICLE_INPUT	int _sign, uint _i,			 const GREEN_TYPE& _g
#define TWO_PARTICLES_INPUT		int _sign, uint _i, uint _j, const GREEN_TYPE& _g

template <size_t _spinNum>
class DQMCavs 
{
public:
	using GREEN_TYPE			=	std::array<arma::Mat<double>, _spinNum>;
	typedef SINGLE_PART_FUN		=	std::complex<double>(*CAL_1P)(SINGLE_PARTICLE_INPUT);
	typedef TWO_PARTS_FUN		=	std::complex<double>(*CAL_2P)(TWO_PARTICLES_INPUT);

	// ############################# F U N C T I O N S #############################
#define DQMC_AV_FUN1(x) virtual std::complex<double>##x(SINGLE_PARTICLE_INPUT)	= 0;
#define DQMC_AV_FUN2(x) virtual std::complex<double>##x(TWO_PARTICLES_INPUT)	= 0;
#include "averageCalculatorSingle.include"
//#include "averageCalculatorSingle.include"
#undef DQMC_AV_FUN1
#undef DQMC_AV_FUN2

	// ############################## M A P P I N G S ##############################
#define DQMC_AV_FUN1(x)	{#x, x},
std::map<std::string, SINGLE_PART_FUN> calFun = {
#include "averageCalculatorSingle.include"
}
#define DQMC_AV_FUN2(x) {#x, x};
#undef DQMC_AV_FUN1
#undef DQMC_AV_FUN2

};
// #################################################################################
struct DQMCdir
{
	std::string mainDir;
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

	// ################ C U R R E N T   P R O P E R T I E S ################
	uint tau_							=		0;							// current Trotter time
	int currentSign_					=		1;							// current sign of the HS configuration probability

	std::shared_ptr<Lattice> lat_;
	pBar pBar_;																// for printing out the progress
	// ######################### A V E R A G E S ###########################
	std::shared_ptr<DQMCavs<spinNum_>> avs_;

	// ############### P H Y S I C A L   P R O P E R T I E S ###############
	double T_							=		1;
	double beta_						=		1;

	// ############# S I M U L A T I O N   P R O P E R T I E S #############
	std::string info_;

	double proba_						=		0.0;
	uint fromScratchNum_				=		1;
	v_1d<int> configSign_;													// keeps track of the configuration signs

public:
	std::shared_ptr<DQMCdir> dir_;											// directories used in the simulation
	randomGen ran_;															// consistent quick random number generator
	std::string info_;														// information about the model

	virtual ~DQMC()					
	{ 
		LOGINFO("Base DQMC is destroyed...", LOG_TYPES::INFO, 3); 
	};

	// ###################### C A L C U L A T O R S ########################
	virtual void relaxes(uint MCs, bool _quiet = false)												= 0;
	virtual void average(uint MCs, uint corrTime, uint avNum, uint bootStraps, bool _quiet = false) = 0;

	// ########################## G E T T E R S ############################
	auto getInvTemperature()			const -> double						{ return this->beta_; };
	auto getTemperature()				const -> double						{ return this->T_; };
	auto getInfo()						const -> std::string				{ return this->info_; };
	auto getLat()						const -> std::shared_ptr<Lattice>	{ return this->lat_; };
	auto getDim()						const -> uint						{ return this->lat_->get_Dim(); };
	auto getNs()						const -> uint						{ return this->lat_->get_Ns(); };
	auto getBC()						const -> BoundaryConditions			{ return this->lat_->get_BC(); };
	auto getAverages()					const -> std::shared_ptr<DQMCavs<spinNumber_>>	
																			{ return this->avs_; };

	// ########################## S E T T E R S ############################
	virtual auto setHS(HS_CONF_TYPES _t)-> void								= 0;
	virtual auto setDir(std::string _m)	-> void								= 0;

	// ###################### C A L C U L A T O R S ########################
protected:
	virtual spinTuple_ calProba(uint _site)									= 0;
	virtual void calQuadratic()												= 0;
	virtual void calInteracts()												= 0;
	virtual void calPropagatB(uint _tau)									= 0;
	virtual void calPropagatB()												= 0;
	// GREENS
	virtual void calGreensFun(uint _tau)									= 0;

	virtual void avSingleStep(int _currI, int _sign)						= 0;
	virtual void eqSingleStep(int _site)									= 0;
	// ######################### E V O L U T I O N #########################
	virtual double sweepForward()											= 0;
	virtual double sweepBackward()											= 0;

	// ########################## U P D A T E R S ##########################
	virtual void updPropagatB(uint _site, uint _t)							= 0;
	virtual void updPropagatB(uint _site)									{ this->updPropagatB(this->tau_); };
	virtual void updInteracts(uint _site, uint _t)							= 0;
	// GREENS
	virtual void updEqlGreens(uint _site,
							  const std::array<double, spinNumber_>& p)		= 0;
	virtual void updNextGreen(uint _t)										= 0;
	virtual void updPrevGreen(uint _t)										= 0;

	void updInteracts(uint _site)											{ this->updInteracts(this->tau_); };
};

#endif