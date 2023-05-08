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

// ##########################################################################################################################################

#ifndef DQMC_H
#define DQMC_H

// ###########################################################################
struct DQMCavs 
{

};
// ###########################################################################
struct DQMCdir
{
	std::string mainDir;
};
// ###########################################################################

/*
* @brief General Determinant Quantum Monte Carlo class
*/
class DQMC 
{
protected:
	using MutexType						=		std::shared_timed_mutex;
	using ReadLock						=		std::shared_lock<MutexType>;
	using WriteLock						=		std::unique_lock<MutexType>;
	mutable MutexType Mutex;
	uint threadNum_						=		1;
	uint Ns_							=		1;

	std::shared_ptr<Lattice> lat_;
	pBar pBar_;																// for printing out the progress
	// ######################### A V E R A G E S ###########################
	std::shared_ptr<DQMCavs> avs_;

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
	virtual void relaxes(uint MCs, bool _quiet = false) = 0;
	virtual void average(uint MCs, uint corrTime, uint avNum, uint bootStraps, bool _quiet = false) = 0;

	// ########################## G E T T E R S ############################
	auto getInvTemperature()			const -> double						{ return this->beta_; };
	auto getTemperature()				const -> double						{ return this->T_; };
	auto getAverages()					const -> std::shared_ptr<DQMCavs>	{ return this->avs_; };
	auto getInfo()						const -> std::string				{ return this->info_; };
	auto getLat()						const -> std::shared_ptr<Lattice>	{ return this->lat_; };
	auto getDim()						const -> uint						{ return this->lat_->get_Dim(); };
	auto getNs()						const -> uint						{ return this->lat_->get_Ns(); };
	auto getBC()						const -> BoundaryConditions			{ return this->lat_->get_BC(); };

	// ########################## S E T T E R S ############################
	virtual auto setHS()				-> void								= 0;
	virtual auto setDir(std::string _m)	-> void								= 0;
};

#endif