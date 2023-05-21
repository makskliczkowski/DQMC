/***************************************
* Defines the general DQMC class.
* It is a base for further more
* complicated Hamiltonians
* development in a finite temperature.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#pragma once

#ifndef LATTICE_H
	#include "../source/src/lattices.h"
#endif

#ifndef BINARY_H
	#include "../source/src/binary.h"
#endif

#include <mutex>
#include <complex>
#include <type_traits>
#include <shared_mutex>

#ifndef DQMC_AV_H
#define DQMC_AV_H

// #################################################################################

#define SINGLE_PARTICLE_INPUT	int _sign, uint _i,			 const GREEN_TYPE& _g
#define TWO_PARTICLES_INPUT		int _sign, uint _i, uint _j, const GREEN_TYPE& _g
#define DQMC_SINGLE_PARAM(name, type) type av_##name = 0;					    \
									  type sd_##name = 0
// #################################################################################

template <size_t _spinNum>
class DQMCavs 
{
public:
	using GREEN_TYPE			=	std::array<arma::Mat<double>, _spinNum>;
	using SINGLE_PART_FUN		=	std::add_pointer<std::complex<double>(SINGLE_PARTICLE_INPUT)>	::type;
	using TWO_PARTS_FUN			=	std::add_pointer<std::complex<double>(TWO_PARTICLES_INPUT)	>	::type;

	DQMC_SINGLE_PARAM(sign, double);

	int norm_					=	1;
	int normSign_				=	1;
	// ############################# F U N C T I O N S #############################
	
	virtual void reset()														= 0;
	virtual void normalize(int _avNum, int _normalization, double _avSign);

	// ########################### C A L C U L A T O R S ###########################
#define DQMC_AV_FUN1(x) virtual std::complex<double>##x(SINGLE_PARTICLE_INPUT)	= 0;
//#define DQMC_AV_FUN2(x) virtual std::complex<double>##x(TWO_PARTICLES_INPUT)	= 0;
#include "averageCalculatorSingle.include"
//#include "averageCalculatorCorrelations.include"
#undef DQMC_AV_FUN1
//#undef DQMC_AV_FUN2

	// ############################## M A P P I N G S ##############################
#define DQMC_AV_FUN1(x)	{#x, x},
	std::map<std::string, SINGLE_PART_FUN> calFun = {
#include "averageCalculatorSingle.include"
	};
//#define DQMC_AV_FUN2(x) {#x, x};
	//std::map<std::string, TWO_PARTS_FUN> calFun2 = {
//#include "averageCalculatorCorrelations.include"
	//}
#undef DQMC_AV_FUN1
//#undef DQMC_AV_FUN2
};

// #################################################################################

template<size_t _spinNum>
inline void DQMCavs<_spinNum>::normalize(int _avNum, int _normalization, double _avSign)
{
	this->norm_			=		_avNum * _normalization;
	this->av_sign		=		_avSign;
	this->normSign_		=		this->norm_ * this->av_sign;
}

#endif // !DQMC_AV_H