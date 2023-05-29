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


#define DQMC_SAVE_H5
#define DQMC_USE_HIRSH
#define DQMC_CAL_TIMES
#ifdef DQMC_CAL_TIMES
	#define DQMC_CAL_TIMES_ALL
#endif
// ##########################################################################################

#define SINGLE_PARTICLE_INPUT	int _sign, uint _i,			 const GREEN_TYPE& _g
#define TWO_PARTICLES_INPUT		int _sign, uint _i, uint _j, const GREEN_TYPE& _g
#define INVOKE_SINGLE_PARTICLE_CAL(CLS, X)	CLS->calOneSite(_sign, _currI, this->G_, #X, CLS->av_##X, CLS->sd_##X);
#define INVOKE_TWO_PARTICLE_CAL(CLS,X,x,y,z)CLS->calTwoSite(_sign, _currI, _currJ, this->G_, #X, CLS->avC_##X, x, y, z);

#define SINGLE_PARTICLE_NORM(nam, norm)		this->av_##nam /=	norm;						\
											this->sd_##nam =	variance(this->sd_##nam, this->av_##nam, norm)
#define SINGLE_PARTICLE_PARAM(name, type)	type av_##name = 0;								\
											type sd_##name = 0


#define TWO_PARTICLE_NORM(nam, x, y, z, nor)this->avC_##nam[x][y][z] /=	nor;						
#define TWO_PARTICLE_PARAM(name, type)		v_3d<type> avC_##name;							\
											v_3d<type> sdC_##name					

// ##########################################################################################

template <size_t _spinNum, typename _retT>
class DQMCavs 
{
protected:
	std::shared_ptr<Lattice> lat_;
public:
	DQMCavs(std::shared_ptr<Lattice> _lat, int _M)
		: lat_(_lat)
	{
		LOGINFO("Building DQMC base averages class", LOG_TYPES::INFO, 2);
		auto [x_num, y_num, z_num] = _lat->getNumElems();
		this->reset();
#ifdef DQMC_CAL_TIMES
		this->normM_ = arma::vec(_M, arma::fill::zeros);
#endif

#ifdef DQMC_CAL_TIMES
		for (int _SPIN_ = 0; _SPIN_ < this->av_GTimeDiff_.size(); _SPIN_++)
		{
			this->av_GTimeDiff_[_SPIN_] = v_1d<arma::mat>(_M, arma::zeros(x_num, y_num));
			this->sd_GTimeDiff_[_SPIN_] = v_1d<arma::mat>(_M, arma::zeros(x_num, y_num));
			for (int tau1 = 0; tau1 < _M; tau1++) {
#ifdef DQMC_CAL_TIMES_ALL
				for (int tau2 = 0; tau2 < _M; tau2++) {
#else
				for (int tau2 = 0; tau2 <= tau1; tau2++) {
#endif
					auto tim = (tau1 - tau2);
					if (tim < 0)
						tim += _M;
					this->normM_ += 1.0;
				}
			}
#endif
		}
	}

public:
	using GREEN_TYPE			=	std::array<arma::Mat<double>, _spinNum>;
	typedef std::function<_retT(SINGLE_PARTICLE_INPUT)>								SINGLE_PART_FUN; // (*_retT)(SINGLE_PARTICLE_INPUT); //std::add_pointer<_retT(SINGLE_PARTICLE_INPUT)>	::type;
	typedef std::function<_retT(TWO_PARTICLES_INPUT)>								TWO_PARTS_FUN;			// =	std::add_pointer<_retT(TWO_PARTICLES_INPUT)>	::type;

	SINGLE_PARTICLE_PARAM(sign, double);
	std::array<v_1d<arma::mat>, _spinNum>	av_GTimeDiff_;
	std::array<v_1d<arma::mat>, _spinNum>	sd_GTimeDiff_;

	int norm_					=	1;
	int normSign_				=	1;
	arma::vec normM_;
	// ################################## F U N C T I O N S #################################
	
	virtual void reset();
	virtual void resetG();
	virtual void normalize(int _avNum, int _normalization, double _avSign);
	virtual void normalizeG();
	void calOneSite(SINGLE_PARTICLE_INPUT, const std::string& choice, _retT& av, _retT& stdev);
	void calTwoSite(TWO_PARTICLES_INPUT, const std::string& choice, v_3d<_retT>& av, int x, int y, int z);
	
	// ################################ C A L C U L A T O R S ###############################
	// --- SINGLE ---
#define DQMC_AV_FUN1(x) virtual _retT cal_##x (SINGLE_PARTICLE_INPUT)	{ return 0;	};		\
						SINGLE_PARTICLE_PARAM( x, double);

#include "averageCalculatorSingle.include"
#undef DQMC_AV_FUN1
	// --- TWO ---
#define DQMC_AV_FUN2(x) virtual _retT cal_##x##_C(TWO_PARTICLES_INPUT)	{ return 0;	};		\
						TWO_PARTICLE_PARAM( x, double);
#include "averageCalculatorCorrelations.include"
#undef DQMC_AV_FUN2

	// #################################### M A P P I N G S #################################
	// --- SINGLE ---
#define DQMC_AV_FUN1(x)	{ #x ,	[&](SINGLE_PARTICLE_INPUT)	{ return cal_##x (_sign, _i, _g); }},
	std::map<std::string, SINGLE_PART_FUN>	calFun = {
#include "averageCalculatorSingle.include"
	};
#undef DQMC_AV_FUN1
	// --- TWO ---
#define DQMC_AV_FUN2(x) {#x ,	[&](TWO_PARTICLES_INPUT)	{ return cal_##x##_C(_sign, _i, _j, _g); }},
	std::map<std::string, TWO_PARTS_FUN>	calFun2 = {
#include "averageCalculatorCorrelations.include"
	};
#undef DQMC_AV_FUN2
};

// ##########################################################################################

/*
* @brief Normalise all the averages taken during simulation.
* @param _avNum number of averages taken
* @param _normalization normalization comming from the lattice properties
* @param _avSign average sign from the simulation
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::normalize(int _avNum, int _normalization, double _avSign)
{
	this->norm_			=		_avNum * _normalization;
	this->av_sign		=		_avSign;
	this->normSign_		=		this->norm_ * this->av_sign;
	
	// OCCUPATION
	SINGLE_PARTICLE_NORM(Occupation, this->normSign_);

	// Mz2
	SINGLE_PARTICLE_NORM(Mz2, this->normSign_);

	// My2
	SINGLE_PARTICLE_NORM(My2, this->normSign_);

	// Mx2
	SINGLE_PARTICLE_NORM(Mx2, this->normSign_);

	// Kinetic energy
	SINGLE_PARTICLE_NORM(Ek, this->normSign_);

	// ----------------- C o r r e l a t i o n s -----------------
	auto [x_num, y_num, z_num] = this->lat_->getNumElems();
	for(int x = 0; x < x_num; x++)
		for(int y = 0; y < y_num; y++)
			for (int z = 0; z < z_num; z++)
			{
				TWO_PARTICLE_NORM(Mz2, x, y, z, this->normSign_);
				//TWO_PARTICLE_NORM(My2, x, y, z, this->normSign_);
				//TWO_PARTICLE_NORM(Mx2, x, y, z, this->normSign_);
				this->avC_Occupation[x][y][z] = this->lat_->get_Ns() * this->avC_Occupation[x][y][z] / this->normSign_;
			}
}

// ##########################################################################################

/*
* @brief Normalizes the Green's given the model parameters and the lattice
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::normalizeG()
{
	// time number
	const auto _M		=	this->av_GTimeDiff_.size();
	auto [xx, yy, zz]	=	this->lat_->getNumElems();
	
	for (int _SPIN_ = 0; _SPIN_ < _spinNum; _SPIN_++) {
		for (int tau = 0; tau < _M; tau++) {
			auto norm = -DQMC_BUCKET_NUM * this->normM_(tau);
			for (int x = 0; x < xx; x++) {
				for (int y = 0; y < yy; y++) {
					const auto norm2 = norm * lat_->getNorm(x, y, 0);
					this->av_GTimeDiff_[_SPIN_][tau](x, y) /= norm2;
					this->sd_GTimeDiff_[_SPIN_][tau](x, y) /= norm2;
				}
			}
		}
	}
}

// ##########################################################################################

template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::calOneSite(SINGLE_PARTICLE_INPUT, const std::string& choice, _retT& av, _retT& stdev)
{
	auto _val		=		this->calFun[choice](_sign, _i, _g);
	av				+=		_val;
	stdev			+=		_val * _val;
};

// ##########################################################################################

template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::calTwoSite(TWO_PARTICLES_INPUT, const std::string& choice, v_3d<_retT>& av, int x, int y, int z)
{
	auto _val		=		this->calFun2[choice](_sign, _i, _j, _g);
	av[x][y][z]		+=		_val;
};

// ##########################################################################################

template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::reset() 
{
		auto [x_num, y_num, z_num]		=		this->lat_->getNumElems();

		this->av_Ek = 0;
		this->sd_Ek = 0;
		
		this->av_Occupation				=		0;
		this->sd_Occupation				=		0;
		
		this->av_Mz2					=		0;
		this->sd_Mz2					=		0;

		this->av_Mx2					=		0;
		this->sd_Mx2					=		0;

		this->av_My2					=		0;
		this->sd_My2					=		0;

		// correlations
		this->avC_Mz2					=		v_3d<_retT>(x_num, v_2d<_retT>(y_num, v_1d<_retT>(z_num, 0.0)));
		this->avC_Occupation			=		v_3d<_retT>(x_num, v_2d<_retT>(y_num, v_1d<_retT>(z_num, 0.0)));
		
#ifdef DQMC_CAL_TIMES
		this->resetG();
#endif
}

// ##########################################################################################

template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::resetG()
{
	for (int _SPIN_ = 0; _SPIN_ < this->av_GTimeDiff_.size(); _SPIN_++)
		for (int tau = 0; tau < this->av_GTimeDiff_[_SPIN_].size(); tau++) {
			this->av_GTimeDiff_[_SPIN_][tau].zeros();
			this->sd_GTimeDiff_[_SPIN_][tau].zeros();
		}
}
#endif // !DQMC_AV_H