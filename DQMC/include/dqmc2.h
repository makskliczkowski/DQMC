/***************************************
* Defines the general DQMC (SPIN 1/2).
* It is a base for further more
* complicated Hamiltonians
* development in a finite temperature.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/

#pragma once

#ifndef DQMC_H
	#include "dqmc.h"
#endif

#ifndef DQMC2_H
#define DQMC2_H

// ######################### G E N E R A L   S P I N   2 ############################

class DQMC2 : public DQMC<size_t(2)>
{
public:
	using matArray				=			std::array<arma::mat, spinNumber_>;
	using vecMatArray			=			std::array<v_1d<arma::mat>, spinNumber_>;
	enum SPINNUM
	{
		_UP_ = 0,
		_DN_ = 1
	};
	BEGIN_ENUM_INLINE(SPINNUM)
	{
		DECL_ENUM_ELEMENT(_UP_),
		DECL_ENUM_ELEMENT(_DN_)
	}
	END_ENUM_INLINE(SPINNUM, DQMC2);

	DQMC2()						=			default;
	DQMC2(double _T, std::shared_ptr<Lattice> _lat, uint _M, uint _M0, int _threadNum = 1)
		: DQMC<spinNumber_>(_T, _lat, _threadNum), M_(_M), M0_(_M0), p_(_M / _M0)
	{
		LOGINFO(std::string("Base DQMC spin-1/2 class is constructed."), LOG_TYPES::TRACE, 2);
	}
	virtual ~DQMC2() 
	{
		LOGINFO("Spin-1/2 DQMC is destroyed...", LOG_TYPES::INFO, 1);
	}
protected:
	uint M_						=			1;								// number of Trotter times

	// ################# O T H E R   F O R M U L A T I O N #################
	uint M0_					=			1;								// in ST - num of time subinterval, in QR num of stable multiplications
	uint p_						=			1;								// p = M / M0

	matArray G_;															// spatial Green's function
	matArray tmpG_;															// Green's function temporary
#ifdef DQMC_CAL_TIMES
	matArray Gtime_;
	std::string GtimeInf_;
#endif
	matArray IExp_;															// interaction exponential
	vecMatArray B_;															// imaginary time propagators
	vecMatArray Bcond_;														// imaginary time propagators - condensation of M0 multiplication - stable
	vecMatArray iB_;														// imaginary time propagators	

	// ############################ S A V E R S ############################
	virtual void saveGreensT(uint _step)									override;
	virtual void saveGreens(uint _step)										override;
};


// ##################################################################################

class DQMCavs2 : public virtual DQMCavs<2, double>
{
	using _T				=	double;
	using GREEN_TYPE		=	DQMCavs<2, double>::GREEN_TYPE;
	using SINGLE_PART_FUN	=	DQMCavs<2, double>::SINGLE_PART_FUN;
	using TWO_PARTS_FUN		=	DQMCavs<2, double>::TWO_PARTS_FUN;
public:

	virtual ~DQMCavs2()
	{
		LOGINFO("Destroying spin-1/2 averages for DQMC", LOG_TYPES::TRACE, 2);
	}

	DQMCavs2(std::shared_ptr<Lattice> _lat, int _M, const v_1d<double>* _t_nn = nullptr)
		: DQMCavs<2, double>(_lat, _M, _t_nn)
	{
		LOGINFO("Building DQMC SPIN-1/2 averages class", LOG_TYPES::INFO, 3);
	};

	// --- SINGLE ---
	virtual _T cal_Ek(SINGLE_PARTICLE_INPUT)			override
	{
		const auto neiNum	=	this->lat_->get_nn(_i);
		double Ek			=	0.0;
		for (int nei = 0; nei < neiNum; nei++)
		{
			const int whereNei	=	this->lat_->get_nn(_i, nei);
			Ek					+=	_g[DQMC2::_DN_](_i, whereNei);
			Ek					+=	_g[DQMC2::_DN_](whereNei, _i);
			Ek					+=	_g[DQMC2::_UP_](_i, whereNei);
			Ek					+=	_g[DQMC2::_UP_](whereNei, _i);
		}
		return _sign * ((this->t_nn_) ? (*this->t_nn_)[_i] * Ek : 1.0);
	}
	virtual _T cal_Occupation(SINGLE_PARTICLE_INPUT)	override
	{
		return (_sign * (1.0 - _g[DQMC2::_DN_](_i, _i)) + _sign * (1.0 - _g[DQMC2::_UP_](_i, _i)));
	}
	virtual _T cal_Mz2(SINGLE_PARTICLE_INPUT)			override
	{
		return _sign	* (	((1.0 - _g[DQMC2::_UP_](_i, _i)) * (1.0 - _g[DQMC2::_UP_](_i, _i))	)
						+	((1.0 - _g[DQMC2::_UP_](_i, _i)) * (_g[DQMC2::_UP_](_i, _i))		)
						-	((1.0 - _g[DQMC2::_UP_](_i, _i)) * (1.0 - _g[DQMC2::_DN_](_i, _i))	)
						-	((1.0 - _g[DQMC2::_DN_](_i, _i)) * (1.0 - _g[DQMC2::_UP_](_i, _i))	)
						+	((1.0 - _g[DQMC2::_DN_](_i, _i)) * (1.0 - _g[DQMC2::_DN_](_i, _i))	)
						+	((1.0 - _g[DQMC2::_DN_](_i, _i)) * (_g[DQMC2::_DN_](_i, _i))		));
	}
	virtual _T cal_My2(SINGLE_PARTICLE_INPUT)			override 
	{
		return 0;
	}
	virtual _T cal_Mx2(SINGLE_PARTICLE_INPUT)			override 
	{
		return		_sign * (1.0 - _g[DQMC2::_UP_](_i, _i)) * (_g[DQMC2::_DN_](_i, _i))
				+	_sign * (1.0 - _g[DQMC2::_DN_](_i, _i)) * (_g[DQMC2::_UP_](_i, _i));
	}

	// --- TWO ---
	virtual _T cal_Occupation_C(TWO_PARTICLES_INPUT)	override
	{
		return _sign * ((_g[DQMC2::_DN_](_j, _i) + _g[DQMC2::_UP_](_j, _i)));
	}
	virtual _T cal_Mz2_C(TWO_PARTICLES_INPUT)			override 
	{
		double delta_ij = (_i == _j) ? 1.0L : 0.0L;
		return _sign	* (	((1.0L		- _g[DQMC2::_UP_](_i, _i)) * (1.0L - _g[DQMC2::_UP_](_j, _j))	)
						+	((delta_ij	- _g[DQMC2::_UP_](_j, _i)) * (_g[DQMC2::_UP_](_i, _j))			)
						-	((1.0L		- _g[DQMC2::_UP_](_i, _i)) * (1.0L - _g[DQMC2::_DN_](_j, _j))	)
						-	((1.0L		- _g[DQMC2::_DN_](_i, _i)) * (1.0L - _g[DQMC2::_UP_](_j, _j))	)
						+	((1.0L		- _g[DQMC2::_DN_](_i, _i)) * (1.0L - _g[DQMC2::_DN_](_j, _j))	)
						+	((delta_ij	- _g[DQMC2::_DN_](_j, _i)) * (_g[DQMC2::_DN_](_i, _j))			));
	}

	// ########################## R E S E T E R S ##########################
public:
	
	///*
	//* @brief Resets all the averages
	//*/
	//void reset()										override {


	//}

	///*
	//* @brief Resets the Green's function in time
	//*/
	//void resetG()										override
	//{

	//}
};

#endif