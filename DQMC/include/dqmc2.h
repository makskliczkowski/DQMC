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

// ############################################### G E N E R A L   S P I N   2 ####################################################

class DQMC2 : public DQMC<2>
{
public:
	using matArray = std::array<arma::mat, spinNumber_>;
	using vecMatArray = std::array<v_1d<arma::mat>, spinNumber_>;
protected:
	enum SPINNUM
	{
		_DN_ = 0,
		_UP_ = 1
	};
	BEGIN_ENUM_INLINE(SPINNUM)
	{
		DECL_ENUM_ELEMENT(_DN_),
			DECL_ENUM_ELEMENT(_UP_)
	}
	END_ENUM_INLINE(SPINNUM, DQMC2);

	matArray G_;															// spatial Green's function
#ifdef DQMC_CAL_TIMES
	matArray Gtime_;
#endif
	matArray IExp_;															// interaction exponential
	vecMatArray B_;															// imaginary time propagators
	vecMatArray Bcond_;														// imaginary time propagators - condensation of M0 multiplication - stable
	vecMatArray iB_;														// imaginary time propagators	

	// ############################ S A V E R S ############################
	virtual void saveGreensT(uint _step)									override;
	virtual void saveGreens(uint _step)										override;
};

#endif