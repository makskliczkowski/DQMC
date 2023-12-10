#pragma once
#ifndef HUBBARD_H
#define HUBBARD_H

#ifndef DQMC2_H
	#include "../dqmc2.h"
#endif

#ifndef ALG_H
	#include "../../source/src/lin_alg.h"
#endif

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

/*
* @brief A basic Hubbard model class for DQMC simulation
*/
class Hubbard : public DQMC2
{
	// ############################# Q R ################################### 
	std::array<std::unique_ptr<algebra::UDT_QR<double>>, spinNumber_> udt_;
	
	// ############### P H Y S I C A L   P R O P E R T I E S ###############
	v_1d<double> t_;														// hopping integrals ('s for multiple Hubbard bands)
	v_1d<double> tt_;														// hopping integrals ('s for multiple Hubbard bands) - second nearest
	v_1d<double> U_;														// Hubbard U ('s for multiple Hubbard bands)
	v_1d<double> mu_;														// chemical potential ('s for multiple Hubbard bands)
	v_1d<bool> isRepulsive_;												// U > 0?
	
	// ############# S I M U L A T I O N   P R O P E R T I E S #############
	double dtau_				=			5e-2;							// Trotter time step

	// vector of parameters for the discrete transformation of n_up*n_down
	v_1d<double> lambda_;													// lambda parameter in HS transform

	// transformation _gammas
	v_2d<spinTuple_> gammaExp_;
	spinTuple_* currentGamma_;

public:
	~Hubbard()
	{
		LOGINFO("Deleting the DQMC Hubbard model class...", LOG_TYPES::INFO, 1);
	}
	Hubbard()					=			default;
	Hubbard(double _T, 
			std::shared_ptr<Lattice> _lat, 
			uint _M, 
			uint _M0,
			v_1d<double> _t, 
			v_1d<double> _U, 
			v_1d<double> _mu,
			double _dtau		=	0.05, 
			uint _bands			=	1,
			v_1d<double> _tt	=	{});
	void init()																override;

protected:
	// ####################### C A L C U L A T O R S #######################
	void calProba(uint _site)												override;
	void calQuadratic()														override;
	void calInteracts()														override;
	void calPropagatBC(uint _sec)											override;
	void calPropagatB(uint _tau)											override;
	void calPropagatB()														override;
	// GREENS
	void compareGreen(uint _tau, double _toll, bool _print)					override;
	void compareGreen()														override;
	void calGreensFun(uint _tau)											override;
	void calGreensFunC(uint _sec)											override;
#ifdef DQMC_CAL_TIMES
	void calGreensFunT()													override;
	void calGreensFunTHirsh()												override;
	void calGreensFunTHirshC()												override;
#endif
	void avSingleStep(int _currI, int _sign)								override;
	void avSingleStepUneq(int xx, int yy, int zz, int _i, int _j, int _s)	override;

	int eqSingleStep(int _site)												override;

	// ######################### E V O L U T I O N #########################
	double sweepForward()													override;
	void equalibrate(uint MCs			= 100, 
					 bool _quiet		= false, 
					 clk::time_point _t = NOW)								override;
	void averaging(	uint MCs			= 100, 
					uint corrTime		= 1, 
					uint avNum			= 50,
					uint buckets		= 50, 
					bool _quiet			= false,
					clk::time_point _t	= NOW)								override;
	// HELPING
	auto calGamma(uint _site)				->								void;
	auto calDelta()							->								spinTuple_;

	// ########################## U P D A T E R S #########################
	void updPropagatB(uint _site, uint _t)									override;
	void updInteracts(uint _site, uint _t)									override;

	// GREENS
	void updEqlGreens(uint _site, const spinTuple_& p)						override;
	void updNextGreen(uint _t)												override;
	void updPrevGreen(uint _t)												override;
	void updGreenStep(uint _t)												override;

	// ############################ S A V E R S ############################
	void saveGreens(uint _step)												override;

	// ########################### S E T T E R S ##########################
public:
	void setHS(HS_CONF_TYPES _t)											override;
	void setDir(std::string _m)												override;
	void setInfo()															override;
};

#endif
