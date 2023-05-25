#pragma once
#ifndef HUBBARD_H
#define HUBBARD_H

#ifndef DQMC2_H
	#include "../dqmc2.h"
#endif

#ifndef ALG_H
	#include "../../source/src/lin_alg.h"
#endif

class Hubbard : public virtual DQMC2
{
	// ############################# Q R ################################### 
	std::array<std::unique_ptr<algebra::UDT<double>>, spinNumber_> udt_;
	spinTuple_ currentGamma_;
	
	// ############### P H Y S I C A L   P R O P E R T I E S ###############
	v_1d<double> t_;														// hopping integrals
	double U_					=			0.0;							// Hubbard U
	double mu_					=			0.0;							// chemical potential

	// ############# S I M U L A T I O N   P R O P E R T I E S #############
	bool REPULSIVE_				=			true;							// U > 0?
	double dtau_				=			1e-2;							// Trotter time step

	double lambda_				=			1.0;							// lambda parameter in HS transform
	arma::Mat<int> HSFields_;												// Hubbard-Stratonovich fields
	v_1d<spinTuple_> gammaExp_;

public:
	Hubbard()					=			default;
	Hubbard(double _T, std::shared_ptr<Lattice> _lat, uint _M, uint _M0,
		v_1d<double> _t, double _U, double _dtau)
		: t_(_t), U_(_U), dtau_(_dtau), DQMC2(_T, _lat, _M, _M0)
	{
		this->setInfo();
		this->init();

		this->ran_				=			randomGen();
		this->avs_				=			std::make_shared<DQMCavs2>(_lat, _M);
		this->avs_->reset();

		// lambda
		this->REPULSIVE_		=			(this->U_ > 0);
		// lambda couples to the auxiliary spins
		this->lambda_			=			(this->REPULSIVE_) ? std::acosh(exp((this->U_ * this->dtau_) / 2.0)) : std::acosh(exp((-this->U_ * this->dtau_) * 0.5));

		// calculate Gamma Exponents
		auto expM				=			std::expm1(-2.0 * this->lambda_);
		auto expP = std::expm1(2.0 * this->lambda_);
		this->gammaExp_ = {{ expM, expP },{ expP, expM }};

		this->fromScratchNum_	=			this->M0_;
	}
	void init()																override;

protected:
	// ####################### C A L C U L A T O R S #######################
	spinTuple_ calProba(uint _site)											override;
	void calQuadratic()														override;
	void calInteracts()														override;
	void calPropagatBC(uint _sec)											override;
	void calPropagatB(uint _tau)											override;
	void calPropagatB()														override;
	// GREENS
	void compareGreen(uint _tau, double _toll, bool _print)					override;
	void calGreensFun(uint _tau)											override;
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
	//double sweepBackward()													override;

	void equalibrate(uint MCs, bool _quiet = false)							override;
	void averaging(uint MCs, uint corrTime, uint avNum,
				 uint bootStraps, bool _quiet = false)						override;
	// HELPING
	auto calGamma(uint _site)				->								void;
	auto calDelta()							->								spinTuple_;

	// ########################### S E T T E R S ##########################
	void setHS(HS_CONF_TYPES _t)											override;
	void setDir(std::string _m)												override;
	void setInfo()															override;

	// ########################## U P D A T E R S #########################
	void updPropagatB(uint _site, uint _t)									override;
	void updInteracts(uint _site, uint _t)									override;

	// Greens
	void updEqlGreens(uint _site, const spinTuple_& p)						override;
	void updNextGreen(uint _t)												override;
	void updPrevGreen(uint _t)												override;
	void updGreenStep(uint _t)												override;
};

#endif
