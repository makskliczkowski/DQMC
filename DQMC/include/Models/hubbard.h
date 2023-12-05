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
	v_1d<double> t_;														// hopping integrals
	double U_					=			0.0;							// Hubbard U
	double mu_					=			0.0;							// chemical potential

	// ############# S I M U L A T I O N   P R O P E R T I E S #############
	bool REPULSIVE_				=			true;							// U > 0?
	double dtau_				=			1e-2;							// Trotter time step

	double lambda_				=			1.0;							// lambda parameter in HS transform
	v_1d<spinTuple_> gammaExp_;
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
			double _U			=	8.0, 
			double _dtau		=	0.05, 
			double _mu			=	0.0)
		: DQMC2(_T, _lat, _M, _M0), t_(_t), U_(_U), mu_(_mu), dtau_(_dtau)
	{
		this->setInfo();
		if (this->M_ % this->M0_ != 0)		
			throw std::runtime_error("Cannot have M0 times that do not divide M.");

		this->ran_				=			randomGen(DQMC_RANDOM_SEED ? DQMC_RANDOM_SEED : std::random_device{}());
		this->avs_				=			std::make_shared<DQMCavs2>(_lat, _M, &this->t_);

		// repulsiveness
		this->REPULSIVE_		=			(this->U_ > 0);

		// lambda couples to the auxiliary spins
		this->lambda_			=			std::acosh(std::exp((std::abs(this->U_) * this->dtau_) / 2.0));

		// calculate Gamma Exponents
		double expM				=			std::expm1(-2.0 * this->lambda_);									// spin * hsfield =  1
		double expP				=			std::expm1(2.0 * this->lambda_);									// spin * hsfield = -1
		this->gammaExp_			=			{{ expM, (this->REPULSIVE_) ? expP : expM }, { expP, expM }};		// [hsfield = 1, hsfield = -1]
		this->currentGamma_		=			&this->gammaExp_[0];
		this->fromScratchNum_	=			this->M0_;
		
		this->init();

		this->setHS(HS_CONF_TYPES::HIGH_T);
		this->calQuadratic();
		this->calInteracts();
		this->calPropagatB();
		for (uint i = 0; i < this->p_; i++)
			this->calPropagatBC(i);
		this->posNum_			=			0;
		this->negNum_			=			0;
	}
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

	// ########################### S E T T E R S ##########################
	void setHS(HS_CONF_TYPES _t)											override;
	void setDir(std::string _m)												override;
	void setInfo()															override;

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

};

#endif
