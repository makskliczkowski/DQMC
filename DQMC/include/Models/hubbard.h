#pragma once
#ifndef HUBBARD_H
#define HUBBARD_H

#ifndef DQMC_H
	#include "../dqmc.h"
#endif

#ifndef ALG_H
	#include "../../source/src/lin_alg.h"
#endif

//#define SAVE_CONF


//namespace hubbard {
//	using namespace arma;
//
//	struct HubbardParams {
//		int dim;
//		double beta;
//		double mu;
//		double U;
//		int Lx;
//		int Ly;
//		int Lz;
//		int M;
//		int M0;
//		int p;
//		double dtau;
//		// constructor
//		HubbardParams(int dim, double beta, double mu, double U, int Lx, int Ly, int Lz, double dtau, int M, int M0, int p) : dim(dim), beta(beta), mu(mu), U(U),
//			Lx(Lx), Ly(Ly), Lz(Lz), M(M), M0(M0), p(p), dtau(dtau) {};
//	};




	//struct directories : public general_directories {

	//	std::string token = "";

	//	std::string fourier_dir = "";
	//	std::string params_dir = "";
	//	std::string conf_dir = "";
	//	std::string greens_dir = "";
	//	std::string time_greens_dir = "";

	//	// filenames
	//	std::string nameFouriers = "";							//
	//	std::string nameFouriersTime = "";						//
	//	std::string nameNormal = "";								//
	//	std::string nameNormalTime = "";							//
	//	std::string nameGreens = "";								//
	//	std::string nameGreensTime = "";							//
	//	std::string nameGreensTimeH5 = "";

	//	// configuration directories
	//	std::string neg_dir = "";								// directory for negative configurations
	//	std::string neg_log = "";								// the name of negative sign log file
	//	std::string pos_dir = "";								// directory for positive configurations
	//	std::string pos_log = "";								// the name of positive sign log file

	//	directories() = default;

	//	void setFileNames() {
	//		this->nameFouriers = fourier_dir + "fouriers_" + this->token + "_" + info + ".dat";
	//		this->nameFouriersTime = fourier_dir + "times" + kPS + "fouriersTime_" + this->token + "_" + info + ".dat";
	//		this->nameNormal = params_dir + "parameters_" + this->token + "_" + info + ".dat";
	//		this->nameNormalTime = params_dir + "times" + kPS + "parametersTime_" + this->token + "_" + info + ".dat";
	//		this->nameGreens = "greens_" + this->token + "_" + info + ".dat";
	//		this->nameGreensTime = "greensTime_" + this->token + "_" + info + ".dat";
	//		this->nameGreensTimeH5 = "greensTime_" + this->token + "_" + info + ".h5";
	//	};
	//};


class Hubbard : public DQMC<2>
{
protected:
	enum SPINNUM
	{
		_DN_					=			0,
		_UP_					=			1
	};
	
	// ############################# Q R ################################### 
	std::unique_ptr<algebra::UDT<double>> udtUP, udtDown;
	spinTuple_ currentGamma_;
	
	// ############### P H Y S I C A L   P R O P E R T I E S ###############
	v_1d<double> t_;														// hopping integrals
	double U_					=			0.0;							// Hubbard U
	double mu_					=			0.0;							// chemical potential

	// ############# S I M U L A T I O N   P R O P E R T I E S #############
	bool REPULSIVE_				=			true;							// U > 0?
	double dtau_				=			1e-2;							// Trotter time step
	uint M_						=			1;								// number of Trotter times

	arma::mat TExp_;														// hopping exponential
	std::array<arma::mat		, spinNumber_> IExp_;						// interaction exponential
	std::array<v_1d<arma::mat>	, spinNumber_> B_;							// imaginary time propagators	
	std::array<v_1d<arma::mat>	, spinNumber_> iB_;							// imaginary time propagators	
	std::array<arma::mat		, spinNumber_> G_;							// spatial Green's function

	double lambda_				=			1.0;							// lambda parameter in HS transform
	arma::Mat<int> HSFields_;												// Hubbard-Stratonovich fields
	v_1d<spinTuple_> gammaExp_;

	// ################# O T H E R   F O R M U L A T I O N #################
	uint M0_					=			1;								// in ST - num of time subinterval, in QR num of stable multiplications
	uint p_						=			1;								// p = M / M0

protected:
	// ####################### C A L C U L A T O R S #######################
	void calQuadratic()														override;
	void calInteracts()														override;
	void calPropagatB()														override;
	void calPropagatB(uint _tau)											override;
	spinTuple_ calProba(uint _site)											override;

	auto calGamma(uint _site)				->								void;
	auto calDelta()							->								spinTuple_;

	// ########################### S E T T E R S ##########################
	void setHS(HS_CONF_TYPES _t)											override;
	void setDir(std::string _m)												override;
	
	// ########################## U P D A T E R S #########################
	void updPropagatB(uint _site, uint _t)									override;
	void updInteracts(uint _site, uint _t)									override;
};









	/*
	* @brief the main class for the Hubbard model
	*/
	class HubbardModel : public LatticeModel {
	protected:
		// -------------------------- ALGORITHM CONVERGENCE PARAMETERS
		using sinOpType = double (*)(int, int, const mat&, const mat&);
		//using sinOpType = std::function<double(int, int, const mat&, const mat&)>;


		//double probability;
		int from_scratch;																				// number of Trotter times for Green to be calculated from scratch
		int config_sign;																				// keep track of the configuration sign
		bool equalibrate;																				// if we shall equalibrate stuff now
		long int pos_num;																				// helps with number of positive signs
		long int neg_num;																				// helps with number of negative signs

		// -------------------------- INITIAL PHYSICAL PARAMETERS
		//v_1d<double> t;																					// hopping integral vector
		//int dim;																						// dimension
		//double U;																						// Coulomb force strength
		//double mu;																						// chemical potential

		// -------------------------- SUZUKI - TROTTER RELATED PARAMETERS
		//int M;																							// number of Trotter times
		//int current_time;																				// current Trotter time
		//double dtau;																					// Trotter time step

		// -------------------------- TRANSFORMATION RELATED PARAMETERS
		//arma::mat hsFields;																				// Hubbard - Stratonovich fields - first time then field
		//std::pair<double, double> gammaExp0;															// precalculated exponent of gammas
		//std::pair<double, double> gammaExp1;															// precalculated exponent of gammas
		//double lambda;																					// lambda parameter in HS transform

		// -------------------------- SPACE - TIME FORMULATION OR QR PARAMETERS
		//int M_0;																						// in ST - num of time subinterval, in QR num of multiplications
		//int p;																							// M/M_0 = p

		// -------------------------- ALGORITHM RELATED PARAMETERS
		v_3d<int> spatialNorm;
		//arma::mat hopping_exp;																			// exponential of a hopping matrix
		//arma::mat int_exp_up, int_exp_down;																// exponentials of up and down spin interaction matrices at all times
		//v_1d<arma::mat> b_mat_up, b_mat_down;															// up and down B matrices vector
		//v_1d<arma::mat> b_mat_up_inv, b_mat_down_inv;													// up and down B matrices inverses vector

		//arma::mat green_up, green_down;																	// Green's matrix up and down at given (equal) time
		arma::mat tempGreen_up; arma::mat tempGreen_down;												// temporary Green's for wrap updating
		arma::mat tempGreen_up_i; arma::mat tempGreen_down_i;											// temporary Green's for wrap updating inverse

		// -------------------------- HELPING PARAMETERS
		//std::string info;																				// info about the model for file creation
		//std::shared_ptr<directories> dir;																// directories for model parameters saving

		// -------------------------- METHODS --------------------------

		// -------------------------- PROTECTED SETTERS
		//void set_hs();																					// setting Hubbard-Stratonovich fields
		//void setConfDir();

		// -------------------------- HELPING FUNCTIONS
		//std::pair<double, double> cal_gamma(int lat_site) const;										// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		//std::pair<double, double> cal_proba(int lat_site, double g_up, double g_down) const;			// calculate probability for both spins (0 <-> up index, 1 <-> down index)
		//virtual void av_single_step(int current_elem_i, int sign) = 0;									// take all the averages of a single step
		void av_normalise(int avNum, int timesNum);														// normalise all the averages after taking them

		// -------------------------- HEAT BATH
		//virtual int heat_bath_single_step(int lat_site) = 0;											// calculates the single step of a heat-bath algorithm
		//virtual void heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens = false) = 0;	// uses heat-bath to equilibrate system
		//virtual void heat_bath_av(int corr_time, int avNum, bool quiet) = 0;							// collect the averages from the simulation
		//virtual double sweep_0_M() = 0;																	// sweep forward in time
		//virtual double sweep_M_0() = 0;																	// sweep backwards

		// -------------------------- CALCULATORS
		//virtual void cal_green_mat(int which_time) = 0;													// calculates the Green matrices
		virtual void compare_green_direct(int tim, double toll, bool print_greens) = 0;					// compares Green's function from stability formulation to direct evaluation
		//void cal_int_exp();																				// calculates interaction exponents at all times
		//void cal_B_mat();																				// calculates B matrices
		//void cal_B_mat(int which_time);																	// recalculates the B matrix at a given time
		//void cal_hopping_exp();																			// calculates hopping exponent for nn

		// -------------------------- UPDATERS
		//void upd_int_exp(int lat_site, double delta_up, double delta_down);
		//void upd_B_mat(int lat_site, double delta_up, double delta_down);
		//virtual void upd_equal_green(int lat_site, double gamma_over_prob_up, \
		//	double gamma_over_prob_down) = 0;															// updates Greens at the same time after spin flip
		//virtual void upd_next_green(int which_time) = 0;
		//virtual void upd_prev_green(int which_time) = 0;
		virtual void upd_Green_step(int im_time_step, bool forward) = 0;

		// -------------------------- EQUAL TIME QUANTITIES TO BE COLLECTED
		void calOneSiteParam(int sign, int current_elem_i, sinOpType op, long double& av, long double& std) {
			double tmp = (*op)(sign, current_elem_i, this->green_up, this->green_down);
			av += tmp;
			std += tmp * tmp;
		};
	public:
		/// nonstatic operators (use fields)
		double cal_kinetic_en(int sign, int current_elem_i, const mat& g_up, const mat& g_down);									// calculate the kinetic energy part for averaging

		double cal_mz2_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down);					// calculate the z-th magnetization d correlation at i and j
		double cal_occupation_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down);			// calculate the average occupation correlation at i and j
		double cal_ch_correlation(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down);			// calculate the charge correlation at i and j

		/// static operators
		static double cal_occupation(int sign, int current_elem_i, const mat& g_up, const mat& g_down);								// calculate the average occupation

		static double cal_mz2(int sign, int current_elem_i, const mat& g_up, const mat& g_down);									// calculate the z-th magnetization squared
		static double cal_my2(int sign, int current_elem_i, const mat& g_up, const mat& g_down);									// calculate the y-th magnetization squared
		static double cal_mx2(int sign, int current_elem_i, const mat& g_up, const mat& g_down);									// calculate the x-th magnetization squared



		// -------------------------- EQUAL TIME FOURIER TRANSFORMS

	public:
		// -------------------------- PRINTERS
		void say_hi() {
			stout << "->M = " << this->M << EL \
				<< "->M0 = " << this->M_0 << EL \
				<< "->p = " << this->p << EL \
				// physical
				<< "->beta = " << this->beta << EL \
				<< "->U = " << this->U << EL \
				<< "->dtau = " << this->dtau << EL \
				<< "->mu = " << this->mu << EL \
				<< "->t = " << this->t << EL \
				// lattice
				<< "->dimension = " << this->getDim() << EL \
				<< "->type = " << this->lattice->get_type() << EL \
				<< "->Lx = " << this->lattice->get_Lx() << EL \
				<< "->Ly = " << this->lattice->get_Ly() << EL \
				<< "->Lz = " << this->lattice->get_Lz() << EL \
				<< "->lambda = " << this->lambda << EL << EL;
			/// Setting info about the model for files
			this->info = "M=" + STR(this->M) + ",M0=" + STR(this->M_0) + \
				",dtau=" + str_p(this->dtau) + ",Lx=" + STR(this->lattice->get_Lx()) + \
				",Ly=" + STR(this->lattice->get_Ly()) + ",Lz=" + STR(this->lattice->get_Lz()) + \
				",beta=" + str_p(this->beta) + ",U=" + str_p(this->U) + \
				",mu=" + str_p(this->mu);
		};
		void print_hs_fields(std::string separator = "\t") const;			// prints current HS fields configuration
		void print_hs_fields(std::string separator, const arma::mat& toPrint) const;
		void save_unequal_greens(int filenum, const vec& signs);

		// -------------------------- SETTERS
		void setDirs(directories* dirs) { this->dir.reset(dirs); };
		void setDirs(std::string working_directory);

		// -------------------------- CALCULATORS OVERRIDE
		void relaxation(impDef::algMC algorithm, int mcSteps, bool conf, bool quiet) override;
		void average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet) override;
	};
}
#endif
