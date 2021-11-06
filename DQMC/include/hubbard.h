#pragma once
#ifndef HUBBARD_h
#define HUBBARD_h

#include "general_model.h"

namespace hubbard {
	using namespace arma;

	struct directories : public general_directories {
		std::string fourier_dir = "";
		std::string params_dir= "";
		std::string conf_dir = "";
		std::string greens_dir = "";
		std::string time_greens_dir = "";

		// filenames											
		std::string nameFouriers = "";							// 
		std::string nameFouriersTime = "";						// 
		std::string nameNormal = "";								// 
		std::string nameNormalTime = "";							// 
		std::string nameGreens = "";								// 
		std::string nameGreensTime = "";							// 

		// configuration directories
		std::string neg_dir = "";								// directory for negative configurations
		std::string neg_log = "";								// the name of negative sign log file
		std::string pos_dir = "";								// directory for positive configurations
		std::string pos_log = "";								// the name of positive sign log file

		directories() = default;

		void setFileNames(){
			this->nameFouriers = fourier_dir + "fouriers_" + info + ".dat";
			this->nameFouriersTime = fourier_dir + "times"+ kPSepS + "fouriersTime_" + info + ".dat";
			this->nameNormal = params_dir + "parameters_" + info + ".dat";
			this->nameNormalTime = params_dir + "times"+ kPSepS + "parametersTime_" + info + ".dat";
			this->nameGreens = "greens" + info + ".dat";
			this->nameGreensTime = "greensTime_" + info + ".dat";
		};


	};


	class HubbardModel : public LatticeModel {
	protected:
		// -------------------------- ALGORITHM CONVERGENCE PARAMETERS
		int from_scratch;																		// number of Trotter times for Green to be calculated from scratch
		int config_sign;																		// keep track of the configuration sign
		bool cal_times;
		long int pos_num;																		// helps with number of positive signs
		long int neg_num;																		// helps with number of negative signs


		// -------------------------- INITIAL PHYSICAL PARAMETERS
		std::vector<double> t;																	// hopping integral vector
		double U;																				// Coulomb force strength
		double mu;																				// chemical potential

		// -------------------------- SUZUKI - TROTTER RELATED PARAMETERS
		int M;																					// number of Trotter times
		double dtau;																			// Trotter time step
		int current_time;																		// current Trotter time

		// -------------------------- TRANSFORMATION RELATED PARAMETERS
		arma::mat hsFields;																	// Hubbard - Stratonovich fields - first time then field
		//std::vector<std::vector<std::string>> hsFields_img;
		std::vector<double> gammaExp;														// precalculated exponent of gammas
		double lambda;																		// lambda parameter in HS transform

		// -------------------------- SPACE - TIME FORMULATION OR QR PARAMETERS
		int M_0;																					// in ST - num of time subinterval, in QR num of multiplications
		int p;																						// M/M_0 = p

		// -------------------------- ALGORITHM RELATED PARAMETERS
		v_3d<int> spatialNorm;
		arma::mat hopping_exp;																		// exponential of a hopping matrix
		arma::mat int_exp_up, int_exp_down;															// exponentials of up and down spin interaction matrices at all times
		std::vector<arma::mat> b_mat_up, b_mat_down;												// up and down B matrices vector
		std::vector<arma::mat> b_mat_up_inv, b_mat_down_inv;										// up and down B matrices inverses vector

		arma::mat green_up, green_down;																// Green's matrix up and down at given (equal) time
		arma::mat tempGreen_up; arma::mat tempGreen_down;											// temporary Green's for wrap updating

		v_1d<arma::mat> g_up_diffs, g_down_diffs;
		v_1d<arma::cx_mat> g_up_diffs_k, g_down_diffs_k;

		// -------------------------- HELPING PARAMETERS
		std::string info;																			// info about the model for file creation
		std::shared_ptr<directories> dir;															// directories for model parameters saving
		

		// -------------------------- METHODS --------------------------

		// -------------------------- PROTECTED SETTERS
		void set_hs();																				// setting Hubbard-Stratonovich fields
		void setConfDir();

		// -------------------------- HELPING FUNCTIONS
		std::tuple<double, double> cal_gamma(int lat_site) const;									// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		std::tuple<double, double> cal_proba(int lat_site, double g_up, double g_down) const;		// calculate probability for both spins (0 <-> up index, 1 <-> down index)
		virtual void av_single_step(int current_elem_i, int sign, bool times) = 0;					// take all the averages of a single step
		void av_normalise(int avNum, int timesNum, bool times = false);								// normalise all the averages after taking them

		// -------------------------- HEAT BATH
		virtual int heat_bath_single_step(int lat_site) = 0;										// calculates the single step of a heat-bath algorithm
		virtual int heat_bath_single_step_no_upd(int lat_site) = 0;									// calculates the single step of a heat-bath algorithm without flipping
		virtual int heat_bath_single_step_conf(int lat_site) = 0;									// calculates the single step of a heat-bath algorithm overloaded for saving directories
		virtual void heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens = false) = 0;// uses heat-bath to equilibrate system
		virtual void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) = 0;			// collect the averages from the simulation
		virtual void sweep_0_M(std::function<int(int)> ptfptr) = 0;									// sweep forward in time
		virtual void sweep_M_0(std::function<int(int)> ptfptr) = 0;									// sweep backwards

		// -------------------------- CALCULATORS
		virtual void cal_green_mat(int which_time) = 0;												// calculates the Green matrices
		virtual void compare_green_direct(int tim, double toll, bool print_greens) = 0;				// compares Green's function from stability formulation to direct evaluation
		void cal_int_exp();																			// calculates interaction exponents at all times
		void cal_B_mat();																			// calculates B matrices
		void cal_B_mat(int which_time);																// recalculates the B matrix at a given time

		void cal_hopping_exp();																		// calculates hopping exponent for nn

		// -------------------------- UPDATERS
		void upd_int_exp(int lat_site, double delta_up, double delta_down);
		void upd_B_mat(int lat_site, double delta_up, double delta_down);
		virtual void upd_equal_green(int lat_site, double gamma_over_prob_up, \
			double gamma_over_prob_down) = 0;														// updates Greens at the same time after spin flip
		virtual void upd_next_green(int which_time) = 0;
		virtual void upd_prev_green(int which_time) = 0;
		virtual void upd_Green_step(int im_time_step, bool forward) = 0;

		// -------------------------- EQUAL TIME QUANTITIES TO BE COLLECTED
		double cal_kinetic_en(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const;									// calculate the kinetic energy part for averaging

		double cal_occupation(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const;									// calculate the average occupation
		double cal_occupation_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down) const;			// calculate the average occupation correlation at i and j

		double cal_mz2(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const;											// calculate the z-th magnetization squared
		double cal_mz2_corr(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down) const;					// calculate the z-th magnetization squared correlation at i and j
		double cal_my2(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const;											// calculate the y-th magnetization squared
		double cal_mx2(int sign, int current_elem_i, const mat& g_up, const mat& g_down) const;											// calculate the x-th magnetization squared

		double cal_ch_correlation(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down) const;			// calculate the charge correlation at i and j

		// -------------------------- EQUAL TIME FOURIER TRANSFORMS


	public:
		// -------------------------- PRINTERS
		void print_hs_fields(std::ostream& output, int which_time_caused, \
			int which_site_caused, short this_site_spin, std::string separator = "\t") const;			// prints current HS fields configuration

		void save_unequal_greens(int filenum, uint bucketnum = 1);

		// -------------------------- GETTERS
		int get_M() const { return this->M; };
		int get_M_0() const { return this->M_0; };
		std::string get_info() const { return this->info; };
		std::shared_ptr<directories> get_directories() const {return this->dir;};

		// -------------------------- SETTERS
		void setDirs(directories directories);
		void setDirs(std::string working_directory);

		// -------------------------- CALCULATORS OVERRIDE
		void relaxation(impDef::algMC algorithm, int mcSteps, bool conf, bool quiet) override;
		void average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet, int times = 0) override;
	};

}
#endif




