#pragma once
#ifndef HUBBARD_h
#define HUBBARD_h

#include "general_model.h"

namespace hubbard {
	class HubbardModel : public LatticeModel {
	protected:
		// -------------------------- ALGORITHM CONVERGENCE PARAMETERS
		int from_scratch;																		// number of Trotter times for Green to be calculated from scratch
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
		std::vector<std::vector<short>> hsFields;												// Hubbard - Stratonovich fields - first time then field
		std::vector<double> gammaExp;														// precalculated exponent of gammas
		double lambda;																		// lambda parameter in HS transform

		// -------------------------- SPACE - TIME FORMULATION OR QR PARAMETERS
		int M_0;																				// in ST - num of time subinterval, in QR num of multiplications
		int p;																					// M/M_0 = p

		// -------------------------- ALGORITHM RELATED PARAMETERS
		arma::mat hopping_exp;																	// exponential of a hopping matrix
		arma::mat int_exp_up, int_exp_down;														// exponentials of up and down spin interaction matrices at all times
		std::vector<arma::mat> b_mat_up, b_mat_down;											// up and down B matrices vector
		arma::mat green_up, green_down;															// Green's matrix up and down at given (equal) time
		arma::mat tempGreen_up; arma::mat tempGreen_down;										// temporary Green's for wrap updating

		// -------------------------- HELPING PARAMETERS
		std::string info;																		// info about the model for file creation
		std::string neg_dir;																	// directory for negative configurations
		std::string neg_log;																	// the name of negative sign log file
		std::string pos_dir;																	// directory for positive configurations
		std::string pos_log;																	// the name of positive sign log file
		// -------------------------- METHODS --------------------------

		// -------------------------- PROTECTED SETTERS
		void set_hs();																			// setting Hubbard-Stratonovich fields

		// -------------------------- HELPING FUNCTIONS
		virtual std::tuple<double, double> cal_gamma(int lat_site) const = 0;					// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		virtual std::tuple<double, double> cal_proba(int lat_site\
			, double gamma_up, double gamma_down) const = 0;
		void av_single_step(int current_elem_i, int sign);										// take all the averages of a single step
		void av_normalise(int avNum, bool times = false);										// normalise all the averages after taking them

		// -------------------------- HEAT BATH
		virtual int heat_bath_single_step(int lat_site) = 0;									// calculates the single step of a heat-bath algorithm
		virtual int heat_bath_single_step_no_upd(int lat_site) = 0;								// calculates the single step of a heat-bath algorithm without flipping
		virtual int heat_bath_single_step_conf(int lat_site) = 0;								// calculates the single step of a heat-bath algorithm overloaded for saving directories
		virtual void heat_bath_eq(int mcSteps, bool conf, bool quiet) = 0;						// uses heat-bath to equilibrate system
		virtual void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) = 0;		// collect the averages from the simulation

		// -------------------------- CALCULATORS
		virtual void cal_green_mat(int which_time) = 0;											// calculates the Green matrices
		virtual void compare_green_direct(int tim, double toll, bool print_greens) = 0;			// compares Green's function from stability formulation to direct evaluation
		virtual void cal_B_mat_cond(int which_sector) = 0;
		void cal_int_exp();																		// calculates interaction exponents at all times
		void cal_B_mat();																		// calculates B matrices
		void cal_B_mat(int which_time);															// recalculates the B matrix at a given time

		void cal_hopping_exp();																	// calculates hopping exponent for nn

		// -------------------------- UPDATERS
		virtual void upd_int_exp(int lat_site, double delta_up, double delta_down) = 0;
		virtual void upd_B_mat(int lat_site, double delta_up, double delta_down) = 0;
		virtual void upd_equal_green(int lat_site, double prob_up, double prob_down, \
			double gamma_up, double gamma_down) = 0;									// updates Greens at the same time after spin flip
		virtual void upd_next_green(int which_time) = 0;

		// -------------------------- EQUAL TIME QUANTITIES TO BE COLLECTED
		double cal_kinetic_en(int sign, int current_elem_i) const;									// calculate the kinetic energy part for averaging

		double cal_occupation(int sign, int current_elem_i) const;									// calculate the average occupation
		double cal_occupation_corr(int sign, int current_elem_i, int current_elem_j) const;			// calculate the average occupation correlation at i and j

		double cal_mz2(int sign, int current_elem_i) const;											// calculate the z-th magnetization squared
		double cal_mz2_corr(int sign, int current_elem_i, int current_elem_j) const;					// calculate the z-th magnetization squared correlation at i and j
		double cal_my2(int sign, int current_elem_i) const;											// calculate the y-th magnetization squared
		double cal_mx2(int sign, int current_elem_i) const;											// calculate the x-th magnetization squared

		double cal_ch_correlation(int sign, int current_elem_i, int current_elem_j) const;			// calculate the charge correlation at i and j

		// -------------------------- EQUAL TIME FOURIER TRANSFORMS

	public:
		// -------------------------- PRINTERS
		void print_hs_fields(std::ostream& output, int which_time_caused, \
			int which_site_caused, short this_site_spin, std::string separator = "\t") const;			// prints current HS fields configuration

		// -------------------------- GETTERS
		int get_M() const { return this->M; };
		int get_M_0() const { return this->M_0; };
		std::string get_info() const { return this->info; };

		// -------------------------- SETTERS
		void setConfDir(std::string dir);														// sets the directory for saving configurations with a given sign
	};

	// -------------------------------------------------------- QR --------------------------------------------------------

	/// <summary>
	/// A DQMC model that uses the QR decomposition scheme
	/// </summary>
	class HubbardQR : public HubbardModel {
	private:
		// -------------------------- HELPING MATRICES FOR QR
		arma::mat Q_down;
		arma::mat Q_up;
		arma::umat P_down;																								// permutation matrix for spin down
		arma::umat P_up;																								// permutation matrix for spin up
		arma::mat R_down;																								// right triangular matrix down
		arma::mat R_up;																									// right triangular matrix up
		arma::vec D_down;
		arma::vec D_up;
		arma::mat T_down;
		arma::mat T_up;
		std::vector<arma::mat> b_up_condensed, b_down_condensed;														// up and down B matrices vector for QR DECOMPOSITION

		// -------------------------- HELPING FUNCTIONS
		std::tuple<double, double> cal_gamma(int lat_site) const override;												// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		std::tuple<double, double> cal_proba(int lat_site, double gamma_up, double gamma_down) const override;			// calculate probability for both spins (0 <-> up index, 1 <-> down index)

		// -------------------------- UPDATERS
		void upd_int_exp(int lat_site, double delta_up, double delta_down) override;
		void upd_B_mat(int lat_site, double delta_up, double delta_down) override;
		void upd_equal_green(int lat_site, double prob_up, double prob_down, double gamma_up, double gamma_down) override;
		void upd_next_green(int which_time) override;
		void cal_B_mat_cond(int which_sector) override;

		// -------------------------- CALCULATORS
		void cal_green_mat(int which_time) override;
		void compare_green_direct(int tim, double toll, bool print_greens) override;

		// -------------------------- HEAT-BATH
		int heat_bath_single_step(int lat_site) override;																// single step with updating
		int heat_bath_single_step_no_upd(int lat_site) override;														// single step without updating
		int heat_bath_single_step_conf(int lat_site) override;															// single step with saving configurations
		void heat_bath_eq(int mcSteps, bool conf, bool quiet) override;
		void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) override;
	public:
		// -------------------------- CONSTRUCTORS
		HubbardQR(const std::vector<double>& t, double dtau, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice, int threads = 1);

		// -------------------------- CALCULATORS OVERRIDE
		void relaxation(impDef::algMC algorithm, int mcSteps, bool conf, bool quiet) override;
		void average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet, int times = 0) override;
	};

	//----------------------------------- SPACE TIME -----------------------------------------

	/*
	/// <summary>
	/// A DQMC model that uses the Space-Time formulation scheme
	/// </summary>
	class HubbardST : public HubbardModel{
	private:
		// HELPING FUNCTIONS
		std::tuple<double,double> cal_gamma(int lat_site) override;														// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		std::tuple<double, double> cal_proba(int lat_site, double gamma_up, double gamma_down) override;
		// UPDATERS
		void upd_int_exp(int lat_site, double delta_sigma, short sigma) override;
		void upd_B_mat(int lat_site, double delta_up, double delta_down) override;
		void upd_equal_green(int lat_site, double prob_up, double prob_down, double gamma_up, double gamma_down) override;
		void upd_next_green() override;

		// CALCULATORS
		void cal_green_mat(int which_time) override;
	public:
		// CONSTRUCTORS
		~HubbardST() = default;
		HubbardST() = default;
		HubbardST(std::vector<double> t, double U, double mu, double T, std::shared_ptr<Lattice>const& lattice);

		// CALCULATORS OVERRIDE
		void relaxation(impDef::algMC algorithm, int mc_steps, bool conf, bool quiet) override;
		void average(impDef::algMC algorithm,int corr_time, int avNum, int bootStraps, bool quiet) override;
	};
	*/
}

#endif