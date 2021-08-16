#pragma once
#ifndef HUBBARD_h
#define HUBBARD_h

#include "general_model.h"
#include "plog/Log.h"
#include "plog/Initializers/RollingFileInitializer.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <armadillo>

namespace hubbard{
	class HubbardModel: public LatticeModel{
	protected:
		// ALGORITHM CONVERGENCE PARAMETERS 
		int from_scratch;																		// number of Trotter times for Green to be calculated from scratch
		uint64_t pos_num;																		// helps with number of positive signs
		uint64_t neg_num;																		// helps with number of negative signs
		// INITIAL PHYSICAL PARAMETERS 
		std::vector<double> t;																	// hopping integral vector
		double U;																				// Coulomb force strength
		double mu;																				// chemical potential

		// SUZUKI - TROTTER RELATED PARAMETERS 
		int M;																					// number of Trotter times
		double dtau;																			// Trotter time step
		int current_time;																		// current Trotter time

		// TRANSFORMATION RELATED PARAMETERS 
		std::vector<std::vector<short>> hsFields;												// Hubbard - Stratonovich fields - first time then field
		std::vector<long double> gammaExp;														// precalculated exponent of gammas
		long double lambda;																		// lambda parameter in HS transform

		// SPACE - TIME FORMULATION OR QR PARAMETERS 
		int M_0;																				// in ST - num of time subinterval, in QR num of multiplications
		int p;																					// M/M_0 = p

		// ALGORITHM RELATED PARAMETERS 
		arma::mat hopping_exp;																	// exponential of a hopping matrix
		arma::mat int_exp_up, int_exp_down;														// exponentials of up and down spin interaction matrices at all times
		std::vector<arma::mat> b_mat_up, b_mat_down;											// up and down B matrices vector
		arma::mat green_up, green_down;															// Green's matrix up and down at given (equal) time
		
		// HELPING PARAMETERS
		std::string info;																		// info about the model for file creation
		std::string neg_dir;																	// directory for negative configurations
		std::string neg_log;																	// the name of negative sign log file
		std::string pos_dir;																	// directory for positive configurations
		std::string pos_log;																	// the name of positive sign log file
		// -------------- METHODS ---------------
		


		// PROTECTED SETTERS 
		void set_hs();																			// setting Hubbard-Stratonovich fields									
		
		// HELPING FUNCTIONS 
		virtual std::tuple<long double,long double> cal_gamma(int lat_site) = 0;				// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		virtual std::tuple<long double,long double> cal_proba(int lat_stie, long double gamma_up, long double gamma_down) = 0;
		void av_single_step(int current_elem_i, int sign);								// take all the averages of a single step
		void av_normalise(int avNum, bool times = false);										// normalise all the averages after taking them

		// heat-bath
		virtual int heat_bath_single_step(int lat_site) = 0;									// calculates the single step of a heat-bath algorithm
		virtual int heat_bath_single_step_conf(int lat_site) = 0;								// calculates the single step of a heat-bath algorithm overloaded for saving directories
		virtual void heat_bath_eq(int mcSteps, bool conf, bool quiet) = 0;						// uses heat-bath to equilibrate system
		virtual void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) = 0;		// collect the averages from the simulation
	
		// CALCULATORS 
		virtual void cal_green_mat(int which_time) = 0;											// calculates the Green matrices
		void cal_int_exp();																		// calculates interaction exponents at all times
		void cal_B_mat();																		// calculates B matrices 
		void cal_hopping_exp();																	// calculates hopping exponent for nn

		// UPDATERS 
		virtual void upd_int_exp(int lat_site, long double delta_sigma, short sigma) = 0;
		virtual void upd_B_mat(int lat_site, long double delta_up, long double delta_down) = 0;
		virtual void upd_equal_green(int lat_site, long double prob_up, long double prob_down,\
			long double gamma_up, long double gamma_down) = 0;									// updates Greens at the same time after spin flip
		virtual void upd_next_green(int which_time) = 0;

		// EQUAL TIME QUANTITIES TO BE COLLECTED
		double cal_kinetic_en(int sign, int current_elem_i);									// calculate the kinetic energy part for averaging
		
		double cal_occupation(int sign, int current_elem_i);									// calculate the average occupation
		double cal_occupation_corr(int sign, int current_elem_i, int current_elem_j);			// calculate the average occupation correlation at i and j
		
		double cal_mz2(int sign, int current_elem_i);											// calculate the z-th magnetization squared
		double cal_mz2_corr(int sign, int current_elem_i, int current_elem_j);					// calculate the z-th magnetization squared correlation at i and j
		double cal_my2(int sign, int current_elem_i);											// calculate the y-th magnetization squared
		double cal_mx2(int sign, int current_elem_i);											// calculate the x-th magnetization squared

		double cal_ch_correlation(int sign, int current_elem_i, int current_elem_j);			// calculate the charge correlation at i and j
		
		// EQUAL TIME FOURIER TRANSFORMS


	public:
		// PRINTERS
		void print_hs_fields(std::ostream& output, int which_time_caused,\
			int which_site_caused, short this_site_spin, std::string separator = "\t");			// prints current HS fields configuration

		// GETTERS 
		int get_M() const;
		int get_M_0() const;
		std::string get_info() const;
		
		// SETTERS
		void setConfDir(std::string dir);														// sets the directory for saving configurations with a given sign
	};	

	//----------------------------------- QR -----------------------------------------

	/// <summary>
	/// A DQMC model that uses the QR decomposition scheme
	/// </summary>
	class HubbardQR: public HubbardModel{
	private:
		// HELPING FUNCTIONS 
		std::tuple<long double, long double> cal_gamma(int lat_site) override;															// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		std::tuple<long double, long double> cal_proba(int lat_stie, long double gamma_up, long double gamma_down) override;			// calculate probability for both spins (0 <-> up index, 1 <-> down index)
		// UPDATERS 
		void upd_int_exp(int lat_site, long double delta_sigma, short sigma) override;
		void upd_B_mat(int lat_site, long double delta_up, long double delta_down) override;
		void upd_equal_green(int lat_site, long double prob_up, long double prob_down, long double gamma_up, long double gamma_down) override;
		void upd_next_green(int which_time) override;
	
		// CALCULATORS
		void cal_green_mat(int which_time) override;

		// heat-bath
		int heat_bath_single_step(int lat_site) override;
		int heat_bath_single_step_conf(int lat_site) override;
		void heat_bath_eq(int mcSteps, bool conf, bool quiet) override;
		void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) override;
	public:
		// CONSTRUCTORS 
		HubbardQR(const std::vector<double>& t, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice);

		// CALCULATORS OVERRIDE
		void relaxation(impDef::algMC algorithm, int mcSteps, bool conf, bool quiet) override;
		void average(impDef::algMC algorithm,int corr_time, int avNum, int bootStraps, bool quiet, int times = 0) override;
	
	};

	//----------------------------------- SPACE TIME -----------------------------------------

	/*
	/// <summary>
	/// A DQMC model that uses the Space-Time formulation scheme
	/// </summary>
	class HubbardST : public HubbardModel{
	private:
		// HELPING FUNCTIONS 
		std::tuple<long double,long double> cal_gamma(int lat_site) override;														// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		std::tuple<long double, long double> cal_proba(int lat_stie, long double gamma_up, long double gamma_down) override;
		// UPDATERS 
		void upd_int_exp(int lat_site, long double delta_sigma, short sigma) override;
		void upd_B_mat(int lat_site, long double delta_up, long double delta_down) override;
		void upd_equal_green(int lat_site, long double prob_up, long double prob_down, long double gamma_up, long double gamma_down) override;
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