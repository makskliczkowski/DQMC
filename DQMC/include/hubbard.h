#pragma once
#ifndef HUBBARD_h
#define HUBBARD_h

#include "general_model.h"
#include <armadillo>

namespace hubbard{


	class HubbardModel: public virtual generalModel::LatticeModel{
	private:
		/* INITIAL PHYSICAL PARAMETERS */
		std::vector<double> t;											// hopping integral vector
		double U;														// Coulomb force strength
		double mu;														// chemical potential

		/* SUZUKI - TROTTER RELATED PARAMETERS */
		int M;															// number of Trotter times
		double dtau;													// Trotter time step
		int current_time;												// current Trotter time

		/* TRANSFORMATION RELATED PARAMETERS */
		std::vector<std::vector<short>> hsFields;						// Hubbard - Stratonovich fields - first time then field
		std::vector<long double> gammaExp;								// precalculated exponent of gammas
		long double lambda;												// lambda parameter in HS transform

		/* SPACE - TIME FORMULATION OR QR PARAMETERS */
		int M_0;														// in ST - num of time subinterval, in QR num of multiplications
		int p;															// M/M_0 = p

		/* ALGORITHM RELATED PARAMETERS */
		arma::mat hopping_exp;											// exponential of a hopping matrix
		arma::mat int_exp_up, int_exp_down;								// exponentials of up and down spin interaction matrices at all times
		std::vector<arma::mat> b_mat_up, b_mat_down;					// up and down B matrices vector
		arma::mat green_up, green_down;									// Green's matrix up and down at given (equal) time
	protected:
		void set_hs();													// setting Hubbard-Stratonovich fields									
	private:
		/* HELPING FUNCTIONS */
		virtual std::tuple<long double,long double> cal_gamma(int lat_site) = 0;															// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		virtual std::tuple<long double, long double> cal_proba(int lat_stie, long double gamma_up, long double gamma_down) = 0;
		/* CALCULATORS */
		virtual void cal_green_mat(int which_time) = 0;					// calculates the Green matrices
		void cal_int_exp();												// calculates interaction exponents at all times
		void cal_B_mat();												// calculates B matrices 
		void cal_hopping_exp_nn();										// calculates hopping exponent for nn;

		/* UPDATERS */
		virtual void upd_int_exp(int lat_site, long double delta_sigma, short sigma) = 0;
		virtual void upd_B_mat(int lat_site, long double delta_up, long double delta_down) = 0;
		virtual void upd_equal_green(int lat_site, long double prob_up, long double prob_down, long double gamma_up, long double gamma_down) = 0;
		virtual void upd_next_green() = 0;
	public:
		/* GETTERS */
		int get_M() const;
		int get_M_0() const;

	};

	//----------------------------------- QR -----------------------------------------

	/// <summary>
	/// A DQMC model that uses the QR decomposition scheme
	/// </summary>
	class HubbardQR: public virtual HubbardModel{
	private:
		/* HELPING FUNCTIONS */
		std::tuple<long double,long double> cal_gamma(int lat_site);															// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		std::tuple<long double, long double> cal_proba(int lat_stie, long double gamma_up, long double gamma_down);			// calculate probability for both spins (0 <-> up index, 1 <-> down index)
		/* UPDATERS */
		void upd_int_exp(int lat_site, long double delta_sigma, short sigma);
		void upd_B_mat(int lat_site, long double delta_up, long double delta_down);
		void upd_equal_green(int lat_site, long double prob_up, long double prob_down, long double gamma_up, long double gamma_down);
		void upd_next_green();
	
	public:
		/* CONSTRUCTORS */
		~HubbardQR() final = default;
		HubbardQR() = default;
		HubbardQR(std::vector<double> t, double U, double mu, double T, std::shared_ptr<generalModel::Lattice>const& lattice);

		/* CALCULATORS OVERRIDE */
		void relaxation(impDef::algMC algorithm, int mc_steps, bool conf, bool quiet);
		void average(impDef::algMC algorithm,int corr_time, int avNum, int bootStraps, bool quiet);

		
	
	
	};

	//----------------------------------- SPACE TIME -----------------------------------------


	/// <summary>
	/// A DQMC model that uses the Space-Time formulation scheme
	/// </summary>
	class HubbardST : public virtual HubbardModel{
	private:
		/* HELPING FUNCTIONS */
		std::tuple<long double,long double> cal_gamma(int lat_site);															// calculate gamma for both spins (0 <-> up index, 1 <-> down index)
		std::tuple<long double, long double> cal_proba(int lat_stie, long double gamma_up, long double gamma_down);
		/* UPDATERS */
		void upd_int_exp(int lat_site, long double delta_sigma, short sigma);
		void upd_B_mat(int lat_site, long double delta_up, long double delta_down);
		void upd_equal_green(int lat_site, long double prob_up, long double prob_down, long double gamma_up, long double gamma_down);
		void upd_next_green();
	public:
		/* CONSTRUCTORS */
		~HubbardST() final = default;
		HubbardST() = default;
		HubbardST(std::vector<double> t, double U, double mu, double T, std::shared_ptr<generalModel::Lattice>const& lattice);

		/* CALCULATORS OVERRIDE */
		void relaxation(impDef::algMC algorithm, int mc_steps, bool conf, bool quiet);
		void average(impDef::algMC algorithm,int corr_time, int avNum, int bootStraps, bool quiet);
	
	
	};



}






#endif