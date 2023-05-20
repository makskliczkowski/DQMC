#pragma once
#include "hubbard.h"

#ifndef HUBBARD_ST_H
#define HUBBARD_ST_H
// ---------------------------------------------------------------------- SPACE TIME ----------------------------------------------------------------------------
namespace hubbard {
	/*
	* @brief Hubbard model with stable matrix multiplication
	* @todo Add more complicated structure of the Hamiltonian
	*/
	class HubbardQR : public hubbard::HubbardModel {
	private:


		v_1d<arma::mat> b_up_condensed, b_down_condensed;																// up and down B matrices vector premultiplied
		/// time displaced Green's functions

		arma::mat g_up_time, g_down_time;

		v_1d<arma::mat> b_ups; v_1d<arma::mat> b_downs;																	// to store mult of B matrices e.g.B_{M-1}^{-1},B_{M-2}^{-1}*B_{M-1}^{-1}, ... 
		v_1d<arma::mat> b_ups_i; v_1d<arma::mat> b_downs_i;																// to store mult of B matrices inverses e.g.B_{M-1}^{-1},B_{M-2}^{-1}*B_{M-1}^{-1}, ... 
		// -------------------------- DIRECTORIES AND SAVERS

		// -------------------------- HELPING FUNCTIONS
		double sweep_0_M() override;																						// sweep forward in time
		double sweep_M_0() override;																						// sweep backwards in time
		int sweep_lat_sites();																							// sweep the lattice sites for auxliary Ising spins

		// -------------------------- UPDATERS
		void upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down) override;			// after auxliary Ising spin update - local
		void upd_next_green(int which_time) override;																	// forward propagator
		void upd_prev_green(int which_time) override;																	// backward propagator
		void upd_Green_step(int im_time_step, bool forward = true) override;
		void cal_B_mat_cond(int which_sector);																			// update the condensation of the B matrices for a given sector

		// -------------------------- CALCULATORS
		/// b matrices mutliplication					
		void b_mat_mult_left(int l_start, int l_end, const mat& toMultUp, const mat& toMultDown, mat& toSetUp, mat& toSetDown);
		void b_mat_mult_left_inv(int l_start, int l_end, const mat& toMultUp, const mat& toMultDown, mat& toSetUp, mat& toSetDown);

		/// eq time Green's matrices calculators
		void cal_green_mat(int which_time) override;
		void cal_green_mat_cycle(int sector);

		/// non-equal time Green's matrices calculators
		void cal_green_mat_times(); 																					// give left time already
		void cal_green_mat_times_hirsh();																				// obtain full space-time Green's matrix - all of them
		void cal_green_mat_times_hirsh_cycle();																			// use wrapping for calculating space time Green's matrix - some of them only
		void uneqG_t1gtt2(int, int, const mat&, const mat&, const mat&, const mat&);									// after obtaining series of inverses and series of non-invs it calculates the Green's mat
		void uneqG_t1ltt2(int t1, int t2);

		/// comparison & stability tests
		void compare_green_direct(int tim, double toll, bool print_greens) override;

		// -------------------------- HEAT-BATH
		void heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens = false) override;						// equalibrate the model
		void heat_bath_av(int corr_time, int avNum, bool quiet) override;												// collect averages
		void av_single_step(int current_elem_i, int sign) override;														// single step of averaging
		void av_unequal_greens_single_step(int xx, int yy, int zz, int i, int j, int sign);								// save unequal greens in single step
		int heat_bath_single_step(int lat_site) override;																// single step with lattice updating
	public:
		// -------------------------- CONSTRUCTORS

		/*
		* @brief initialize memory for all of the variables used later
		*/
		void initializeMemory();

		/*
		* Standard constructor for the Hubbard model
		* @constructor
		* @param t The vector of hopping integrals between lattice sites
		* @param dtau Time step for inverse time cycle
		* @param M_0 Number of stable multiplications in the cycle
		* @param U Coloumb interaction
		* @param mu Chemical potential in the GC ensemble
		* @param beta Inverse temperature $\\frac{1}{T}$
		* @param threads Number of the inner threads
		* @param ct Shall calculate time-displaced as well?
		*/
		HubbardQR(const v_1d<double>& t, const hubbard::HubbardParams& params, std::shared_ptr<Lattice> lattice, int threads = 1);
	};


	//? -------------------------------------------------------- CONSTRUCTORS
	inline hubbard::HubbardQR::HubbardQR(const v_1d<double>& t, const hubbard::HubbardParams& params, std::shared_ptr<Lattice> lattice, int threads)
	{
		this->lattice = lattice;
		this->inner_threads = threads;
		this->dir = std::make_shared<hubbard::directories>();
		// if we use hirsh calculation of the time displaced greens then all times are not necessary

		this->dim = this->lattice->get_Dim();
		this->Lx = this->lattice->get_Lx();
		this->Ly = this->lattice->get_Ly();
		this->Lz = this->lattice->get_Lz();
		this->t = t;
		this->config_sign = 1;

		// Params
		this->U = params.U;
		this->mu = params.mu;
		this->beta = params.beta;
		this->T = 1.0 / this->beta;
		this->Ns = this->lattice->get_Ns();

		// random number generator initialization
		this->ran = randomGen(std::random_device{}());

		// Trotter
		this->dtau = params.dtau;
		this->M = params.M;																				// number of Trotter times
		this->M_0 = params.M0;																			// number of stable decompositions in Trotter times
		this->p = params.p;																				// number of QR decompositions (sectors)

		this->avs = std::make_shared<averages_par>(this->lattice, M);
		// Calculate alghorithm parameters // lambda couples to the auxiliary spins
		this->lambda = (U > 0) ? std::acosh(exp((this->U * this->dtau) / 2.0)) : std::acosh(exp((-this->U * this->dtau) * 0.5));

		// Calculate changing exponents before, not to calculate exp all the time
		// 0 -> sigma * hsfield = 1, 1 -> sigma * hsfield = -1
		this->gammaExp0 = std::make_pair(
			std::expm1(-2.0 * this->lambda),
			std::expm1(2.0 * this->lambda)
		);
		// 0 -> sigma * hsfield = -1, 1 -> sigma * hsfield = 1
		this->gammaExp1 = std::make_pair(
			this->gammaExp0.second,
			this->gammaExp0.first
		);

		// Helping params
		this->from_scratch = this->M_0;																	// after what time shall we recalculate the equal time Greens
		this->pos_num = 0;
		this->neg_num = 0;

		// Say hi to the world
#pragma omp critical
		stout << "CREATING THE HUBBARD MODEL WITH QR DECOMPOSITION WITH PARAMETERS:" << EL;
#pragma omp critical
		this->say_hi();

		// Initialize memory
		this->initializeMemory();

		// Set HS fields												
		this->set_hs();

		// Calculate something
		this->cal_hopping_exp();
		this->cal_int_exp();
		this->cal_B_mat();

		// Precalculate the multipliers of B matrices for convinience
		for (int i = 0; i < this->p; i++) this->cal_B_mat_cond(i);

		// progress bar initialize
		this->pbar = std::make_unique<pBar>(34, 1);
	}

}

#endif