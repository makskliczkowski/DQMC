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
		/// decomposition for stable matrix multiplication. We first want to reserve memory for them
		arma::mat Q_down;																								// unitary matrix for spin down
		arma::mat Q_up;																									// unitary matrix for spin up
		arma::umat P_down;																								// permutation matrix for spin down
		arma::umat P_up;																								// permutation matrix for spin up
		arma::mat R_down;																								// right triangular matrix down (for QR decomposition)
		arma::mat R_up;																									// right triangular matrix up (for QR decomposition)
		arma::vec D_down;																								// diagonal matrix vector for UDT decomposition for spin down
		arma::vec D_up;																									// diagonal matrix vector for UDT decomposition for spin up
		arma::mat T_down;																								// UDT upper triangular matrix for spin down
		arma::mat T_up;																									// UDT upper triangular matrix for spin up
		arma::vec D_tmp;																								// tmp diagonal matrix vector - for time displaced Green's

		v_1d<arma::mat> b_up_condensed, b_down_condensed;																// up and down B matrices vector premultiplied
		/// time displaced Green's functions

		arma::mat g_up_time, g_down_time;
		v_1d<arma::mat> g_up_eq; v_1d<arma::mat> g_down_eq;
		v_1d<arma::mat> g_up_tim; v_1d<arma::mat> g_down_tim;
		v_1d<mat> b_up_inv_cond; v_1d<mat> b_down_inv_cond;																// to store mult of B matrices inverses e.g.B_{M-1}^{-1},B_{M-2}^{-1}*B_{M-1}^{-1}, ... 
		// -------------------------- DIRECTORIES AND SAVERS

		// -------------------------- HELPING FUNCTIONS
		arma::mat multiplyMatrices(const mat&, const mat&, bool);														// choose between stable and normal multiplication
		void sweep_0_M() override;																						// sweep forward in time
		void sweep_M_0() override;																						// sweep backwards in time
		int sweep_lat_sites();																							// sweep the lattice sites for auxliary Ising spins

		// -------------------------- UPDATERS
		void upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down) override;			// after auxliary Ising spin update - local
		void upd_next_green(int which_time) override;																	// forward propagator
		void upd_prev_green(int which_time) override;																	// backward propagator
		void upd_Green_step(int im_time_step, bool forward = true) override;											
		void cal_B_mat_cond(int which_sector);																			// update the condensation of the B matrices for a given sector

		// -------------------------- CALCULATORS
		/// b matrices mutliplication					
		void b_mat_mult_left(int l_start, int l_end, const mat& toMultUp,const mat& toMultDown, mat& toSetUp, mat& toSetDown);
		void b_mat_mult_left_inv(int l_start, int l_end, const mat& toMultUp,const mat& toMultDown, mat& toSetUp, mat& toSetDown);

		/// eq time Green's matrices calculators
		void cal_green_mat(int which_time) override;
		void cal_green_mat_cycle(int sector);

		/// non-equal time Green's matrices calculators
		void cal_green_mat_times(); 																					// give left time already
		void cal_green_mat_times_cycle();																				// only at given positions in the M_0 cycle
		void cal_green_mat_times_hirsh();																				// obtain full space-time Green's matrix - all of them
		void cal_green_mat_times_hirsh_cycle();																			// use wrapping for calculating space time Green's matrix - some of them only
		void uneqG_t1gtt2(int, int, const mat&, const mat&, const mat&, const mat&);									// after obtaining series of inverses and series of non-invs it calculates the Green's mat
		void uneqG_t1ltt2(int t1, int t2);

		/// comparison & stability tests
		void compare_green_direct(int tim, double toll, bool print_greens) override;

		// -------------------------- HEAT-BATH
		void heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens = false) override;						// equalibrate the model
		void heat_bath_av(int corr_time, int avNum, bool quiet) override;									// collect averages
		void av_single_step(int current_elem_i, int sign) override;														// single step of averaging
		int heat_bath_single_step(int lat_site) override;																// single step with lattice updating
	public:
		// -------------------------- CONSTRUCTORS
		/*
		* @brief initialize memory for all of the variables used later
		*/	
		void initializeMemory();

		/**
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
#ifdef ALL_TIMES
		this->all_times = true;
#else 
		this->all_times = false;
#endif

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

		this->avs = std::make_shared<averages_par>(this->lattice, M, this->cal_times);
		// Calculate alghorithm parameters
		this->lambda = std::acosh(exp((abs(this->U) * this->dtau) * 0.5));								// lambda couples to the auxiliary spins

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