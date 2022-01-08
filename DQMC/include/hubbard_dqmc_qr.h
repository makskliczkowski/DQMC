#pragma once
#include "hubbard.h"

#ifndef HUBBARD_ST_H
#define HUBBARD_ST_H
// ---------------------------------------------------------------------- SPACE TIME ----------------------------------------------------------------------------
namespace hubbard {
	/**
	* Hubbard model with stable matrix multiplication
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
		arma::vec D_up;																									// diagonal matrix vector for UDT decomposition	for spin up
		arma::mat T_down;																								// UDT upper triangular matrix for spin down
		arma::mat T_up;																									// UDT upper triangular matrix for spin up
		arma::vec D_tmp;																								// tmp diagonal matrix vector - for time displaced Green's

		v_1d<arma::mat> b_up_condensed, b_down_condensed;																// up and down B matrices vector premultiplied
		/// time displaced Green's functions
		arma::mat g_up_time, g_down_time;
		v_1d<arma::mat> g_up_eq; v_1d<arma::mat> g_down_eq;
		v_1d<arma::mat> g_up_tim; v_1d<arma::mat> g_down_tim;
		// -------------------------- DIRECTORIES AND SAVERS

		// -------------------------- HELPING FUNCTIONS
		void sweep_0_M(std::function<int(int)> ptfptr) override;														// sweep forward in time
		void sweep_M_0(std::function<int(int)> ptfptr) override;														// sweep backwards in time
		int sweep_lat_sites(std::function<int(int)> ptfptr);															// sweep the lattice sites for auxliary Ising spins

		// -------------------------- UPDATERS
		void upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down) override;			// after auxliary Ising spin update - local
		void upd_next_green(int which_time) override;																	// forward propagator
		void upd_prev_green(int which_time) override;																	// backward propagator
		void upd_Green_step(int im_time_step, bool forward = true) override;											
		void cal_B_mat_cond(int which_sector);																			// update the condensation of the B matrices for a given sector

		// -------------------------- CALCULATORS
		/// b matrices mutliplication
		void b_mat_multiplier_right(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);					
		void b_mat_multiplier_left(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);
		void b_mat_multiplier_right_inv(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);
		void b_mat_multiplier_left_inv(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);

		/// eq time Green's matrices calculators
		void cal_green_mat(int which_time) override;
		void cal_green_mat_cycle(int sector);

		/// non-equal time Green's matrices calculators
		void cal_green_mat_times(); 																					// give left time already
		void cal_green_mat_times_cycle();																				// only at given positions in the M_0 cycle
		void cal_green_mat_times_hirsh();																				// obtain full space-time Green's matrix - all of them
		void cal_green_mat_times_hirsh_cycle();																			// use wrapping for calculating space time Green's matrix - some of them only
		void unequalG_greaterFirst(int t1, int t2, const arma::mat& inv_series_up, const arma::mat& inv_series_down);
		void unequalG_greaterLast(int t1, int t2, const arma::mat& inv_series_up, const arma::mat& inv_series_down);

		/// comparison & stability tests
		void compare_green_direct(int tim, double toll, bool print_greens) override;

		// -------------------------- HEAT-BATH
		void heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens = false) override;						// equalibrate the model
		int heat_bath_single_step(int lat_site) override;																// single step with lattice updating
		void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) override;									// collect averages
		void av_single_step(int current_elem_i, int sign, bool times) override;											// single step of averaging
	public:
		// -------------------------- CONSTRUCTORS
		/**
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
		HubbardQR(const v_1d<double>& t, double dtau, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice, int threads = 1, bool ct = false);
	};
}

#endif