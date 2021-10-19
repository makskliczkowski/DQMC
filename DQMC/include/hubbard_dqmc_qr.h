#pragma once
#include "hubbard.h"

#ifndef HUBBARD_ST_H
#define HUBBARD_ST_H	
// ---------------------------------------------------------------------- SPACE TIME ----------------------------------------------------------------------------
namespace hubbard {


	/// <summary>
/// A DQMC model that uses the QR decomposition scheme
/// </summary>
	class HubbardQR : public hubbard::HubbardModel {
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

		std::vector<arma::mat> b_up_condensed, b_down_condensed;														// up and down B matrices vector premultiplied
		v_1d<arma::mat> g_up_eq, g_down_eq;																				// saved equal time Greens
		arma::mat g_up_time, g_down_time;																				// time displaced Greens

		// -------------------------- HELPING FUNCTIONS
		void sweep_0_M(std::function<int(int)> ptfptr, bool save_greens) override;																		// sweep forward in time
		void sweep_M_0(std::function<int(int)> ptfptr, bool save_greens) override;																		// sweep backwards
		int sweep_lat_sites(std::function<int(int)> fptr);

		// -------------------------- UPDATERS
		void upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down) override;
		void upd_next_green(int which_time) override;
		void upd_prev_green(int which_time) override;
		void upd_Green_step(int im_time_step, bool forward = true) override;
		void cal_B_mat_cond(int which_sector);

		// -------------------------- CALCULATORS
		void b_mat_multiplier_right(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);
		void b_mat_multiplier_left(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);
		void b_mat_multiplier_right_inv(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);
		void b_mat_multiplier_left_inv(int l_start, int l_end, arma::mat& tmp_up, arma::mat& tmp_down);
		// eq time
		void cal_green_mat(int which_time) override;
		void cal_green_mat_cycle(int sector);
		// times
		void cal_green_mat_times(); // give left time already
		void cal_green_mat_times_cycle();
		void cal_green_mat_times_hirsh();																				// obtain full space-time Green's matrix
		void cal_green_mat_times_hirsh_cycle();																			// use wrapping for calculating space time Green's matrix
		// comparison
		void compare_green_direct(int tim, double toll, bool print_greens) override;

		// -------------------------- HEAT-BATH
		int heat_bath_single_step(int lat_site) override;																// single step with updating
		int heat_bath_single_step_no_upd(int lat_site) override;														// single step without updating
		int heat_bath_single_step_conf(int lat_site) override;															// single step with saving configurations
		void heat_bath_eq(int mcSteps, bool conf, bool quiet) override;
		void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) override;
		void av_single_step(int current_elem_i, int sign) override;
	public:
		// -------------------------- CONSTRUCTORS
		HubbardQR(const std::vector<double>& t, double dtau, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice, int threads = 1);

	};
}


#endif