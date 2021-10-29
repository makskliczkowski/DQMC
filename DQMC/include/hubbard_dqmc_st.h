#pragma once
#include "hubbard.h"
#ifndef HUBBARD_QR_H
#define HUBBARD_QR_H

// ------------------------------------------------------------------------------------------- QR -------------------------------------------------------------------------------------------

namespace hubbard {
	/// <summary>
	/// A DQMC model that uses the Space-Time formulation scheme
	/// </summary>
	class HubbardST : public HubbardModel {
	private:
		int current_time_slice;
		int current_time_in_silce;
		int green_size;

		// -------------------------- DIRECTORIES ETC.

		// -------------------------- HELPING FUNCTIONS
		int total_time(int tim, int tim_sector) { return tim_sector * this->M_0 + tim; };
		void sweep_0_M(std::function<int(int)>, bool save_greens) override {};																		// sweep forward in time
		void sweep_M_0(std::function<int(int)>, bool save_greens) override {};																		// sweep backwards
		// -------------------------- UPDATERS
		void upd_equal_green(int lat_site, double gamma_over_prob_up, double gamma_over_prob_down) override;
		void upd_next_green(int which_time) override;
		void upd_prev_green(int which_time) override;
		void upd_Green_step(int im_time_step, bool forward) override;

		// -------------------------- CALCULATORS
		void cal_green_mat(int which_time) override;
		void compare_green_direct(int tim, double toll, bool print_greens) override;

		// -------------------------- HEAT-BATH
		int heat_bath_single_step(int lat_site) override;																// single step with updating
		int heat_bath_single_step_no_upd(int lat_site) override;														// single step without updating
		int heat_bath_single_step_conf(int lat_site) override;															// single step with saving configurations
		void heat_bath_eq(int mcSteps, bool conf, bool quiet, bool save_greens = false) override;
		void heat_bath_av(int corr_time, int avNum, bool quiet, bool times) override;
		void av_single_step(int current_elem_i, int sign) override;
	public:
		// -------------------------- CONSTRUCTORS
		HubbardST(const std::vector<double>& t, double dtau, int M_0, double U, double mu, double beta, std::shared_ptr<Lattice> lattice, int threads = 1);
	};
}
#endif



