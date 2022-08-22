#pragma once
#ifndef GENERAL_H
#define GENERAL_H

#include "../src/progress.h"
#include "../src/statistical.h"
#include "../src/random.h"

#ifndef SQUARE_H
#include "../src/Lattices/square.h"
#endif
#ifndef HEXAGONAL_H
#include "../src/Lattices/hexagonal.h"
#endif

#include <chrono>
#include <stdlib.h>




#define USE_QR
//#define CAL_TIMES

#ifdef CAL_TIMES
#define USE_HIRSH
#define ALL_TIMES
#endif


using namespace std;
using namespace arma;
/* 
* In this file we define the virtual class for Monte Carlo simulations models of condensed matter systems
*/

struct general_directories {
	string LxLyLz;
	string lat_type;
	string working_dir;
	string info;
};

// -------------------------------------------------------- GENERAL LATTICE --------------------------------------------------------

/* 
* Structure for storing the averages from the Quantum Monte Carlo simulation
*/
struct averages_par {
	averages_par(const std::shared_ptr<Lattice>& lat, int M = 1, bool times = false) {
		auto [x_num, y_num, z_num] = lat->getNumElems();

		// Correlations - depend on the dimension - equal time
		this->av_occupation_corr = v_3d<double>(x_num, v_2d<double>(y_num, v_1d<double>(z_num, 0.0)));
		this->av_M2z_corr = v_3d<double>(x_num, v_2d<double>(y_num, v_1d<double>(z_num, 0.0)));
		this->av_ch2_corr = v_3d<double>(x_num, v_2d<double>(y_num, v_1d<double>(z_num, 0.0)));
		// Setting av Greens
#ifdef CAL_TIMES
			this->g_up_diffs = v_1d<mat>(M, arma::zeros(x_num, y_num));
			this->g_down_diffs = v_1d<mat>(M, arma::zeros(x_num, y_num));
			this->sd_g_up_diffs = v_1d<mat>(M, arma::zeros(x_num, y_num));
			this->sd_g_down_diffs = v_1d<mat>(M, arma::zeros(x_num, y_num));
#endif
	}
	//! ----------------- functions for Green's
	/**
	* @brief Resets the Green's functions
	*/
	void resetGreens() {
		for (int i = 0; i < g_up_diffs.size(); i++) {
			this->g_up_diffs[i].zeros();
			this->g_down_diffs[i].zeros();
			this->sd_g_up_diffs[i].zeros();
			this->sd_g_down_diffs[i].zeros();
			//this->g_up_diffs_k[i].zeros();
			//this->g_down_diffs_k[i].zeros();
			//this->sd_g_up_diffs_k[i].zeros();
			//this->sd_g_down_diffs_k[i].zeros();
		}
	}

	/**
	* @brief Normalizes the Green's given the model parameters and the lattice
	* @param lattice general lattice class -> allows the normalisation
	* @param bucketNum number of bucket on which the Green's are averaged
	*/
	void normaliseGreens(std::shared_ptr<Lattice>& lat, int bucketNum, bool all = true) {
		const auto M = this->g_up_diffs.size();
		auto [xx, yy, zz] = lat->getNumElems();
		for (int tau = 0; tau < M; tau++) {
			auto norm = bucketNum * (all ? -double(M) : -(1.0 * (M - tau)));
			for (int x = 0; x < xx; x++) {
				for (int y = 0; y < yy; y++) {
					const auto norm2 = norm * lat->get_norm(x, y, 0);
					this->g_up_diffs[tau](x, y) /= norm2;
					this->g_down_diffs[tau](x, y) /= norm2;
					this->sd_g_up_diffs[tau](x, y) = variance(this->sd_g_up_diffs[tau](x, y), this->g_up_diffs[tau](x, y), norm2);
					this->sd_g_down_diffs[tau](x, y) = variance(this->sd_g_down_diffs[tau](x, y), this->g_down_diffs[tau](x, y), norm2);
				}
			}
		}
	}

	// Specific heat related parameters
	double C = 0;															// specific heat
	double sd_C = 0;														// standard deviation C
	// Magnetic susceptibility related parameters
	double Xi = 0;															// suscability
	double sd_Xi = 0;														// standard deviation suscability
	// Magnetisation related parameters
	long double av_M = 0;													// average M
	long double sd_M = 0;													// standard deviation M
	long double av_M2 = 0;													// average M^2
	long double sd_M2 = 0;													// standard deviation of M^2
	long double av_M2z = 0;													// average squared z-th magnetization component
	long double sd_M2z = 0;													// sd of it
	long double av_M2x = 0;													// averaxe squared mg x-th component
	long double sd_M2x = 0;													// sd of it
	v_3d<double> av_M2z_corr;												// equal time magnetic correlation
	//timeCorrel av_M2z_corr_uneqTime;											// unequal time magnetic correlation
	//std::vector<std::vector<std::vector<double>>>spin_structure_factor;			// equal time magnetic structure factor
	// Charge related parameters
	v_3d<double> av_ch2_corr;												// equal time charge correlations
	//timeCorrel av_Charge2_corr_uneqTime;										// unequal time charge correlations
	// Energy related parameters
	long double av_E = 0;													// average E
	long double av_E2 = 0;													// average E^2
	long double sd_E = 0;													// standard deviation E
	long double av_Ek = 0;													// average kinetic energy
	long double av_Ek2 = 0;													// average square of kinetic energy
	long double sd_Ek = 0;													// std of kinetic energy

	// Occupation related parameters
	long double av_sign = 0;												// average sign for probabilities
	long double sd_sign = 0;												// sd of it
	long double av_occupation = 0;											// average site ocupation -> varies from 0 to 2 for one band fermions
	long double sd_occupation = 0;											// sd of it
	// Green functions
	v_1d<mat> g_up_diffs, g_down_diffs;
	v_1d<mat> sd_g_up_diffs, sd_g_down_diffs;
	v_1d<cx_mat> g_up_diffs_k, g_down_diffs_k;
	v_1d<cx_mat> sd_g_up_diffs_k, sd_g_down_diffs_k;

	v_3d<double> av_occupation_corr;										// \sum _sigma <c_jsigma c_isigma> -> AVERAGE OF EQUAL TIME GREEN FUNCTION ELEMENTS
};

// -------------------------------------------------------- LATTICE MODEL --------------------------------------------------------

/*
* A general abstract class for models on a lattice that use Monte Carlo
*/
class LatticeModel {
protected:
	int inner_threads;											// threads for the inner loops
	// ----------------------- Certain physical parameters
	int Lx, Ly, Lz;
	double T;
	double beta;												// in kB units
	unsigned int Ns;											// number of lattice sites
	randomGen ran;												// consistent quick random number generator
	std::shared_ptr<Lattice> lattice;							// contains all the information about the lattice
	std::shared_ptr<averages_par> avs;							// structure containing all the averages
	std::unique_ptr<pBar> pbar;									// for printing progress

public:
	virtual ~LatticeModel() = default;							// pure virtual destructor

	// ----------------------- CALCULATORS
	virtual void relaxation(impDef::algMC algorithm, int mc_steps, bool conf, bool quiet) = 0;
	virtual void average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet) = 0;

	// ----------------------- CHECKERS


	// ----------------------- GETTERS
	auto getDim()						const RETURNS(this->lattice->get_Dim());
	auto getNs()						const RETURNS(this->Ns);
	auto getT()							const RETURNS(this->T);
	auto get_avs()						const RETURNS(this->avs);
	// ----------------------- SETTERS
};
#endif // !GENERAL_H