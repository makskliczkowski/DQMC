#pragma once
#ifndef GENERAL_H
#define GENERAL_H

#include "../src/progress.h"
#include "random.h"
#include <chrono>
#include <stdlib.h>

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
* Pure virtual lattice class, it will allow to distinguish between different geometries in the models
*/
class Lattice {
protected:
	// ----------------------- LATTICE PARAMETERS
	unsigned int dimension;											// the dimensionality of the lattice 1,2,3
	unsigned int Ns;												// number of lattice sites
	string type;													// type of the lattice
	int boundary_conditions;										// boundary conditions 0 = PBC, 1 = OBC
	v_2d<int> nearest_neighbors;									// vector of the nearest neighbors
	v_2d<int> next_nearest_neighbors;								// vector of the next nearest neighbors
	v_2d<int> coordinates;											// vector of real coordiates allowing to get the distance between lattice points
	v_3d<int> spatialNorm;											// norm for averaging over all spatial sites

public:
	virtual ~Lattice() = default;
	// ----------------------- VIRTUAL GETTERS
	virtual int get_Lx() const = 0;
	virtual int get_Ly() const = 0;
	virtual int get_Lz() const = 0;
	virtual std::tuple<int, int, int> getSiteDifference(uint i, uint j) const = 0;
	// ----------------------- GETTERS
	virtual int get_norm(int x, int y, int z) const = 0;
	int get_Ns() const { return this->Ns; };
	int get_Dim() const { return this->dimension; };
	int get_nn(int lat_site, int nei_num) const 
		{ return this->nearest_neighbors[lat_site][nei_num]; };		// returns given nearest nei at given lat site
	int get_nnn(int lat_site, int nei_num) const 
		{ return this->next_nearest_neighbors[lat_site][nei_num]; };// returns given next nearest nei at given lat site
	int get_nn_number(int lat_site) const 
		{ return this->nearest_neighbors[lat_site].size(); };		// returns the number of nn
	int get_nnn_number(int lat_site) const 
		{ return this->next_nearest_neighbors[lat_site].size(); };	// returns the number of nnn
	int get_coordinates(int lat_site, int axis) const 
		{ return this->coordinates[lat_site][axis]; };				// returns the given coordinate
	string get_type() const 
		{ return this->type; };										// returns the type of the lattice as a string

	// ----------------------- CALCULATORS
	virtual void calculate_nn_pbc() = 0;
	virtual void calculate_nnn_pbc() = 0;
	virtual void calculate_coordinates() = 0;
};


/* 
* Structure for storing the averages from the Quantum Monte Carlo simulation
*/
struct averages_par {
	averages_par(int Lx, int Ly, int Lz, int M = 1, bool times = false) {
		// Correlations - depend on the dimension - equal time
		this->av_occupation_corr = v_3d<double>(2 * Lx - 1, v_2d<double>(2 * Ly - 1, v_1d<double>(2 * Lz - 1, 0.0)));
		this->av_M2z_corr = v_3d<double>(2 * Lx - 1, v_2d<double>(2 * Ly - 1, v_1d<double>(2 * Lz - 1, 0.0)));
		this->av_ch2_corr = v_3d<double>(2 * Lx - 1, v_2d<double>(2 * Ly - 1, v_1d<double>(2 * Lz - 1, 0.0)));
		// Setting av Greens
		if (times) {
			this->g_up_diffs = v_1d<mat>(M, arma::zeros(Lx / 2 + 1, Ly / 2 + 1));
			this->g_down_diffs = v_1d<mat>(M, arma::zeros(Lx / 2 + 1, Ly / 2 + 1));
			this->sd_g_up_diffs = v_1d<mat>(M, arma::zeros(Lx / 2 + 1, Ly / 2 + 1));
			this->sd_g_down_diffs = v_1d<mat>(M, arma::zeros(Lx / 2 + 1, Ly / 2 + 1));
			this->g_up_diffs_k = v_1d<cx_mat>(M, cx_mat(Lx, Ly, fill::zeros));
			this->g_down_diffs_k = v_1d<cx_mat>(M, cx_mat(Lx, Ly, fill::zeros));
			this->sd_g_up_diffs_k = v_1d<cx_mat>(M, cx_mat(Lx, Ly, fill::zeros));
			this->sd_g_down_diffs_k = v_1d<cx_mat>(M, cx_mat(Lx, Ly, fill::zeros));
		}
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
		for (int tau = 0; tau < M; tau++) {
			for (int x = 0; x < this->g_up_diffs[tau].n_rows; x++) {
				for (int y = 0; y < this->g_up_diffs[tau].n_cols; y++) {
					const auto norm = bucketNum * (all ? -double(M) : -1.0*(M-tau)) * lat->get_norm(x, y, 0);
					this->g_up_diffs[tau](x, y) /= norm;
					this->g_down_diffs[tau](x, y) /= norm;
					this->sd_g_up_diffs[tau](x, y) = variance(this->sd_g_up_diffs[tau](x, y), this->g_up_diffs[tau](x, y), norm);
					this->sd_g_down_diffs[tau](x, y) = variance(this->sd_g_down_diffs[tau](x, y), this->g_down_diffs[tau](x, y), norm);
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

/// <summary>
/// A general abstract class for models on a lattice that use Monte Carlo
/// </summary>
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
	virtual void average(impDef::algMC algorithm, int corr_time, int avNum, int bootStraps, bool quiet, int times = 0) = 0;

	// ----------------------- GETTERS
	int getDim() const { return this->lattice->get_Dim(); };
	int getNs() const { return this->Ns; };
	double getT() const { return this->T; };
	std::shared_ptr<averages_par> get_avs() const { return this->avs; };
	// ----------------------- SETTERS
};
#endif // !GENERAL_H