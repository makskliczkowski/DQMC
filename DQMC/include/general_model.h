#pragma once

//#include "../include/plog/Log.h"
//#include "../include/plog/Initializers/RollingFileInitializer.h"
#include "../src/common.h"
#include "random.h"

#include <chrono>
#include <armadillo>
#include <filesystem>
#include <stdlib.h>

// -------------------------------------------------------- armadillo definitions --------------------------------------------------------
#define ARMA_USE_WRAPPER
#define ARMA_BLAS_CAPITALS
#define ARMA_BLAS_UNDERSCORE
#define ARMA_BLAS_LONG
#define ARMA_BLAS_LONG_LONG
#define ARMA_USE_MKL_ALLOC
#define ARMA_USE_MKL_TYPES1
#define ARMA_DONT_USE_OPENMP

// In this file we define the virtual class for Monte Carlo simulations models of condensed matter systems

struct averages_par {
	averages_par(int Lx, int Ly, int Lz) {
		const int Ns = Lx * Ly * Lz;
		// Correlations - depend on the dimension - equal time
		this->av_occupation_corr = v_3d<double>(2 * Lx - 1, v_2d<double>(2 * Ly - 1, v_1d<double>(2 * Lz - 1, 0.0)));
		this->av_M2z_corr = this->av_occupation_corr;
		this->av_ch2_corr = this->av_occupation_corr;
		// Setting av Greens
		this->av_gr_down.zeros(Ns, Ns);
		this->av_gr_up.zeros(Ns, Ns);
	}

	// Specific heat related parameters
	double C = 0;																// specific heat
	double sd_C = 0;															// standard deviation C
	// Magnetic susceptibility related parameters
	double Xi = 0;																// suscability
	double sd_Xi = 0;															// standard deviation suscability
	// Magnetisation related parameters
	long double av_M = 0;															// average M
	long double sd_M = 0;															// standard deviation M
	long double av_M2 = 0;															// average M^2
	long double sd_M2 = 0;															// standard deviation of M^2
	long double av_M2z = 0;															// average squared z-th magnetization component
	long double sd_M2z = 0;															// sd of it
	long double av_M2x = 0;															// averaxe squared mg x-th component
	long double sd_M2x = 0;															// sd of it
	v_3d<double> av_M2z_corr;												// equal time magnetic correlation
	//timeCorrel av_M2z_corr_uneqTime;											// unequal time magnetic correlation
	//std::vector<std::vector<std::vector<double>>>spin_structure_factor;			// equal time magnetic structure factor
	// Charge related parameters
	v_3d<double> av_ch2_corr;												// equal time charge correlations
	//timeCorrel av_Charge2_corr_uneqTime;										// unequal time charge correlations
	// Energy related parameters
	long double av_E = 0;															// average E
	long double av_E2 = 0;															// average E^2
	long double sd_E = 0;															// standard deviation E
	long double av_Ek = 0;															// average kinetic energy
	long double av_Ek2 = 0;															// average square of kinetic energy
	long double sd_Ek = 0;															// std of kinetic energy

	// Occupation related parameters
	long double av_sign = 0;													// average sign for probabilities
	double sd_sign = 0;															// sd of it
	long double av_occupation = 0;													// average site ocupation -> varies from 0 to 2 for one band fermions
	double sd_occupation = 0;													// sd of it
	// Green functions
	arma::mat av_gr_up;														// average Green up matrix
	arma::mat av_gr_down;													// average Green down matrix
	//timeCorrel av_green_up;		// spin up average of Green's function correlation in different times
	//timeCorrel av_green_down;	// spin down average of Green's function correlation in different times
	v_3d<double> av_occupation_corr;										// \sum _sigma <c_jsigma c_isigma> -> AVERAGE OF EQUAL TIME GREEN FUNCTION ELEMENTS
};

// -------------------------------------------------------- GENERAL LATTICE --------------------------------------------------------
/// <summary>
/// Pure virtual lattice class, it will allow to distinguish between different geometries in the model
/// </summary>
class Lattice {
protected:
	// ----------------------- LATTICE PARAMETERS 
	unsigned int dimension;										// the dimensionality of the lattice 1,2,3
	unsigned int Ns;											// number of lattice sites
	std::string type;											// type of the lattice
	int boundary_conditions;									// boundary conditions 0 = PBC, 1 = OBC
	//impDef::lattice_types lattice_type;							// the type of lattice this is
	v_2d<int> nearest_neighbors;								// vector of the nearest neighbors
	v_2d<int> next_nearest_neighbors;							// vector of the next nearest neighbors
	v_2d<int> coordinates;										// vector of real coordiates allowing to get the distance between lattice points

public:
	virtual ~Lattice() = default;
	// ----------------------- VIRTUAL GETTERS 
	virtual int get_Lx() const = 0;
	virtual int get_Ly() const = 0;
	virtual int get_Lz() const = 0;

	// ----------------------- GETTERS 
	int get_Ns() const { return this->Ns; };
	int get_Dim() const { return this->dimension; };
	int get_nn(int lat_site, int nei_num) const { return this->nearest_neighbors[lat_site][nei_num]; };					// returns given nearest nei at given lat site
	int get_nnn(int lat_site, int nei_num) const { return this->next_nearest_neighbors[lat_site][nei_num]; };			// returns given next nearest nei at given lat site
	int get_nn_number(int lat_site) const { return this->nearest_neighbors[lat_site].size(); };							// returns the number of nn
	int get_nnn_number(int lat_site) const { return this->next_nearest_neighbors[lat_site].size(); };					// returns the number of nnn
	int get_coordinates(int lat_site, int axis) const { return this->coordinates[lat_site][axis]; };					// returns the given coordinate
	std::string get_type() const { return this->type; };																// returns the type of the lattice as a string

	// ----------------------- CALCULATORS 
	virtual void calculate_nn_pbc() = 0;
	virtual void calculate_nnn_pbc() = 0;
	virtual void calculate_coordinates() = 0;
};

// -------------------------------------------------------- LATTICE MODEL --------------------------------------------------------

/// <summary>
/// A general abstract class for models on a lattice that use Monte Carlo
/// </summary>
class LatticeModel {
protected:
	int inner_threads;											// threads for the inner loops
	// ----------------------- Certain physical parameters

	double T;
	double beta;												// in kB units
	unsigned int Ns;											// number of lattice sites
	randomGen ran;												// consistent quick random number generator
	std::shared_ptr<Lattice> lattice;							// contains all the information about the lattice
	std::shared_ptr<averages_par> avs;							// structure containing all the averages
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
