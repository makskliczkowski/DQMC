#pragma once
#ifndef GENERAL_H
#define GENERAL_H

#include "../src/user_interface.h"
#include "random.h"
#include "../src/common.h"
#include <iostream>
#include <vector>
#include <armadillo>






// In this file we define the virtual class for Monte Carlo simulations models of condensed matter systems


struct averages_par{
	// Specific heat related parameters
	double C;																// specific heat
	double sd_C;															// standard deviation C
	// Magnetic susceptibility related parameters
	double Xi;																// suscability
	double sd_Xi;															// standard deviation suscability
	// Magnetisation related parameters
	double av_M;															// average M
	double sd_M;															// standard deviation M
	double av_M2;															// average M^2
	double sd_M2;															// standard deviation of M^2
	double av_M2z;															// average squared z-th magnetization component 
	double sd_M2z;															// sd of it
	double av_M2x;															// averaxe squared mg x-th component
	double sd_M2x;															// sd of it
	v_3d<double> av_M2z_corr;												// equal time magnetic correlation 
	//timeCorrel av_M2z_corr_uneqTime;											// unequal time magnetic correlation 
	//std::vector<std::vector<std::vector<double>>>spin_structure_factor;			// equal time magnetic structure factor
	// Charge related parameters 
	v_3d<double> av_ch2_corr;														// equal time charge correlations
	//timeCorrel av_Charge2_corr_uneqTime;										// unequal time charge correlations
	// Energy related parameters 
	double av_E;															// average E
	double av_E2;															// average E^2
	double sd_E;															// standard deviation E
	double av_Ek;															// average kinetic energy
	double av_Ek2;															// average square of kinetic energy
	double sd_Ek;															// std of kinetic energy
	
	// Occupation related parameters
	double av_sign;															// average sign for probabilities
	double sd_sign;															// sd of it
	double av_occupation;													// average site ocupation -> varies from 0 to 2 for one band fermions
	double sd_occupation;													// sd of it
	// Green functions
	arma::mat av_gr_up;														// average Green up matrix
	arma::mat av_gr_down;													// average Green down matrix
	//timeCorrel av_green_up;		// spin up average of Green's function correlation in different times
	//timeCorrel av_green_down;	// spin down average of Green's function correlation in different times
	v_3d<double> av_occupation_corr;										// \sum _sigma <c_jsigma c_isigma> -> AVERAGE OF EQUAL TIME GREEN FUNCTION ELEMENTS
} ;




/* GENERAL LATTICE */
/// <summary>
/// Pure virtual lattice class, it will allow to distinguish between different geometries in the model
/// </summary>
class Lattice{
protected:
	/* LATTICE PARAMETERS */
	unsigned int dimension;										// the dimensionality of the lattice 1,2,3
	unsigned int Ns;											// number of lattice sites
	int boundary_conditions;									// boundary conditions 0 = PBC, 1 = OBC
	//impDef::lattice_types lattice_type;							// the type of lattice this is
	v_2d<int> nearest_neighbors;								// vector of the nearest neighbors
	v_2d<int> next_nearest_neighbors;							// vector of the next nearest neighbors
	v_2d<int> coordinates;										// vector of real coordiates allowing to get the distance between lattice points
	

public:
	virtual ~Lattice() = default;
	/* VIRTUAL GETTERS */
	virtual int get_Lx() = 0;
	virtual int get_Ly() = 0;
	virtual int get_Lz() = 0;
	/* GETTERS */
	int get_Ns() const;
	int get_Dim() const;
	int get_nn(int lat_site, int nei_num) const;				// returns given nearest nei at given lat site 
	int get_nnn(int lat_site, int nei_num) const;				// returns given next nearest nei at given lat site 
	int get_nn_number(int lat_site) const;						// returns the number of nn
	int get_nnn_number(int lat_site) const;						// returns the number of nnn
	int get_coordinates(int lat_site, int axis) const;			// returns the given coordinate
	/* CALCULATORS */
	virtual void calculate_nn_pbc() = 0;
	virtual void calculate_nnn_pbc() = 0;
	virtual void calculate_coordinates() = 0;

};

/// <summary>
/// A general abstract class for models on a lattice that use Monte Carlo 
/// </summary>
class LatticeModel{
protected:
	/* Certain physical parameters */
	double T;
	double beta;												// in kB units
	unsigned int Ns;											// number of lattice sites
	randomGen ran;												// consistent quick random number generator
	std::shared_ptr<Lattice> lattice;							// contains all the information about the lattice
	averages_par avs;											// structure containing all the averages
public:
	virtual ~LatticeModel() = default;							// pure virtual destructor
		
	/* CALCULATORS */
	virtual void relaxation(impDef::algMC algorithm, int mc_steps, bool conf, bool quiet) = 0;
	virtual void average(impDef::algMC algorithm,int corr_time, int avNum, int bootStraps, bool quiet, int times = 0) = 0;
		
	/* GETTERS */
	int getDim() const;
	int getNs() const;
	double getT() const;

	/* SETTERS */


};





#endif