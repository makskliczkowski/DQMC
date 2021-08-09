#pragma once
#ifndef GENERAL_H
#define GENERAL_H

#include "../src/user_interface.h"
#include "random.h"
#include <iostream>
#include <vector>






// In this file we define the virtual class for Monte Carlo simulations models of condensed matter systems

namespace generalModel{

	using namespace generalModel;

	/* GENERAL LATTICE */
	/// <summary>
	/// Pure virtual lattice class, it will allow to distinguish between different geometries in the model
	/// </summary>
	class Lattice{
	private:
		/* LATTICE PARAMETERS */
		int dimension;												// the dimensionality of the lattice 1,2,3
		int Ns;														// number of lattice sites
		//impDef::lattice_types lattice_type;							// the type of lattice this is
		std::vector<std::vector<int>> nearest_neighbors;			// vector of the nearest neighbors
		std::vector<std::vector<int>> next_nearest_neighbors;		// vector of the next nearest neighbors

	public:
		virtual ~Lattice() = 0;
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

		/* CALCULATORS */
		virtual void calculate_nn_pbc() = 0;
		virtual void calculate_nnn_pbc() = 0;

	};

	/// <summary>
	/// A general abstract class for models on a lattice that use Monte Carlo 
	/// </summary>
	class LatticeModel{
	private:
		/* Certain physical parameters */
		double T;
		double beta;												// in kB units
		int Ns;														// number of lattice sites
		randomGen ran;												// consistent quick random number generator
		std::shared_ptr<Lattice> lattice;							// contains all the information about the lattice
	public:
		virtual ~LatticeModel() = 0;								// pure virtual destructor
		
		/* CALCULATORS */
		virtual void relaxation(impDef::algMC algorithm, int mc_steps, bool conf, bool quiet) = 0;
		virtual void average(impDef::algMC algorithm,int corr_time, int avNum, int bootStraps, bool quiet) = 0;
		
		/* GETTERS */
		int getDim() const;
		int getNs() const;
		double getT() const;

		/* SETTERS */


	};






}





#endif