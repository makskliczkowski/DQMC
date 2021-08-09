#pragma once
#ifndef LATTICES_H
#define LATTICES_H

#include "general_model.h"

class SquareLattice : public virtual generalModel::Lattice{
private:
	int Lx;																// spatial x-length
	int Ly;																// spatial y-length
	int Lz;																// spatial z-length
public:
	/* CONSTRUCTORS */
	~SquareLattice();
	SquareLattice() = default;
	SquareLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1);			// general constructor
	SquareLattice(const SquareLattice &A);								// copy constructor
	SquareLattice(SquareLattice&& A);									// move constructor
	/* GETTERS */
	int get_Lx() override;
	int get_Ly() override;
	int get_Lz() override;
	/* CALCULATORS */
	void calculate_nn_pbc() override;									
	void calculate_nnn_pbc() override;

};





#endif // !LATTICES_H
