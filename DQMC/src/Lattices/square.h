#pragma once
#ifndef LATTICE_H
#include "../lattices.h"
#endif // !LATTICE_H

// -------------------------------------------------------- SQUARE LATTICE --------------------------------------------------------

#ifndef SQUARE_H
#define SQUARE_H
class SquareLattice : public Lattice {
private:
	int Lx;																												// spatial x-length
	int Ly;																												// spatial y-length
	int Lz;																												// spatial z-length
public:
	// CONSTRUCTORS
	~SquareLattice() = default;	
	SquareLattice() = default;
	SquareLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0);											// general constructor

	// GETTERS
	int get_Lx() const override { return this->Lx; };
	int get_Ly() const override { return this->Ly; };
	int get_Lz() const override { return this->Lz; };
	int get_norm(int x, int y, int z) const override { return this->spatialNorm[x][y][z]; };

	// CALCULATORS
	void calculate_nn_pbc() override;
	void calculate_nn_obc() override;
	void calculate_nnn_pbc() override;
	void calculate_coordinates() override;
};

#endif // !SQUARE_H