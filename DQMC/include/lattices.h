#pragma once


#include "general_model.h"

class SquareLattice : public Lattice{
private:
	int Lx;																		// spatial x-length
	int Ly;																		// spatial y-length
	int Lz;																		// spatial z-length
public:
	/* CONSTRUCTORS */
	SquareLattice() = default;
	SquareLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int bc = 0);		// general constructor
	/* GETTERS */
	int get_Lx() const override;
	int get_Ly() const override;
	int get_Lz() const override;
	/* CALCULATORS */
	void calculate_nn_pbc() override;									
	void calculate_nnn_pbc() override;
	void calculate_coordinates() override;

};

