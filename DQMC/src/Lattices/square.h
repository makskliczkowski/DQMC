#pragma once
#ifndef LATTICE_H
#include "../lattices.h"
#endif // !LATTICE_H

// -------------------------------------------------------- SQUARE LATTICE --------------------------------------------------------

#ifndef SQUARE_H
#define SQUARE_H
class SquareLattice : public Lattice {
private:
	bool symmetry = false;																								// if we shall include symmetry in saving greens
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
	
	// SYMMETRIES
	std::tuple<int, int, int> getNumElems() override {
		if(!this->symmetry)
			return std::make_tuple(this->Lx, this->Ly, this->Lz);

		switch (this->_BC)
		{
		//case 0:
		//	return std::make_tuple(this->Lx / 2, this->Ly / 2, this->Lz / 2);
		//	break;
		default:
			return std::make_tuple(this->Lx, this->Ly, this->Lz);
			break;
		}
	}
	
	std::tuple<int, int, int> getSymPos(int x, int y, int z) override {
		if (!this->symmetry)
			return std::make_tuple(abs(x), abs(y), abs(z));
		
	}

	bool symmetry_checker(int xx, int yy, int zz) override {
		return
			(xx <= this->Lx / 2 && xx >= 0) &&
			(yy <= this->Ly / 2 && yy >= 0) &&
			(zz <= this->Lz / 2 && zz >= 0);
	};

};

#endif // !SQUARE_H