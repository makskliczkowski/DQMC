#pragma once
#ifndef LATTICE_H
#include "../lattices.h"
#endif // !LATTICE_H


// -------------------------------------------------------- HEXAGONAL LATTICE --------------------------------------------------------
#ifndef HEXAGONAL_H
#define HEXAGONAL_H

class HexagonalLattice : public Lattice {
private:
	// elementary cells numbering
	int Lx;																												// spatial x-length
	int Ly;																												// spatial y-length
	int Lz;																												// spatial z-length
public:
	// CONSTRUCTORS
	~HexagonalLattice() = default;
	HexagonalLattice() = default;
	HexagonalLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0);											// general constructor

	// GETTERS
	int get_Lx() const override { return this->Lx; };
	int get_Ly() const override { return this->Ly; };
	int get_Lz() const override { return this->Lz; };
	int get_norm(int x, int y, int z) const override { return this->spatialNorm[abs(x)][abs(y)][abs(z)]; };

	// CALCULATORS
	void calculate_nn_pbc() override;
	void calculate_nn_obc() override;
	void calculate_nnn_pbc() override;
	void calculate_coordinates() override;

	// SYMMETRIES
	std::tuple<int, int, int> getNumElems() override {
		return std::make_tuple(this->Lx, 2 * this->Ly, this->Lz);
	}

	std::tuple<int, int, int> getSymPos(int x, int y, int z) override {
		return std::make_tuple(abs(x), abs(y), abs(z));
	}

	bool symmetry_checker(int xx, int yy, int zz) override {
		return true;
	};
};


#endif // ! HEXAGONAL_H