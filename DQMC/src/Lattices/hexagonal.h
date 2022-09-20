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

	// lattice parameters
	double a = 1;
	double c = 1;
	

public:
	// CONSTRUCTORS
	~HexagonalLattice() = default;
	HexagonalLattice() = default;
	HexagonalLattice(int Lx, int Ly = 1, int Lz = 1, int dim = 1, int _BC = 0);											// general constructor

	// GETTERS
	int get_Lx()												const override { return this->Lx; };
	int get_Ly()												const override { return this->Ly; };
	int get_Lz()												const override { return this->Lz; };
	int get_norm(int x, int y, int z) const override			{ return this->spatialNorm[x][y][z]; };
	vec get_real_space_vec(int x, int y, int z)					const override;
	v_1d<uint> get_nn_forward_number(int lat_site)				const override;

	// CALCULATORS
	void calculate_nn_pbc() override;
	void calculate_nn_obc() override;
	void calculate_nnn_pbc() override;
	void calculate_coordinates() override;

	// SYMMETRIES
	t_3d<int> getNumElems() override {
		return std::make_tuple(2 * this->Lx - 1, 4 * this->Ly - 1, 2 * this->Lz - 1);
	}

	t_3d<int> getSymPos(int x, int y, int z) override {
		return std::make_tuple(x + Lx - 1, y + 2 * Ly - 1, z + Lz - 1);
	}

	t_3d<int> getSymPosInv(int x, int y, int z) override {
		return std::make_tuple(x - (Lx - 1), y - (2 * Ly - 1), z - (Lz - 1));
	}

	bool symmetry_checker(int xx, int yy, int zz) override {
		return true;
	};
private:
	void calculate_k_vectors() override;
};


#endif // ! HEXAGONAL_H