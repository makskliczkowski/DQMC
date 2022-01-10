#include "include/lattices.h"
// -------------------------------------------------------- SQUARE LATTICE --------------------------------------------------------

SquareLattice::SquareLattice(int Lx, int Ly, int Lz, int dim, int bc)
{
	this->dimension = dim;
	this->Lx = Lx; this->Ly = Ly; this->Lz = Lz;
	this->boundary_conditions = bc;
	this->type = "square";
	switch (this->dimension)
	{
	case 1:
		this->Ly = 1; this->Lz = 1;
		break;
	case 2:
		this->Lz = 1;
		break;
	default:
		break;
	}
	this->Ns = this->Lx * this->Ly * this->Lz;

	switch (this->boundary_conditions)
	{
	case 0:
		this->calculate_nn_pbc();
		//this->calculate_nnn_pbc();
	default:
		break;
	}
	this->calculate_coordinates();
	// spatial norm
	this->spatialNorm = v_3d<int>(2 * Lx - 1, v_2d<int>(2 * Ly - 1, v_1d<int>(2 * Lz - 1, 0)));
	for (int i = 0; i < this->Ns; i++) {
		for (int j = 0; j < this->Ns; j++) {
			const auto [x, y, z] = this->getSiteDifference(i, j);
			spatialNorm[x][y][z]++;
		}
	}
}
/**
 * @brief Returns the real space difference between lattice site cooridinates given in ascending order.
 * From left to right. Then second row left to right etc.
 * @param i First coordinate
 * @param j Second coordinate
 * @return Three-dimensional tuple (vector of vec[i]-vec[j])
 */
std::tuple<int, int, int> SquareLattice::getSiteDifference(uint i, uint j) const
{
	const int z = this->get_coordinates(i, 2) - this->get_coordinates(j, 2);
	const int y = this->get_coordinates(i, 1) - this->get_coordinates(j, 1);
	const int x = this->get_coordinates(i, 0) - this->get_coordinates(j, 0);
	return std::tuple<int, int, int>(x + Lx - 1, y + Ly - 1, z + Lz - 1);
}

/// <summary>
/// Calculate the nearest neighbors with PBC
/// </summary>
void SquareLattice::calculate_nn_pbc()
{
	switch (this->dimension)
	{
	case 1:
		/* One dimension */
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = myModuloEuclidean(i + 1, Lx);											// right
			this->nearest_neighbors[i][1] = myModuloEuclidean(i - 1, Lx);											// left
		}
		break;
	case 2:
		/* Two dimensions */
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			this->nearest_neighbors[i][0] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i + 1, Lx);		// right
			this->nearest_neighbors[i][1] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i - 1, Lx);		// left
			this->nearest_neighbors[i][2] = myModuloEuclidean(i + Lx, Ns);											// bottom
			this->nearest_neighbors[i][3] = myModuloEuclidean(i - Lx, Ns);											// top
		}
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}
/// <summary>
/// Calculate the next nearest neighbors with PBC
/// </summary>
void SquareLattice::calculate_nnn_pbc()
{
	switch (this->dimension)
	{
	case 1:
		/* One dimension */
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = myModuloEuclidean(i + 2, Lx);											// right
			this->nearest_neighbors[i][1] = myModuloEuclidean(i - 2, Lx);											// left
		}
		break;
	case 2:
		/* Two dimensions */
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}
/// <summary>
/// Returns real space coordinates from a lattice site number
/// </summary>
void SquareLattice::calculate_coordinates()
{
	const int LxLy = Lx * Ly;
	this->coordinates = v_2d<int>(this->Ns, v_1d<int>(3, 0));
	for (int i = 0; i < Ns; i++) {
		this->coordinates[i][0] = i % Lx;												// x axis coordinate
		this->coordinates[i][1] = (static_cast<int>(1.0 * i / Lx)) % Ly;				// y axis coordinate
		this->coordinates[i][2] = (static_cast<int>(1.0 * i / (LxLy))) % Lz;			// z axis coordinate
		//std::cout << "(" << this->coordinates[i][0] << "," << this->coordinates[i][1] << "," << this->coordinates[i][2] << ")\n";
	}
}