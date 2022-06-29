#include "../../src/Lattices/square.h"


/*
* @brief Constructor for the square lattice
*/
SquareLattice::SquareLattice(int Lx, int Ly, int Lz, int dim, int _BC)
	: Lx(Lx), Ly(Ly), Lz(Lz)
{
	this->dim = dim;
	this->_BC = _BC;
	this->type = "square";
	// fix sites depending on _BC
	switch (this->dim)
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

	this->calculate_nn();
	this->calculate_coordinates();
	this->calculate_spatial_norm(Lx, Ly, Lz);

}



/*
* @brief Calculate the nearest neighbors with PBC
*/
void SquareLattice::calculate_nn_pbc()
{
	switch (this->dim)
	{
	case 1:
		// One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = myModuloEuclidean(i + 1, Lx);											// right
			this->nearest_neighbors[i][1] = myModuloEuclidean(i - 1, Lx);											// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			this->nearest_neighbors[i][0] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i + 1, Lx);		// right
			this->nearest_neighbors[i][1] = static_cast<int>(1.0 * i / Lx) * Lx + myModuloEuclidean(i - 1, Lx);		// left
			this->nearest_neighbors[i][2] = myModuloEuclidean(i + Lx, Ns);											// top
			this->nearest_neighbors[i][3] = myModuloEuclidean(i - Lx, Ns);											// bottom
		}
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the nearest neighbors with OBC
*/
void SquareLattice::calculate_nn_obc()
{
	switch (this->dim)
	{
	case 1:
		//* One dimension 
		this->nearest_neighbors = std::vector<std::vector<int>>(Lx, std::vector<int>(2, 0));
		for (int i = 0; i < Lx; i++) {
			this->nearest_neighbors[i][0] = (i + 1) >= Lx ? -1 : i + 1;										// right
			this->nearest_neighbors[i][1] = (i - 1) == 0 ? -1 : i - 1;										// left
		}
		break;
	case 2:
		// Two dimensions 
		/* numeration begins from the bottom left as 0 to the top right as N-1 with a snake like behaviour */
		this->nearest_neighbors = std::vector<std::vector<int>>(Ns, std::vector<int>(4, 0));
		for (int i = 0; i < Ns; i++) {
			this->nearest_neighbors[i][0] = (i + 1) < Lx ? static_cast<int>(1.0 * i / Lx) * Lx + i + 1 : -1;		// right
			this->nearest_neighbors[i][1] = (i - 1) >= 0 ? static_cast<int>(1.0 * i / Lx) * Lx + i - 1 : -1;		// left
			this->nearest_neighbors[i][2] = i + Lx < Ns ? i + Lx : -1;												// top
			this->nearest_neighbors[i][3] = i - Lx >= 0 ? i - Lx : -1;												// bottom
		}
		break;
	case 3:
		/* Three dimensions */
		break;
	default:
		break;
	}
}

/*
* @brief Calculate the next nearest neighbors with PBC
*/
void SquareLattice::calculate_nnn_pbc()
{
	switch (this->dim)
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

/*
* @brief Returns real space coordinates from a lattice site number
*/
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