
#ifndef COMMON_H
#include "common.h"
#endif

#ifndef LATTICE_H
#define LATTICE_H

using namespace std;


// -------------------------------------------------------- GENERAL LATTICE --------------------------------------------------------



/*
* Pure virtual lattice class, it will allow to distinguish between different geometries in the models
*/
class Lattice {
protected:
	// ----------------------- LATTICE PARAMETERS



	unsigned int dim = 1;											// the dimensionality of the lattice 1,2,3
	unsigned int Ns = 1;											// number of lattice sites
	string type;													// type of the lattice
	int _BC = 0;													// boundary conditions 0 = PBC, 1 = OBC
	v_2d<int> nearest_neighbors;									// vector of the nearest neighbors
	uint nearest_nei_forward_num;									// number of nearest neighbors forward

	v_2d<int> next_nearest_neighbors;								// vector of the next nearest neighbors
	v_2d<int> coordinates;											// vector of real coordiates allowing to get the distance between lattice points
	v_3d<int> spatialNorm;											// norm for averaging over all spatial sites

	// reciprocal vectors
	vec a1, a2, a3;
	mat k_vectors;													// allowed values of k - to be used in the lattice
public:
	virtual ~Lattice() = default;
	// ----------------------- VIRTUAL GETTERS
	virtual int get_Lx() const = 0;
	virtual int get_Ly() const = 0;
	virtual int get_Lz() const = 0;
	t_3d<int> getSiteDifference(t_3d<int> i, uint j) const;
	t_3d<int> getSiteDifference(uint i, uint j) const;

	// ----------------------- GETTERS
	virtual vec get_real_space_vec(int x, int y, int z)		const = 0;
	virtual int get_norm(int x, int y, int z)				const = 0;
	virtual v_1d<uint> get_nn_forward_number(int lat_site)	const = 0;
	auto get_Ns()											const RETURNS(this->Ns);												// returns the number of sites
	auto get_Dim()											const RETURNS(this->dim);												// returns dimension of the lattice
	auto get_nn(int lat_site, int nei_num)					const RETURNS(this->nearest_neighbors[lat_site][nei_num]);				// returns given nearest nei at given lat site
	auto get_nnn(int lat_site, int nei_num)					const RETURNS(this->next_nearest_neighbors[lat_site][nei_num]);			// returns given next nearest nei at given lat site
	auto get_nn_number(int lat_site)						const RETURNS(this->nearest_neighbors[lat_site].size());				// returns the number of nn
	auto get_nnn_number(int lat_site)						const RETURNS(this->next_nearest_neighbors[lat_site].size());			// returns the number of nnn
	//auto get_coordinates(int lat_site)						const RETURNS(make_tuple(this->coordinates[lat_site][0], this->coordinates[lat_site][1], this->coordinates[lat_site][2]));
	auto get_coordinates(int lat_site, int axis)			const RETURNS(this->coordinates[lat_site][axis]);						// returns the given coordinate
	auto get_spatial_norm()									const RETURNS(this->spatialNorm);										// returns the spatial norm
	auto get_spatial_norm(int x, int y, int z)				const RETURNS(this->spatialNorm[x][y][z]);								// returns the spatial norm at x,y,z
	auto get_type()											const RETURNS(this->type);												// returns the type of the lattice as a string
	auto get_info()											const RETURNS(VEQ(type) + "," + VEQ(_BC) + "," + VEQ(dim) + "," + VEQ(Ns) + "," + VEQ(get_Lx()) + "," + VEQ(get_Ly()) + "," + VEQ(get_Lz()));
	auto get_k_vectors()									const RETURNS(this->k_vectors);											// returns all k vectors in the RBZ
	auto get_k_vectors(uint row)							RETURNS(this->k_vectors.row(row));								// returns the given k vector row
	auto get_nei(int lat_site, int corr_len)				const;


	// ----------------------- CALCULATORS
	void calculate_nn();
	void calculate_spatial_norm();
	virtual void calculate_nn_pbc() = 0;
	virtual void calculate_nn_obc() = 0;
	virtual void calculate_nnn_pbc() = 0;
	virtual void calculate_coordinates() = 0;

	// ----------------------- SYMMETRY
	virtual t_3d<int> getNumElems() = 0;																							// returns the number of elements if the symmetry is possible
	virtual t_3d<int> getSymPosInv(int x, int y, int z) = 0;																		// from symmetrised form return coordinates
	virtual t_3d<int> getSymPos(int x, int y, int z) = 0;																			// from given coordinates return their symmetrised form

	virtual bool symmetry_checker(int xx, int yy, int zz) = 0;

private:
	virtual void calculate_k_vectors() = 0;
};

/*
* @brief calculates the nearest neighbors
*/
inline void Lattice::calculate_nn() {
	switch (this->_BC)
	{
	case 0:
		this->calculate_nn_pbc();
		stout << "->Using PBC" << EL;
		//this->calculate_nnn_pbc();
		break;
	case 1:
		this->calculate_nn_obc();
		stout << "->Using OBC" << EL;
		break;
	default:
		this->calculate_nn_pbc();
		break;
	}
}

/*
* @brief calculates the spatial repetition of difference between the lattice sites considering _BC and enumeration
*/
inline void Lattice::calculate_spatial_norm()
{
	// spatial norm
	auto [x_n, y_n, z_n] = this->getNumElems();
	this->spatialNorm = SPACE_VEC(x_n, y_n, z_n);

	for (int i = 0; i < this->Ns; i++) {
		for (int j = 0; j < this->Ns; j++) {
			const auto [xx, yy, zz] = this->getSiteDifference(i, j);
			auto [a, b, c] = this->getSymPos(xx, yy, zz);
			spatialNorm[a][b][c]++;
		}
	}
}

/*
* @brief gets the neighbor from a given lat_site lattice site at corr_len length
*/
inline auto Lattice::get_nei(int lat_site, int corr_len) const
{
	switch (this->_BC) {
	case 0:
		return myModuloEuclidean((lat_site + corr_len), this->Ns);
		break;
	case 1:
		return (lat_site + corr_len) > this->Ns ? -1 : (lat_site + corr_len);
		break;
	default:
		return myModuloEuclidean((lat_site + corr_len), this->Ns);
	}
}

/*
* @brief Returns the real space difference between lattice site cooridinates given in ascending order.
* From left to right. Then second row left to right etc.
* @param i First coordinate
* @param j Second coordinate
* @return Three-dimensional tuple (vector of vec[i]-vec[j])
*/
inline t_3d<int> Lattice::getSiteDifference(t_3d<int> i, uint j) const
{	
	auto [x1, y1, z1] = i;
	const int z = z1 - this->get_coordinates(j, 2);
	const int y = y1 - this->get_coordinates(j, 1);
	const int x = x1 - this->get_coordinates(j, 0);
	// returns the site difference
	return std::tuple<int, int, int>(x, y, z);
}

/*
* @brief Returns the real space difference between lattice site cooridinates given in ascending order.
* From left to right. Then second row left to right etc.
* @param i First coordinate
* @param j Second coordinate
* @return Three-dimensional tuple (vector of vec[i]-vec[j])
*/
inline t_3d<int> Lattice::getSiteDifference(uint i, uint j) const
{
	const int z = this->get_coordinates(i, 2) - this->get_coordinates(j, 2);
	const int y = this->get_coordinates(i, 1) - this->get_coordinates(j, 1);
	const int x = this->get_coordinates(i, 0) - this->get_coordinates(j, 0);
	// returns the site difference
	return std::tuple<int, int, int>(x, y, z);
}

#endif // !LATTICE_H