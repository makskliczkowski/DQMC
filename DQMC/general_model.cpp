#include "include/general_model.h"

/* ------------------------------------------- GENERAL MODEL -------------------------------------------*/

/// <summary>
/// Returns the dimension of the lattice model
/// </summary>
/// <returns> Dimension size </returns>
int LatticeModel::getDim() const
{
	return this->lattice->get_Dim();
}
/// <summary>
/// Returns the number of lattice sites
/// </summary>
/// <returns> Number of Lattice sites </returns>
int LatticeModel::getNs() const
{
	return this->lattice->get_Ns();
}
/// <summary>
/// Returns the temperature
/// </summary>
/// <returns>Temperature</returns>
double LatticeModel::getT() const
{
	return this->T;
}

/* ------------------------------------------- LATTICE -------------------------------------------*/

int Lattice::get_Ns() const
{
	return this->Ns;
}

int Lattice::get_Dim() const
{
	return this->dimension;
}
/// <summary>
/// Get the "nei_num" nearest neighbor at "lat_site" position
/// </summary>
/// <param name="lat_site">a position on lattice</param>
/// <param name="nei_num">a neighbor number</param>
/// <returns>nei_num nearest neighbor at lat_site</returns>
int Lattice::get_nn(int lat_site, int nei_num) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	if(nei_num > this->nearest_neighbors[lat_site].size()) throw "Not having this nn \n";
	return this->nearest_neighbors[lat_site][nei_num];
}
/// <summary>
/// Get the "nei_num" next nearest neighbor at "lat_site" position
/// </summary>
/// <param name="lat_site">a position on lattice</param>
/// <param name="nei_num">a neighbor number</param>
/// <returns>nei_num next nearest neighbor at lat_site</returns>
int Lattice::get_nnn(int lat_site, int nei_num) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	if(nei_num > this->next_nearest_neighbors[lat_site].size()) throw "Not having this nnn \n";
	return this->next_nearest_neighbors[lat_site][nei_num];
}
/// <summary>
/// Get the number of nn at given lattice site
/// </summary>
/// <param name="lat_site">The lattice site </param>
/// <returns>the number of nn at given lattice site</returns>
int Lattice::get_nn_number(int lat_site) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	return this->nearest_neighbors[lat_site].size();
}
/// <summary>
/// Get the number of nnn at given lattice site
/// </summary>
/// <param name="lat_site">The lattice site </param>
/// <returns>the number of nn at given lattice site</returns>
int Lattice::get_nnn_number(int lat_site) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	return this->next_nearest_neighbors[lat_site].size();
}
/// <summary>
/// Returns given coordinate on given axis
/// </summary>
/// <param name="lat_site">lattice site</param>
/// <param name="axis">axis = x,y,z</param>
/// <returns>given coordinate on given axis</returns>
int Lattice::get_coordinates(int lat_site, int axis) const
{
	return this->coordinates[lat_site][axis];
}
