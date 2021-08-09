#include "include/general_model.h"

/* ------------------------------------------- GENERAL MODEL -------------------------------------------*/

/// <summary>
/// Returns the dimension of the lattice model
/// </summary>
/// <returns> Dimension size </returns>
int generalModel::LatticeModel::getDim() const
{
	return this->lattice->get_Dim();
}
/// <summary>
/// Returns the number of lattice sites
/// </summary>
/// <returns> Number of Lattice sites </returns>
int generalModel::LatticeModel::getNs() const
{
	return this->lattice->get_Ns();
}
/// <summary>
/// Returns the temperature
/// </summary>
/// <returns>Temperature</returns>
double generalModel::LatticeModel::getT() const
{
	return this->T;
}

/* ------------------------------------------- LATTICE -------------------------------------------*/

int generalModel::Lattice::get_Ns() const
{
	return this->Ns;
}

int generalModel::Lattice::get_Dim() const
{
	return this->dimension;
}

int generalModel::Lattice::get_nn(int lat_site, int nei_num) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	if(nei_num > this->nearest_neighbors[lat_site].size()) throw "Not having this nn \n";
	return this->nearest_neighbors[lat_site][nei_num];
}

int generalModel::Lattice::get_nnn(int lat_site, int nei_num) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	if(nei_num > this->next_nearest_neighbors[lat_site].size()) throw "Not having this nnn \n";
	return this->next_nearest_neighbors[lat_site][nei_num];
}

int generalModel::Lattice::get_nn_number(int lat_site) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	return this->nearest_neighbors[lat_site].size();
}

int generalModel::Lattice::get_nnn_number(int lat_site) const
{
	if(lat_site >= this->Ns) throw "To high lattice site requested \n";
	return this->next_nearest_neighbors[lat_site].size();
}
