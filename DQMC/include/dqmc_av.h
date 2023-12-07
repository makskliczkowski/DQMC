/***************************************
* Defines the general DQMC class.
* It is a base for further more
* complicated Hamiltonians
* development in a finite temperature.
* APRIL 2023. UNDER CONSTANT DEVELOPMENT
* MAKSYMILIAN KLICZKOWSKI, WUST, POLAND
***************************************/
#pragma once

#ifndef LATTICE_H
	#include "../source/src/lattices.h"
#endif

#ifndef BINARY_H
	#include "../source/src/binary.h"
#endif

#include <mutex>
#include <complex>
#include <type_traits>
#include <shared_mutex>

#ifndef DQMC_AV_H
#define DQMC_AV_H

#define DQMC_SAVE_H5
#define DQMC_USE_HIRSH
#define DQMC_CAL_TIMES
#ifdef DQMC_CAL_TIMES
#	define DQMC_CAL_TIMES_ALL
#endif

// ##########################################################################################

#define SINGLE_PARTICLE_INPUT	int _sign, uint _i,			 const GREEN_TYPE& _g
#define TWO_PARTICLES_INPUT		int _sign, uint _i, uint _j, const GREEN_TYPE& _g
#define INVOKE_SINGLE_PARTICLE_CAL(CLS,X,BAND)		CLS->calOneSite(_sign, _currI, this->G_, #X, CLS->av_##X##_##BAND, CLS->sd_##X##_##BAND);
#define INVOKE_TWO_PARTICLE_CAL(CLS,X,BAND,x,y,z)	CLS->calTwoSite(_sign, _currI, _currJ, this->G_, #X, CLS->avC_##X##_##BAND, x, y, z);

#define SINGLE_PARTICLE_NORM(nam, norm, band)		this->av_##nam##_##band /=		norm;						\
													this->sd_##nam##_##band =		variance(this->sd_##nam##_##band, this->av_##nam##_##band, norm)
#define SINGLE_PARTICLE_PARAM(name, type, band)		type av_##name##_##band =		0;							\
													type sd_##name##_##band =		0


#define TWO_PARTICLE_NORM(nam, x, y, z, nor, band)	this->avC_##nam##_##band[x][y][z] /=	nor;						
#define TWO_PARTICLE_PARAM(name, type, band)		VectorGreenCube<type> avC_##name##_##band;					\
													VectorGreenCube<type> sdC_##name##_##band					

// ##########################################################################################

/*
* @brief Saves the Green's function for at most 3 dimension in a more compact form
*/
template <typename _T>
class VectorGreenCube
{
private:
	uint x_num_ = 1;
	uint y_num_ = 1;
	uint z_num_ = 1;

	v_3d<_T> green_;

public:
	~VectorGreenCube()
	{
		green_.clear();
	}
	VectorGreenCube()
		: x_num_(1), y_num_(1), z_num_(1)
	{
		this->reset();
	}
	VectorGreenCube(uint _x_num, uint _y_num = 1, uint _z_num = 1)
		: x_num_(_x_num), y_num_(_y_num), z_num_(_z_num)
	{
		// initialize the Green's function
		this->reset();
	}
	VectorGreenCube(const VectorGreenCube& other)
		: x_num_(other.x_num_), y_num_(other.y_num_), z_num_(other.z_num_)
	{
		// initialize the Green's function
		this->green_ = other.green_;
	}
	VectorGreenCube(VectorGreenCube&& other)
		: x_num_(other.x_num_), y_num_(other.y_num_), z_num_(other.z_num_)
	{
		// initialize the Green's function
		this->green_ = std::exchange(other.green_, v_3d<_T>{});
	}


	// ############ S L I C E S ############
	
	/*
	* @brief Slices the 3D vector to arma matrix by adding the last dimension to the right 
	* - thus creating z_num such matrices
	* The 3D tensor at the beginning is (2 * Lx - 1, 2 * Ly - 1, 2 * Lz - 1) dimensional and then
	* is transformed into a matrix of size (2 * Lx - 1, (2 * Ly - 1) repeated (2 * Lz - 1) times on the z axis
	* @returns matrix slice
	*/
	arma::Mat<_T> slice()
	{
		arma::Mat<_T> _slice(this->x_num_, this->y_num_ * this->z_num_, arma::fill::zeros);
		for (auto k = 0; k < this->z_num_; ++k)
			for (auto i = 0; i < this->x_num_; ++i)
				for (auto j = 0; j < this->y_num_; ++j)
					_slice(i, j + k * this->y_num_) = this->green_[i][j][k];
		return _slice;
	}

	/*
	* @brief Slices the 3D vector to arma matrix
	* @param _z - the cut over last dimension
	* @returns matrix slice
	*/
	arma::Mat<_T> slice(uint _z)
	{
		arma::Mat<_T> _slice(this->x_num_, this->y_num_, arma::fill::zeros);
		for (auto i = 0; i < this->x_num_; ++i)
			for (auto j = 0; j < this->y_num_; ++j)
				_slice(i, j) = this->green_[i][j][_z];
		return _slice;
	}

	/*
	* @brief Slices the 3D vector to arma column
	* @param _y - the cut over y
	* @param _z - the cut over z
	* @returns column slice
	*/
	arma::Col<_T> slice(uint _y, uint _z)
	{
		arma::Col<_T> _slice(this->x_num_, arma::fill::zeros);
		for (auto i = 0; i < this->x_num_; ++i)
				_slice(i) = this->green_[i][_y][_z];
		return _slice;
	}

	// ############ R E S E T S ############
	/*
	* @brief Resets the Green's function to zero state
	*/
	void reset()
	{
		this->green_ = v_3d<double>(x_num_, v_2d<double>(y_num_, v_1d<double>(z_num_, 0)));
	}
	void reset(uint _x_num, uint _y_num, uint _z_num)
	{
		this->green_ = v_3d<double>(_x_num, v_2d<double>(_y_num, v_1d<double>(_z_num, 0)));
	}
	
	// ######### O P E R A T O R S #########

	VectorGreenCube& operator = (const VectorGreenCube& other)
	{
		if (this == &other)
			return *this;

		this->x_num_ = other.x_num_;
		this->y_num_ = other.y_num_;
		this->z_num_ = other.z_num_;
		this->green_ = other.green_;
		return *this;
	}
	VectorGreenCube& operator = (VectorGreenCube&& other)
	{
		if (this == &other)
			return *this;

		this->x_num_ = other.x_num_;
		this->y_num_ = other.y_num_;
		this->z_num_ = other.z_num_;
		this->green_ = std::exchange(other.green_, v_3d<_T>{});
		return *this;
	}
	// algebraic
	VectorGreenCube& operator + (const VectorGreenCube& other)
	{
		if (this->x_num_ != other.x_num_ ||
			this->y_num_ != other.y_num_ ||
			this->z_num_ != other.z_num_)
			throw std::runtime_error("Cannot add such Green's... Wrong dimensionality");

		for (auto i = 0; i < this->x_num_; ++i)
			for (auto j = 0; j < this->y_num_; ++j)
				for (auto k = 0; k < this->z_num_; ++k)
					this->green_[i][j][k] += other.green_[i][j][k];
		return *this;
	}
	template<typename _T2>
	VectorGreenCube& operator / (const _T2& other)
	{
		for (auto i = 0; i < this->x_num_; ++i)
			for (auto j = 0; j < this->y_num_; ++j)
				for (auto k = 0; k < this->z_num_; ++k)
					this->green_[i][j][k] /= other;
		return *this;
	}
	template<typename _T2>
	VectorGreenCube& operator * (const _T2& other)
	{
		for (auto i = 0; i < this->x_num_; ++i)
			for (auto j = 0; j < this->y_num_; ++j)
				for (auto k = 0; k < this->z_num_; ++k)
					this->green_[i][j][k] *= other;
		return *this;
	}
	// obtaining
	_T& operator ()(size_t _x = 0, size_t _y = 0, size_t _z = 0)
	{
		if (_x >= x_num_)
			throw std::runtime_error("Out of bounds in X");
		if (_y >= y_num_)
			throw std::runtime_error("Out of bounds in Y");
		if (_z >= z_num_)
			throw std::runtime_error("Out of bounds in Z");
		return this->green_[_x][_y][_z];
	}
	const _T& operator ()(size_t _x = 0, size_t _y = 0, size_t _z = 0) const
	{
		if (_x >= x_num_)
			throw std::runtime_error("Out of bounds in X");
		if (_y >= y_num_)
			throw std::runtime_error("Out of bounds in Y");
		if (_z >= z_num_)
			throw std::runtime_error("Out of bounds in Z");
		return this->green_[_x][_y][_z];
	}
	v_2d<_T>& operator [](size_t _x)
	{
		if (_x >= x_num_)
			throw std::runtime_error("Out of bounds in X");
		return this->green_[_x];
	}
	const v_2d<_T>& operator [](size_t _x) const
	{
		if (_x >= x_num_)
			throw std::runtime_error("Out of bounds in X");
		return this->green_[_x];
	}

};

// ##########################################################################################

/*
* @brief Stores all the averages for the DQMC simulation
*/
template <size_t _spinNum, typename _retT>
class DQMCavs 
{
protected:
	size_t bucketNum_									= 100;
	size_t Nbands_										= 1;
	std::shared_ptr<Lattice> lat_;
	const v_1d<double>* t_nn_;							// nn hopping integrals
public:
	DQMCavs(std::shared_ptr<Lattice> _lat, int _M, uint _Nbands, const v_1d<double>* _t_nn = nullptr)
		: Nbands_(_Nbands), lat_(_lat), t_nn_(_t_nn)
	{

		LOGINFO(LOG_TYPES::INFO, "Building DQMC base averages class", 30, '%', 2);
		// get the number of elements and reset values
		auto [x_num, y_num, z_num] = _lat->getNumElems();
		this->reset();
#ifdef DQMC_CAL_TIMES
		this->normM_ = arma::vec(_M, arma::fill::zeros);
#endif

#ifdef DQMC_CAL_TIMES
		for (int _SPIN_ = 0; _SPIN_ < this->av_GTimeDiff_.size(); _SPIN_++)
		{
			// create awfull three dimensional vectors
			this->av_GTimeDiff_[_SPIN_] = v_1d<VectorGreenCube<double>>(_M, VectorGreenCube<double>(x_num * _Nbands, y_num * _Nbands, z_num * _Nbands));
			this->sd_GTimeDiff_[_SPIN_] = v_1d<VectorGreenCube<double>>(_M, VectorGreenCube<double>(x_num * _Nbands, y_num * _Nbands, z_num * _Nbands));
			for (int tau1 = 0; tau1 < _M; tau1++) {
#ifdef DQMC_CAL_TIMES_ALL
				for (int tau2 = 0; tau2 < _M; tau2++) {
#else
				for (int tau2 = 0; tau2 <= tau1; tau2++) {
#endif
					auto tim = (tau1 - tau2);
					if (tim < 0)
						tim += _M;
					this->normM_(tim)	+= 1.0;
				}
			}
		}
#endif
	}

public:
	using GREEN_TYPE			=	std::array<arma::Mat<double>, _spinNum>;
	typedef std::function<_retT(SINGLE_PARTICLE_INPUT)>								SINGLE_PART_FUN;		
	typedef std::function<_retT(TWO_PARTICLES_INPUT)>								TWO_PARTS_FUN;	

	SINGLE_PARTICLE_PARAM(sign, double, );

#ifdef DQMC_CAL_TIMES
	std::array<v_1d<VectorGreenCube<double>>, _spinNum>	av_GTimeDiff_;
	std::array<v_1d<VectorGreenCube<double>>, _spinNum>	sd_GTimeDiff_;
#endif // DQMC_CAL_TIMES

	int norm_					=	1;
	int normSign_				=	1;
	arma::vec normM_;
	// ################################## F U N C T I O N S #################################
	
	virtual void reset();
	virtual void reset(size_t buckets);
	virtual void resetG();
	virtual void setBucketNum(size_t buckets);
	virtual void normalize(int _avNum, int _normalization, double _avSign);
	virtual void normalizeG();
	void calOneSite(SINGLE_PARTICLE_INPUT, const std::string& choice, _retT& av, _retT& stdev);
	void calTwoSite(TWO_PARTICLES_INPUT, const std::string& choice, VectorGreenCube<_retT>& av, int x, int y, int z);
	
	// ################################ C A L C U L A T O R S ###############################
	// --- SINGLE ---
#define DQMC_AV_FUN1(x) virtual _retT cal_##x (SINGLE_PARTICLE_INPUT)	{ return 0;	};		\
						SINGLE_PARTICLE_PARAM( x, double, 0);								\
						SINGLE_PARTICLE_PARAM( x, double, 1);								\
						SINGLE_PARTICLE_PARAM( x, double, 2);

#include "averageCalculatorSingle.include"
#undef DQMC_AV_FUN1
	// ----- TWO ----
#define DQMC_AV_FUN2(x) virtual _retT cal_##x##_C(TWO_PARTICLES_INPUT)	{ return 0;	};		\
						TWO_PARTICLE_PARAM( x, double, 0);									\
						TWO_PARTICLE_PARAM( x, double, 1);									\
						TWO_PARTICLE_PARAM( x, double, 2);
#include "averageCalculatorCorrelations.include"
#undef DQMC_AV_FUN2

	// #################################### M A P P I N G S #################################
	// --- SINGLE ---
#define DQMC_AV_FUN1(x)	{ #x ,	[&](SINGLE_PARTICLE_INPUT)	{ return cal_##x (_sign, _i, _g); }},
	std::map<std::string, SINGLE_PART_FUN>	calFun = {
#include "averageCalculatorSingle.include"
	};
#undef DQMC_AV_FUN1
	// ----- TWO ----
#define DQMC_AV_FUN2(x) {#x ,	[&](TWO_PARTICLES_INPUT)	{ return cal_##x##_C(_sign, _i, _j, _g); }},
	std::map<std::string, TWO_PARTS_FUN>	calFun2 = {
#include "averageCalculatorCorrelations.include"
	};
#undef DQMC_AV_FUN2
};

// ##########################################################################################

/*
* @brief Setups the bucket number for the averages
* @param buckets number of the buckets used
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::setBucketNum(size_t buckets)
{
	this->bucketNum_ = buckets;
}

// ##########################################################################################

/*
* @brief Normalise all the averages taken during simulation.
* @param _avNum number of averages taken
* @param _normalization normalization comming from the lattice properties
* @param _avSign average sign from the simulation
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::normalize(int _avNum, int _normalization, double _avSign)
{
	this->norm_			=		_avNum * _normalization;
	this->av_sign_		=		_avSign;
	this->normSign_		=		this->norm_ * this->av_sign_;
	
	// 0
	{
		// OCCUPATION
		SINGLE_PARTICLE_NORM(Occupation, this->normSign_, 0);

		// Mz2
		SINGLE_PARTICLE_NORM(Mz2, this->normSign_, 0);

		// My2
		SINGLE_PARTICLE_NORM(My2, this->normSign_, 0);

		// Mx2
		SINGLE_PARTICLE_NORM(Mx2, this->normSign_, 0);

		// Kinetic energy
		SINGLE_PARTICLE_NORM(Ek, this->normSign_, 0);
	}
	// 1
	{
		// OCCUPATION
		SINGLE_PARTICLE_NORM(Occupation, this->normSign_, 1);

		// Mz2
		SINGLE_PARTICLE_NORM(Mz2, this->normSign_, 1);

		// My2
		SINGLE_PARTICLE_NORM(My2, this->normSign_, 1);

		// Mx2
		SINGLE_PARTICLE_NORM(Mx2, this->normSign_, 1);

		// Kinetic energy
		SINGLE_PARTICLE_NORM(Ek, this->normSign_, 1);
	}
	// 2
	{
		// OCCUPATION
		SINGLE_PARTICLE_NORM(Occupation, this->normSign_, 2);

		// Mz2
		SINGLE_PARTICLE_NORM(Mz2, this->normSign_, 2);

		// My2
		SINGLE_PARTICLE_NORM(My2, this->normSign_, 2);

		// Mx2
		SINGLE_PARTICLE_NORM(Mx2, this->normSign_, 2);

		// Kinetic energy
		SINGLE_PARTICLE_NORM(Ek, this->normSign_, 2);
	}

	// ----------------- C o r r e l a t i o n s -----------------
	auto [x_num, y_num, z_num] = this->lat_->getNumElems();
	for (int x = 0; x < x_num; x++)
	{
		for (int y = 0; y < y_num; y++)
		{
			for (int z = 0; z < z_num; z++)
			{
				// 0
				{
					TWO_PARTICLE_NORM(Mz2, x, y, z, this->normSign_, 0);
					//TWO_PARTICLE_NORM(My2, x, y, z, this->normSign_);
					//TWO_PARTICLE_NORM(Mx2, x, y, z, this->normSign_);
					this->avC_Occupation_0[x][y][z] = this->lat_->get_Ns() * this->avC_Occupation_0[x][y][z] / this->normSign_;
				}
				// 1
				{
					TWO_PARTICLE_NORM(Mz2, x, y, z, this->normSign_, 1);
					//TWO_PARTICLE_NORM(My2, x, y, z, this->normSign_);
					//TWO_PARTICLE_NORM(Mx2, x, y, z, this->normSign_);
					this->avC_Occupation_1[x][y][z] = this->lat_->get_Ns() * this->avC_Occupation_1[x][y][z] / this->normSign_;
				}
				// 2
				{
					TWO_PARTICLE_NORM(Mz2, x, y, z, this->normSign_, 2);
					//TWO_PARTICLE_NORM(My2, x, y, z, this->normSign_);
					//TWO_PARTICLE_NORM(Mx2, x, y, z, this->normSign_);
					this->avC_Occupation_2[x][y][z] = this->lat_->get_Ns() * this->avC_Occupation_2[x][y][z] / this->normSign_;
				}
			}
		}
	}
}

// ##########################################################################################

/*
* @brief Normalizes the Green's given the model parameters and the lattice
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::normalizeG()
{
	// time number
	const auto _M		=	this->av_GTimeDiff_[0].size();
	auto [xx, yy, zz]	=	this->lat_->getNumElems();
	
	// go through spins in the system
	for (int _SPIN_ = 0; _SPIN_ < _spinNum; _SPIN_++) 
	{
		// number of imaginary times
		for (int tau = 0; tau < _M; tau++) 
		{
			for (int _band = 0; _band < this->Nbands_; _band++)
			{
				// find time normalization
				auto norm = -this->bucketNum_ * this->normM_(tau);
				for (int x = 0; x < xx; x++) 
				{
					for (int y = 0; y < yy; y++)
					{
						for (int z = 0; z < zz; z++)
						{
							// get norm comming from the lattice sites
							const auto norm2 = norm * lat_->getNorm(x, y, z);
							// divide by the norm
							this->av_GTimeDiff_[_SPIN_][tau](x + _band * xx, y + _band * yy, z + _band * zz) /=	norm2;
							// save the variance 
							this->sd_GTimeDiff_[_SPIN_][tau](x + _band * xx, y + _band * yy, z + _band * zz) = variance(this->sd_GTimeDiff_[_SPIN_][tau](x + _band * xx, y + _band * yy, z + _band * zz), 
																														this->av_GTimeDiff_[_SPIN_][tau](x + _band * xx, y + _band * yy, z + _band * zz),
																														norm2);
						}
					}
				}
			}
		}
	}
}

// ##########################################################################################

template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::calOneSite(SINGLE_PARTICLE_INPUT, const std::string& choice, _retT& av, _retT& stdev)
{
	auto _val		=		this->calFun[choice](_sign, _i, _g);
	av				+=		_val;
	stdev			+=		_val * _val;
};

// ##########################################################################################

template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::calTwoSite(TWO_PARTICLES_INPUT, const std::string& choice, VectorGreenCube<_retT>& av, int x, int y, int z)
{
	auto _val		=		this->calFun2[choice](_sign, _i, _j, _g);
	av(x, y, z)		+=		_val;
};

// ##########################################################################################

/*
* @brief Resets all averages calculated previously
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::reset() 
{
		auto [x_num, y_num, z_num]		=		this->lat_->getNumElems();

		// 0
		{
			this->av_Ek_0					=		0;
			this->sd_Ek_0					=		0;
		
			this->av_Occupation_0			=		0;
			this->sd_Occupation_0			=		0;
		
			this->av_Mz2_0					=		0;
			this->sd_Mz2_0					=		0;

			this->av_Mx2_0					=		0;
			this->sd_Mx2_0					=		0;

			this->av_My2_0					=		0;
			this->sd_My2_0					=		0;
			// correlations
			this->avC_Mz2_0					=		VectorGreenCube<_retT>(x_num, y_num, z_num);
			this->avC_Occupation_0			=		VectorGreenCube<_retT>(x_num, y_num, z_num);
		}
		// 1
		{
			this->av_Ek_1					=		0;
			this->sd_Ek_1					=		0;
		
			this->av_Occupation_1			=		0;
			this->sd_Occupation_1			=		0;
		
			this->av_Mz2_1					=		0;
			this->sd_Mz2_1					=		0;

			this->av_Mx2_1					=		0;
			this->sd_Mx2_1					=		0;

			this->av_My2_1					=		0;
			this->sd_My2_1					=		0;
			// correlations
			this->avC_Mz2_1					=		VectorGreenCube<_retT>(x_num, y_num, z_num);
			this->avC_Occupation_1			=		VectorGreenCube<_retT>(x_num, y_num, z_num);
		}
		// 2
		{
			this->av_Ek_2					=		0;
			this->sd_Ek_2					=		0;
		
			this->av_Occupation_2			=		0;
			this->sd_Occupation_2			=		0;
		
			this->av_Mz2_2					=		0;
			this->sd_Mz2_2					=		0;

			this->av_Mx2_2					=		0;
			this->sd_Mx2_2					=		0;

			this->av_My2_2					=		0;
			this->sd_My2_2					=		0;
			// correlations
			this->avC_Mz2_2					=		VectorGreenCube<_retT>(x_num, y_num, z_num);
			this->avC_Occupation_2			=		VectorGreenCube<_retT>(x_num, y_num, z_num);
		}
		
#ifdef DQMC_CAL_TIMES
		this->resetG();
#endif
}

/*
* @brief Resets all averages calculated previously and additionally sets the bucket number
* @param buckets number of the buckets used
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::reset(size_t buckets)
{
	this->setBucketNum(buckets);
	this->reset();
}

// ##########################################################################################

/*
* @brief Resets the Greens function for all spin channels
*/
template<size_t _spinNum, typename _retT>
inline void DQMCavs<_spinNum, _retT>::resetG()
{
	for (int _SPIN_ = 0; _SPIN_ < this->av_GTimeDiff_.size(); _SPIN_++)
	{
		for (int tau = 0; tau < this->av_GTimeDiff_[_SPIN_].size(); tau++) 
		{
			this->av_GTimeDiff_[_SPIN_][tau].reset();
			this->sd_GTimeDiff_[_SPIN_][tau].reset();
		}
	}
}
#endif // !DQMC_AV_H