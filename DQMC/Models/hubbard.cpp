#include "../include/Models/hubbard.h"

#include <numeric>
#include <utility>

// ################################################# C O N S T R U C T O R S ######################################################

Hubbard::Hubbard(double _T, 
				 std::shared_ptr<Lattice> _lat, 
				 uint _M, 
				 uint _M0,
				 v_1d<double> _t, 
				 v_1d<double> _U, 
				 v_1d<double> _mu,
				 double _dtau, 
				 uint _bands,
				 v_1d<double> _tt)
	: DQMC2(_T, _lat, _M, _M0, _bands), t_(_t), U_(_U), mu_(_mu), tt_(_tt), dtau_(_dtau)
{
	this->setInfo();
	if (this->M_ % this->M0_ != 0)		
		throw std::runtime_error("Cannot have M0 times that do not divide M.");

	// initialize band parameters (single value for mu indicates that each band has the same filling)
	this->transformSize_	=			this->Ns_ * _bands;		
	if (this->t_.size() != this->transformSize_ && 
		this->tt_.size()!= this->t_.size()		&&
		this->U_.size() != this->t_.size()		&&
		this->mu_.size()!= this->t_.size())
		throw std::runtime_error("Cannot initialize such Hamiltonian. Check lengths of the parameters...");

	// initialize random numbers and averages
	this->ran_				=			randomGen(DQMC_RANDOM_SEED ? DQMC_RANDOM_SEED : std::random_device{}());
	this->avs_				=			std::make_shared<DQMCavs2>(_lat, _M, NBands_, &this->t_);

	// repulsiveness and lambda couplings
	for (auto i = 0; i < this->transformSize_; i++)
	{
		// is > 0?
		this->isRepulsive_.push_back((this->U_[i] > 0));

		// push lambda values - lambda couples to the auxiliary spins
		this->lambda_.push_back(std::acosh(std::exp((std::abs(this->U_[i]) * this->dtau_) / 2.0)));

		// precalculate gamma exponents
		if (this->isRepulsive_[i])
		{
			double expP			=			std::expm1(-2.0 * this->lambda_[i]);				// spin * hsfield =  1
			double expM			=			std::expm1(2.0 * this->lambda_[i]);					// spin * hsfield = -1
			this->gammaExp_.push_back({{ expP, expM }, { expM, expP }});						// [hsfield = 1, hsfield = -1]
		}
		else
		{
			double expP			=			std::expm1((-2.0 + 1) * this->lambda_[i]);			// spin * hsfield =  1
			double expM			=			std::expm1(-(-2.0 + 1) * this->lambda_[i]);			// spin * hsfield = -1
			this->gammaExp_.push_back({ { expP, expP }, { expM, expM } });						// [hsfield = 1, hsfield = -1]
		}
	}
	// parameters for the simulation
	this->currentGamma_		=			&this->gammaExp_[0][0];
	this->fromScratchNum_	=			this->M0_;
		
	// initialize variables
	this->init();
	this->setHS(HS_CONF_TYPES::HIGH_T);
	this->calQuadratic();
	this->calInteracts();
	this->calPropagatB();
	for (uint i = 0; i < this->p_; i++)
		this->calPropagatBC(i);
	this->posNum_			=			0;
	this->negNum_			=			0;
}

// ################################################# I N I T I A L I Z E R S ######################################################

/*
* @brief initializes the memory for all of the matrices used later
*/
void Hubbard::init()
{
	// set the write lock
	// WriteLock lock(this->Mutex);

	// hopping exponent
	this->TExp_.zeros(this->transformSize_, this->transformSize_);

	// HS transformation fields
	this->HSFields_.ones(this->M_, this->transformSize_);

	// all the spin matrices
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
	{
		// Green's matrix
		this->G_		[_SPIN_].zeros(this->transformSize_, this->transformSize_);
		// interaction
		this->IExp_		[_SPIN_].zeros(this->transformSize_, this->M_);
		// propagators
		this->B_		[_SPIN_]	=	v_1d<arma::mat>(this->M_, ZEROM(this->transformSize_));
		this->iB_		[_SPIN_]	=	v_1d<arma::mat>(this->M_, ZEROM(this->transformSize_));
		this->Bcond_	[_SPIN_]	=	v_1d<arma::mat>(this->p_, ZEROM(this->transformSize_));
		// initialize UDT decomposition
		this->udt_		[_SPIN_]	=	std::make_unique<algebra::UDT_QR<double>>(this->G_[_SPIN_]);

#ifdef DQMC_CAL_TIMES
#	ifdef DQMC_CAL_TIMES_ALL
		this->Gtime_	[_SPIN_].zeros(this->M_ * this->transformSize_, this->M_ * this->transformSize_);
#	else
		this->Gtime_	[_SPIN_].zeros(this->p_ * this->transformSize_, this->p_ * this->transformSize_);
#	endif
#endif
	}
	LOGINFO("Finished initializing Hubbard-DQMC properties.", LOG_TYPES::INFO, 4);
}

// ###################################################### H E L P E R S ###########################################################

/*
* @brief Compare decomposition created Green's functions with directly calculated
* @param _tau at which time shall I compare them?
* @param _toll tollerance for them being equal
* @param _print shall I print both explicitly?
*/
void Hubbard::compareGreen(uint _tau, double _toll, bool _print)
{
	LOGINFO("Comparing the exact and numerical Green's functions at $\tau$=" + STR(_tau), LOG_TYPES::TRACE, 2);
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
	{
		arma::mat _tmpG		=	arma::eye(this->transformSize_, this->transformSize_);

		// calculate the Green's function directly
		for (int _t = 0; _t < this->M_; _t++)
		{
			_tmpG			=	this->B_[_SPIN_][_tau] * _tmpG;
			_tau			=	(_tau + 1) % this->M_;
		}
		_tmpG				=	(EYE(this->transformSize_) + _tmpG).i();
		if (_print) {
			LOGINFO("Calculating (exact) Green's function for spin " + std::string(getSTR_SPINNUM(static_cast<SPINNUM>(_SPIN_))), LOG_TYPES::TRACE, 3);
			LOGINFO(_tmpG, LOG_TYPES::TRACE, 3);
		}
		bool isEqual		=	arma::approx_equal(this->G_[_SPIN_], _tmpG, "absdiff", _toll);
		LOGINFO(isEqual ? "Is the same!!!" : "Is different!!!", LOG_TYPES::TRACE, 4);
		if (_print) {
			LOGINFO(this->G_[_SPIN_], LOG_TYPES::TRACE, 3);
		}
	}
}

void Hubbard::compareGreen()
{
	this->tmpG_[_UP_] = this->G_[_UP_];
	this->tmpG_[_DN_] = this->G_[_DN_];
	this->calGreensFun(this->tau_);
	stout << VEQ(this->tau_) << "\n";
	auto UP = arma::approx_equal(this->G_[_UP_], this->tmpG_[_UP_], "absdiff", 1e-7);
	auto DN = arma::approx_equal(this->G_[_DN_], this->tmpG_[_DN_], "absdiff", 1e-7);
	stout << "_UP_: " << UP << "\n";
	stout << "_DN_: " << DN << "\n";
	if (!UP)
	{
		this->tmpG_[_UP_].print("\n");
		this->G_[_UP_].print("\n");
	}
	if (!DN)
	{
		this->tmpG_[_DN_].print("\n");
		this->G_[_DN_].print("\n");
	}
	stout << "--- \n";
}

// ####################################################### G A M M A S ############################################################

/*
* @brief Function to calculate the change in the interaction exponent
* @param _site site on which the change has been made
* @returns A pair for gammas for two spin channels, 0 is spin up, 1 is spin down
*/
void Hubbard::calGamma(uint _site)
{
	if (this->HSFields_(this->tau_, _site) > 0)
		this->currentGamma_ = &this->gammaExp_[_site][0];
	else
		this->currentGamma_ = &this->gammaExp_[_site][1];
}

// ################################################## C A L C U L A T O R S #######################################################

/*
* @brief Allows to calculate the change in the interaction exponent
*/
Hubbard::spinTuple_ Hubbard::calDelta()
{
	spinTuple_ _out = { 0, 0 };
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; ++_SPIN_)
		_out[_SPIN_] = 1.0 + (*this->currentGamma_)[_SPIN_];
	return _out;
}

/*
* @brief Return probabilities of spin flip for both spin channels
* @param _site flipping candidate site
* @param gii gammas for both spin channels
* @returns tuple for probabilities on both spin channels, 
* @warning remember, 0 is spin up, 1 is spin down
*/
void Hubbard::calProba(uint _site)
{
	this->currentProba_[_UP_] = 1.0 + ((*currentGamma_)[_UP_] * (1.0 - this->G_[SPINNUM::_UP_](_site, _site)));
	this->currentProba_[_DN_] = 1.0 + ((*currentGamma_)[_DN_] * (1.0 - this->G_[SPINNUM::_DN_](_site, _site)));
}

// ################################################### H A M I L T O N I A N ######################################################

/*
* @brief Function to calculate the hopping matrix exponential.
* c_iq^+ * c_jq, where {i,j} are lattice sites and q numbers the band
*/
void Hubbard::calQuadratic()
{
	// cacluate the hopping matrix
	this->TExp_.zeros(this->transformSize_, this->transformSize_);

	// ----- IN BAND HOPPING -----
	// go through the Hubbard bands
	for (int _band = 0; _band < this->NBands_; _band++)
	{
		// go through the lattice sites
		for (int _site = 0; _site < this->Ns_; _site++)
		{
			const auto bandSite			=	this->Ns_ * _band + _site;
			const auto neiSize			=	this->lat_->get_nn(_site);
			// go through the nearest neighbors
			for (int neiNum = 0; neiNum < neiSize; neiNum++) {
				const auto nei				=	this->Ns_ * _band + this->lat_->get_nn(_site, neiNum);	// get given nn
				this->TExp_(bandSite, nei)	=	this->dtau_ * this->t_[bandSite];						// assign non-diagonal elements
			}

			// check the next nearest neighbors
			if (this->tt_[bandSite] != 0.0)
			{
				const auto neiSizeNN		=	this->lat_->get_nnn(_site);
				// go through the nearest neighbors
				for (int neiNum = 0; neiNum < neiSizeNN; neiNum++) {
					const auto nei				=	this->Ns_ * _band + this->lat_->get_nnn(_site, neiNum);		// get given nn
					this->TExp_(bandSite, nei)	=	this->dtau_ * this->tt_[bandSite];							// assign non-diagonal elements
				}
			}
		}
	}
	// matrix exponential
	this->TExp_								=		arma::expmat(this->TExp_);
}

/*
* @brief Function to calculate the interaction exponential at all Trotter times, each column represents the given Trotter time.
*/
auto Hubbard::calInteracts() -> void
{
	// Trotter times
	for (int l = 0; l < this->M_; l++) 
	{
		// bands
		for (auto _band = 0; _band < this->NBands_; _band++)
		{
			for (int _site = 0; _site < this->Ns_; _site++)
			{
				auto _innerSite						=		_band * this->Ns_ + _site;
				auto _eta							=		this->isRepulsive_[_innerSite] ? 1.0 : -1.0;

				this->IExp_[_UP_](_innerSite, l)	=		this->dtau_ * this->mu_[_innerSite] + (1.0 + (_eta - 1.0) / 4.0) * this->HSFields_(l, _innerSite) * this->lambda_[_innerSite];
				this->IExp_[_DN_](_innerSite, l)	=		this->dtau_ * this->mu_[_innerSite] + (-_eta + (_eta - 1.0) / 4.0) * this->HSFields_(l, _innerSite) * this->lambda_[_innerSite];
			}
		}
		this->IExp_[_UP_].col(l)		=		arma::exp(this->IExp_[_UP_].col(l));
		this->IExp_[_DN_].col(l)		=		arma::exp(this->IExp_[_DN_].col(l));
	}
}

/*
* @brief Function to calculate all B propagators for a Hubbard model. Those are used for the Gibbs weights.
*/
auto Hubbard::calPropagatB() -> void
{
	for(int _spin = 0; _spin < this->spinNumber_; ++_spin)
	{
		for (int l = 0; l < this->M_; ++l) {
			// Trotter times
			this->B_[_spin][l]		=		this->TExp_ * DIAG(this->IExp_[_spin].col(l));
			this->iB_[_spin][l]		=		arma::inv(this->B_[_spin][l]);
		}
	}
}

/*
* @brief Function to calculate all B propagators for a Hubbard model. Those are used for the Gibbs weights.
* @param _tau Specific Trotter imaginary time
*/
auto Hubbard::calPropagatB(uint _tau) -> void
{
	for (auto _spin = 0; _spin < this->spinNumber_; _spin++) 
	{
		this->B_[_spin][_tau]		=		this->TExp_ * DIAG(this->IExp_[_spin].col(_tau));
		this->iB_[_spin][_tau]		=		arma::inv(this->B_[_spin][_tau]);
	}
}

/*
* @brief Precalculate the multiplications of B matrices according to M0 stable ones
* @param _sec the sector to calculate the stable multiplication
*/
void Hubbard::calPropagatBC(uint _sec)
{
	const int _time						=		_sec * this->M0_;
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; ++_SPIN_) {
		this->Bcond_[_SPIN_][_sec]		=		this->B_[_SPIN_][_time];
		for (int i = 1; i < this->M0_; ++i)
		{
			const int _timeIn			=		_time + i;
			this->Bcond_[_SPIN_][_sec]	=		this->B_[_SPIN_][_timeIn] * this->Bcond_[_SPIN_][_sec];
		}
	}
	
}

/*
* @brief Directly calculate the Green's function without the decomposition.
* @param _tau Green's function time
*/
void Hubbard::calGreensFun(uint _tau)
{
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; ++_SPIN_)
	{
		uint tau				=		_tau;
		this->G_[_SPIN_]		=		this->B_[_SPIN_][tau];
		for (int i = 1; i < this->M_; ++i) 
		{
			++tau;
			if (tau == this->M_)
				tau = 0;
			this->G_[_SPIN_]	=		this->B_[_SPIN_][tau] * this->G_[_SPIN_];
		}
		this->G_[_SPIN_]		=		arma::inv(arma::eye(this->G_[_SPIN_].n_rows, this->G_[_SPIN_].n_cols) + this->G_[_SPIN_]);
	}
}

/*
* Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023 with premultiplied B matrices composition.
* For more look into :
* @copydetails "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems".
* In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices.
* @param _sec starting time sector
*/
void Hubbard::calGreensFunC(uint _sec)
{
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; ++_SPIN_) {

		uint sector		=		_sec;

		// decompose the matrices
		this->udt_[_SPIN_]->decompose(this->Bcond_[_SPIN_][sector]);

		// go through each sector
		for (int i = 1; i < this->p_; i++) {
			sector++;
			if (sector == this->p_)
				sector = 0;
			this->udt_[_SPIN_]->factMult(this->Bcond_[_SPIN_][sector]);
		}

		// save the Green's - using LOH already
		this->G_[_SPIN_] = this->udt_[_SPIN_]->inv1P();
	}
}

// ################################################# U N E Q U A L   T I M E S ####################################################
#ifdef DQMC_CAL_TIMES

/*
* @brief Use the space-time formulation for Green's function calculation.
* @trace Using stable multiplication number.
* @warning Inversion can be unstable.
* @cite Stable Monte Carlo algorit&sn for fermion lattice systems at low temperatures
*/
void Hubbard::calGreensFunTHirshC()
{
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++) {
		this->Gtime_[_SPIN_].eye();
		algebra::setSubMFromM(this->Gtime_[_SPIN_], this->Bcond_[_SPIN_][this->p_ - 1], 0, (this->M_ - 1) * transformSize_, transformSize_, transformSize_, true, false);
		// other sectors
		for (int _sec = 0; _sec < this->p_ - 1; _sec++) {
			const auto row	=	(_sec + 1	) * transformSize_;
			const auto col	=	(_sec		) * transformSize_;
			algebra::setSubMFromM(this->Gtime_[_SPIN_], this->Bcond_[_SPIN_][_sec], row, col, transformSize_, transformSize_, true, true);
		}
		arma::inv(this->Gtime_[_SPIN_], this->Gtime_[_SPIN_]);
	}
}

/*
* @brief Use the space-time formulation for Green's function calculation.
* @warning Inversion can be unstable.
* @cite Stable Monte Carlo algorit&sn for fermion lattice systems at low temperatures
*/
void Hubbard::calGreensFunTHirsh()
{
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++) {
		this->Gtime_[_SPIN_].eye();
		algebra::setSubMFromM(this->Gtime_[_SPIN_], this->B_[_SPIN_][this->M_ - 1], 0, (this->M_ - 1) * transformSize_, transformSize_, transformSize_, true, false);
		// other sectors
		for (int _sec = 0; _sec < this->M_ - 1; _sec++) {
			const auto row = (_sec + 1	) * transformSize_;
			const auto col = (_sec		) * transformSize_;
			algebra::setSubMFromM(this->Gtime_[_SPIN_], this->B_[_SPIN_][_sec], row, col, transformSize_, transformSize_, true, true);
		}
		this->Gtime_[_SPIN_] = arma::inv(this->Gtime_[_SPIN_]);
	}
}

/*
* @brief Calculate the unequal time Green's function using stable multiplication method
*/
void Hubbard::calGreensFunT()
{

}

#endif
// ######################################################## S E T T E R S #########################################################

/*
* @brief Sets the initial state of the Hubbard-Stratonovich fields
*/
auto Hubbard::setHS(HS_CONF_TYPES _t) -> void
{
	switch (_t)
	{
	case HIGH_T:
		for (int i = 0; i < this->transformSize_; i++) 
			for (int l = 0; l < this->M_; l++) 
				this->HSFields_(l, i) = this->ran_.random(0.0, 1.0) > 0.5 ? 1 : -1;
		break;
	case LOW_T:
		this->HSFields_.ones(this->M_, this->transformSize_);
		break;
	}
	//this->HSFields_.print();
}

/*
* @brief Sets the DQMC simulation directories
* @param _m main directory to be saved onto
*/
auto Hubbard::setDir(std::string _m) -> void
{
	this->dir_->mainDir			= _m + this->lat_->get_info()		+ kPS	+	this->getInfo()		+	kPS;
	this->dir_->equalTimeDir	= this->dir_->mainDir	+ "equal"	+ kPS;
	this->dir_->unequalTimeDir	= this->dir_->mainDir	+ "unequal"	+ kPS;

	this->dir_->unequalGDir		= this->dir_->unequalTimeDir		+	"green"						+	kPS;
	this->dir_->equalGDir		= this->dir_->equalTimeDir			+	"green"						+	kPS;

	this->dir_->uneqalCorrDir	= this->dir_->unequalTimeDir		+	"corr"						+	kPS;
	this->dir_->equalCorrDir	= this->dir_->equalTimeDir			+	"corr"						+	kPS;

	this->dir_->createDQMCDirs(this->ran_);
}

/*
* @brief Sets the information about the model
*/
void Hubbard::setInfo()
{
	bool _different_U	= !std::equal(this->U_.begin() + 1, this->U_.end(), this->U_.begin());
	bool _different_mu	= !std::equal(this->mu_.begin() + 1, this->mu_.end(), this->mu_.begin());
	bool _different_t	= !std::equal(this->t_.begin() + 1, this->t_.end(), this->t_.begin());

	// next nearest
	bool _different_tt	= !std::equal(this->tt_.begin() + 1, this->tt_.end(), this->tt_.begin());
	auto _zeroVec		= v_1d<double>(this->transformSize_, 0);
	bool _zero_tt		= std::equal(this->tt_.begin(), this->tt_.end(), _zeroVec.begin());

	this->info_ = "Hubbard,";
	this->info_ +=			VEQV(M,		M_);
	this->info_ += "," +	VEQV(M0,	M0_);
	this->info_ += "," +	VEQV(p,		p_);
	this->info_ += "," +	VEQVP(dt,	dtau_,	3);
	this->info_ += "," +	VEQVP(beta, beta_,	3);
	this->info_ += "," +	(_different_U	? "U=r"		:	VEQVP(U, U_[0], 3));
	this->info_ += "," +	(_different_mu	? "mu=r"	:	VEQVP(mu, mu_[0], 3));
	this->info_ += "," +	(_different_t	? "t=r"		:	VEQVP(t, t_[0], 3));

	// check if next nearest exist
	if(!_zero_tt)
		this->info_ += "," + (_different_tt	? "tt=r"	:	VEQVP(tt, tt_[0], 3));

	for (const auto& par : splitStr(this->lat_->get_info(), ","))
		LOGINFO(par, LOG_TYPES::TRACE, 2);

	for (const auto& par : splitStr(this->info_, ","))
		LOGINFO(par, LOG_TYPES::TRACE, 2);

}

// ####################################################### U P D A T E R S ########################################################

/*
* @brief Update the interaction matrix for current spin whenever the given lattice site HS field is changed.
*/
void Hubbard::updInteracts(uint _site, uint _t)
{
	auto _delta							=	this->calDelta();
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum_)
#endif
	for(auto _spin = 0; _spin < this->spinNumber_; _spin++)
		this->IExp_[_spin](_site, _t)	*=	_delta[_spin];
}

/*
* @brief After accepting spin change update the B matrix by multiplying it by the diagonal element (the delta)
* @param _site current lattice site for the update
*/
void Hubbard::updPropagatB(uint _site, uint _t)
{
	const auto _delta					=	this->calDelta();
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum_)
#endif
	for (int i = 0; i < this->transformSize_; i++)
	{
		this->B_[_UP_][_t](i, _site)	*=	_delta[_UP_];
		this->B_[_DN_][_t](i, _site)	*=	_delta[_DN_];
		this->iB_[_UP_][_t](_site, i)	*=	_delta[_DN_];
		this->iB_[_DN_][_t](_site, i)	*=	_delta[_UP_];
	}
}

/*
* @brief After changing one spin we need to update the Green matrices via the Dyson equation
* @param _site the site on which HS field has been changed
* @param p probability of the change
*/
void Hubbard::updEqlGreens(uint _site, const spinTuple_& p)
{
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++) {
		// use the D matrix from UDT to save the row which does not change
		this->udt_[_SPIN_]->D	=	((this->G_[_SPIN_].row(_site)).as_col());
		const double gammaOverP	=	(*this->currentGamma_)[_SPIN_] / p [_SPIN_];
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum_)
#endif
		for (int _a = 0; _a < transformSize_; _a++) {
			const double _kron [[maybe_unused]]		=	(_a == _site) ? 1.0 : 0.0;
			const double G_ai						=	this->G_[_SPIN_](_a, _site);
			for (int _b = 0; _b < this->transformSize_; _b++)
				this->G_[_SPIN_](_a, _b)			-=	(_kron - G_ai) * gammaOverP * this->udt_[_SPIN_]->D(_b);
		}
	}
}

/*
* @brief Update the Green's matrices after going to next Trotter time
* @warning Remember, the time is taken to be the previous one
* @param _t time that we update to _t + 1
*/
void Hubbard::updNextGreen(uint _t)
{
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum_)
#endif
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
		this->G_[_SPIN_]		=	(this->B_[_SPIN_][_t] * this->G_[_SPIN_]) * this->iB_[_SPIN_][_t];
}

/*
* @brief Update the Green's matrices after going to next Trotter time
* @warning Remember, the time is taken to be the previous one
* @param _t time that we update to _t - 1
*/
void Hubbard::updPrevGreen(uint _t)
{
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum_)
#endif
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
		this->G_[_SPIN_]		=	(this->iB_[_SPIN_][int(_t) - 1] * this->G_[_SPIN_]) * this->iB_[_SPIN_][int(_t) - 1];
}

/*
* @brief How to update the Green's function to the next time.
* If the time step % from_scratch == 0 -> we recalculate from scratch.
* @param _t Current time step that needs to be propagated
*/
void Hubbard::updGreenStep(uint _t)
{
	if (_t % this->fromScratchNum_ == 0)
	{
		const int cast				= static_cast<int>(_t / double(this->M0_));
		const int sectorToUpdate	= modEUC<int>((cast - 1), this->p_);
		this->calPropagatBC(sectorToUpdate);
		this->calGreensFunC(cast);
	}
	else
		this->updNextGreen(_t - 1);
}

// ###################################################### E V O L U T I O N ########################################################

/*
* @brief heat - bath based algorithm for the propositon of HS field spin flip
* @param _site lattice site at which we try a flip
* @returns sign of the probility
*/
int Hubbard::eqSingleStep(int _site)
{
	this->calGamma(_site);
	this->calProba(_site);
	this->proba_				=	this->currentProba_[_UP_] * this->currentProba_[_DN_];
	this->proba_				=	this->proba_ / (1.0 + this->proba_);
	const int _sign				=	(this->proba_ >= 0) ? 1 : -1;
	if (this->ran_.random<double>(0.0, 1.0) <= _sign * this->proba_)
	{
		this->HSFields_(this->tau_, _site) *= -1.0;
		this->updPropagatB(_site, this->tau_);
		this->updEqlGreens(_site, this->currentProba_);
	}
	return _sign;
}

/*
* @brief sweep space-time forward in time
*/
double Hubbard::sweepForward()
{
	this->configSigns_	=	{};
	this->configSign_	=	1;
	for (uint _tau = 0; _tau < this->M_; ++_tau)
	{
		this->tau_		=	_tau;
		this->updGreenStep(this->tau_);
		configSign_		=	(this->sweepLattice() > 0) ? +this->configSign_ : -this->configSign_;
		configSigns_.push_back(configSign_);
	}
	return std::accumulate(configSigns_.begin(), configSigns_.end(), 0.0) / configSigns_.size();
}

// ##################################################### S I M U L A T I O N #######################################################

/*
* @brief Drive the system to equilibrium
* @param MCs number of Monte Carlo steps
* @param _quiet wanna talk?
*/
void Hubbard::equalibrate(uint MCs, bool _quiet, clk::time_point _t)
{
	if (_quiet && MCs != 1)
	{
#pragma omp critical 
		LOGINFO("Starting the relaxation for " + this->info_, LOG_TYPES::TRACE, 2);
		this->posNum_		=		0;
		this->negNum_		=		0;
	}

#ifdef DQMC_SAVE_CONF
	LOGINFO("Saving configurations of Hubbard Stratonovich fields", LOG_TYPES::TRACE, 3);
#endif

	// reset the progress bar
	this->pBar_				=		pBar(20, MCs, _t);

	// sweep all
	for (int step = 0; step < MCs; step++) {
		auto _sign	[[maybe_unused]] = this->sweepForward();
#ifdef DQMC_SAVE_CONF
		this->saveConfig("\t");
#endif
		this->configSign_ > 0 ? this->posNum_++ : this->negNum_++;
		PROGRESS_UPD_Q(step, this->pBar_, "PROGRESS RELAXATION", !_quiet);
	}
}

/*
* @brief Average the system physical measurements in the equilibrium
* @param corrTime time after which the new measurements are uncorrelated
* @param avNum number of averages to be taken
* @param quiet quiet?
*/
void Hubbard::averaging(uint MCs, uint corrTime, uint avNum, uint buckets, bool _quiet, clk::time_point _t)
{
#pragma omp critical
	LOGINFO(LOG_TYPES::TRACE, "Starting the averaging for " + this->info_, 50, '#', 1);
	LOGINFO(2);

	// initialize stuff
	this->negNum_					=		0;
	this->posNum_					=		0;
	this->avs_->reset(avNum);
	this->pBar_						=		pBar(10, avNum * buckets, _t);

	// check if this saved already
	for (int step = 1; step <= avNum * buckets; step++) 
	{
		// check the calculation of time Green's
#ifdef DQMC_CAL_TIMES
		// check the usage of Hirsh or by hand
#	ifdef DQMC_USE_HIRSH
		this->calGreensFunTHirsh();
#	else
		this->calGreensFunT();
#	endif
#endif
		// go through the imaginary times
		for (auto _tau = 0; _tau < this->M_; _tau++) 
		{
			this->tau_				=		_tau;
			// if Hirsh is performed, we don't need to calculate the update of the Green's function to next time
#if !defined DQMC_CAL_TIMES || defined DQMC_CAL_TIMES && !defined DQMC_USE_HIRSH
			this->updGreenStep(this->tau_);
#endif
			// save the diagonal part of the Green's function on the fly
#ifdef DQMC_CAL_TIMES
			const uint _element		=		this->tau_ * transformSize_;
			for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
#	ifdef DQMC_USE_HIRSH
				algebra::setMFromSubM(this->G_[_SPIN_], this->Gtime_[_SPIN_], _element, _element, transformSize_, transformSize_, false);
#	else
				algebra::setSubMFromM(this->Gtime_[_SPIN_], this->G_[_SPIN_], _element, _element, transformSize_, transformSize_, false);
#	endif
#endif
			// go through the lattice sites
			for (int _site = 0; _site < transformSize_; _site++)
				this->avSingleStep(_site, this->configSign_);
		}
		// check the sign
		this->configSigns_.push_back(this->configSign_);
		this->configSign_ > 0 ? this->posNum_++ : this->negNum_++;

		// save the averages
		if (step % avNum == 0)
			this->saveBuckets(step, avNum, _t, step != (avNum * buckets));

		// kill correlations
		for (int _cor = 0; _cor < corrTime; _cor++)
			this->sweepForward();

		PROGRESS_UPD_Q(step, this->pBar_, "PROGRESS AVERAGES", !_quiet);
	}
	// calculate the average sign

}

// ####################################################### A V E R A G E S #########################################################

/*
* @brief A single step for calculating averages inside a simulation loop.
* @param _currI current HS spin
* @param _sign current sign to multiply the averages by
*/
void Hubbard::avSingleStep(int _currI, int _sign)
{
	auto _band	=	static_cast<uint>(_currI / this->Ns_);

	// swich band
	if (_band == 0)
	{
		// mz2
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mz2, 0);
		// mx2
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mx2, 0);
		// n
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Occupation, 0);
		// Ek
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Ek, 0);
	}
	else if (_band == 1)
	{
		// mz2
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mz2, 1);
		// mx2
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mx2, 1);
		// n
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Occupation, 1);
		// Ek
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Ek, 1);
	}
	else
	{
		// mz2
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mz2, 2);
		// mx2
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mx2, 2);
		// n
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Occupation, 2);
		// Ek
		INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Ek, 2);
	}

	// save i'th point coordinates
	const auto xi			=	this->lat_->get_coordinates(_currI % this->Ns_, Lattice::X);
	const auto yi			=	this->lat_->get_coordinates(_currI % this->Ns_, Lattice::Y);
	const auto zi			=	this->lat_->get_coordinates(_currI % this->Ns_, Lattice::Z);
	const auto ith_coord	=	std::make_tuple(xi, yi, zi);

	// -------------------------------- CORRELATIONS ----------------------------------------
#ifdef DQMC_CAL_TIMES
	auto [xNum, yNum, zNum]	=	this->lat_->getNumElems();
#endif
	for (int _J = 0; _J < this->lat_->get_Ns(); _J++)
	{
		auto _currJ			=	_band * this->Ns_ + _J;
		auto [x, y, z]		=	this->lat_->getSiteDifference(ith_coord, _J);
		auto [xx, yy, zz]	=	this->lat_->getSymPos(x, y, z);

		if (_band == 0)
		{
			INVOKE_TWO_PARTICLE_CAL(this->avs_, Mz2, 0, xx, yy, zz);
			INVOKE_TWO_PARTICLE_CAL(this->avs_, Occupation, 0, xx, yy, zz);
		}
		else if (_band == 1)
		{
			INVOKE_TWO_PARTICLE_CAL(this->avs_, Mz2, 1, xx, yy, zz);
			INVOKE_TWO_PARTICLE_CAL(this->avs_, Occupation, 1, xx, yy, zz);
		}
		else if (_band == 2)
		{
			INVOKE_TWO_PARTICLE_CAL(this->avs_, Mz2, 2, xx, yy, zz);
			INVOKE_TWO_PARTICLE_CAL(this->avs_, Occupation, 2, xx, yy, zz);
		}
#ifdef DQMC_CAL_TIMES
		this->avSingleStepUneq(xx + _band * xNum, yy + _band * yNum, zz + _band * zNum, _currI, _currJ, _sign);
#endif
	}
}

// ############################################### A V E R A G E S   U N E Q U A L #################################################

/*
* @brief Calculates the single step for unequal-time simulation properties
* @param xx X-direction position of saved value
* @param yy Y-direction position of saved value
* @param zz Z-direction position of saved value
* @param _i current Green's function row
* @param _j current Green's function col
* @param _s current configuration sign
*/
void Hubbard::avSingleStepUneq(int xx, int yy, int zz, int _i, int _j, int _s)
{
	//? handle zero time difference here in greens
	//! we handle it with the calculated current Green's functions
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++) 
	{
		//this->avs_->av_GTimeDiff_[_SPIN_][0](xx, yy)		+=		_s * this->G_[_SPIN_](_i, _j);
		//this->avs_->sd_GTimeDiff_[_SPIN_][0](xx, yy)		+=		this->G_[_SPIN_](_i, _j) * this->G_[_SPIN_](_i, _j);
#ifdef DQMC_CAL_TIMES_ALL
#ifndef _DEBUG
#	pragma omp parallel for num_threads(this->threadNum_)
#endif
		for (int tim2 = 0; tim2 < this->M_; tim2++) {
#else
		for (int tim2 = 0; tim2 < this->tau_; tim2++) {
#endif
			int tim		=	(int)this->tau_ - tim2;
			auto xk		=	_s;
			// handle antiperiodicity
#ifdef DQMC_CAL_TIMES_ALL
			if (tim < 0) 
			{
				xk		*=	-1;
				tim		+=	this->M_;
			}
#endif
			const auto col		=	tim2 * this->transformSize_;
			const auto row		=	this->tau_ * this->transformSize_;
			const auto elem		=	xk * this->Gtime_[_SPIN_](row + _i, col + _j);
			// save only the positive first half
			this->avs_->av_GTimeDiff_[_SPIN_][tim](xx, yy, zz)	+=		elem;
			this->avs_->sd_GTimeDiff_[_SPIN_][tim](xx, yy, zz)	+=		elem * elem;

		}
	}
}

// ######################################################### S A V E R S ###########################################################

/*
* @brief Save the equal time Green's functions
* @param _step current step time
*/
void Hubbard::saveGreens(uint _step)
{
	const std::string _signStr	= (this->configSign_ == 1) ? "+" : "-";
	this->tmpG_[0]				= (this->G_[_UP_] + this->G_[_DN_]) / 2.0;

	this->tmpG_[0].save	(arma::hdf5_name(this->dir_->equalGDir		+ "G_" + STR(_step) + "_" + _signStr + "_"		+ this->dir_->randomSampleStr + ".h5", "G(t)"));
	this->G_[_UP_].save	(arma::hdf5_name(this->dir_->equalGDir		+ "G_" + STR(_step) + "_" + _signStr + "_"		+ this->dir_->randomSampleStr + ".h5", "Gup(t)", arma::hdf5_opts::append));
	this->G_[_DN_].save	(arma::hdf5_name(this->dir_->equalGDir		+ "G_" + STR(_step) + "_" + _signStr + "_"		+ this->dir_->randomSampleStr + ".h5", "Gdn(t)", arma::hdf5_opts::append));
	this->HSFields_.save(arma::hdf5_name(this->dir_->equalGDir		+ "G_" + STR(_step) + "_" + _signStr + "_"		+ this->dir_->randomSampleStr + ".h5", "HS",	 arma::hdf5_opts::append));
}


// TODO ------------------------>

//void hubbard::HubbardModel::cal_hopping_exp()
//{
//	bool checkerboard = false;
//	const int Lx = this->lattice->get_Lx();
//	const int Ly = this->lattice->get_Ly();
//
//	// USE CHECKERBOARD
//	const int dim = this->lattice->get_Dim();
//	if (checkerboard && this->getDim() == 2 && Lx == Ly) {
//		arma::mat Kx_a, Kx_b, Ky_a, Ky_b, Kz_a, Kz_b;
//		Kx_a.zeros(this->Ns, this->Ns);
//		Kx_b = Kx_a;
//		if (dim >= 2) {
//			// 2D
//			Ky_a = Kx_a;
//			Ky_b = Ky_a;
//			if (dim == 3) {
//				// 3D
//				Kz_a = Kx_a;
//				Kz_b = Kz_a;
//			}
//		}
//		// set elements
//		for (int i = 0; i < this->Ns; i++) {
//			const int n_of_neigh = this->lattice->get_nn_number(i);												// take number of nn at given site
//			for (int j = 0; j < n_of_neigh; j++) {
//				const int where_neighbor = this->lattice->get_nn(i, j);											// get given nn
//				const int y = i / Lx;
//				const int x = i - y * Lx;
//				const int y_nei = where_neighbor / Ly;
//				const int x_nei = where_neighbor - y_nei * Lx;
//				if (y_nei == y) {
//					// even rows
//					if (i % 2 == 0) {
//						if (x_nei == (x + 1) % Lx) {
//							Kx_a(i, where_neighbor) = 1;
//						}
//						else {
//							Kx_b(i, where_neighbor) = 1;
//						}
//					}
//					// odd rows
//					else {
//						if (x_nei == (x + 1) % Lx) {
//							Kx_b(i, where_neighbor) = 1;
//						}
//						else {
//							Kx_a(i, where_neighbor) = 1;
//						}
//					}
//				}
//				else {
//					// ky
//					if (where_neighbor % 2 == 0) {
//						Ky_a(i, where_neighbor) = 1;
//						Ky_a(where_neighbor, i) = 1;
//					}
//					else {
//						Ky_b(i, where_neighbor) = 1;
//						Ky_b(where_neighbor, i) = 1;
//					}
//				}
//			}
//		}
//		/*arma::mat K(Ns, Ns, arma::fill::zeros);
//		for(int x = 0; x < Lx; ++x) {
//			for(int y = 0; y < Ly; ++y) {
//				// chemical potential 'mu' on the diagonal
//				//K(x + Lx * y, x + Lx * y) -= this->mu;
//				K(x + Lx * y, ((x + 1) % Lx) + Lx * y) = this->t[0];
//				K(((x + 1) % Lx) + Lx * y, x + Lx * y) = this->t[0];
//				K(x + Lx * y, x + Lx * ((y + 1) % Lx)) = this->t[0];
//				K(x + Lx * ((y + 1) % Lx), x + Lx * y) = this->t[0];
//			}
//		}*/
//
//		//Kx_a.print("Kx a:");
//		//Kx_b.print("Kx b:");
//		//Ky_a.print("Ky a:");
//		//Ky_b.print("Ky b:");
//		//this->hopping_exp = Kx_a + Kx_b + Ky_a + Ky_b;
//		//(this->hopping_exp - K).print();
//		//this->hopping_exp.print("HOPPING MATRIX:");
//
//		//arma::mat tmp_exp = arma::expmat(this->hopping_exp);
//		//tmp_exp.print("NORMALLY CALCULATED EXPONENT");
//
//		arma::mat one = arma::eye(this->Ns, this->Ns);
//		one *= cosh(this->dtau * t[0]);
//		const double sinus = sinh(this->dtau * this->t[0]);
//
//		Kx_a = (Kx_a * sinus + one);
//		Kx_b = (Kx_b * sinus + one);
//		Ky_a = (Ky_a * sinus + one);
//		Ky_b = (Ky_b * sinus + one);
//		this->hopping_exp = Ky_a * Kx_a * Ky_b * Kx_b;
//		//this->hopping_exp.print("BETTER CALCULATED EXP");
//		return;
//	}
//	else
//	{
//		for (int i = 0; i < this->Ns; i++) {
//			//this->hopping_exp(i, i) = this->dtau * this->mu;														// diagonal elements
//			const auto n_of_neigh = this->lattice->get_nn_number(i);											// take number of nn at given site
//			for (int j = 0; j < n_of_neigh; j++) {
//				const int where_neighbor = this->lattice->get_nn(i, j);											// get given nn
//				this->hopping_exp(i, where_neighbor) = this->dtau * this->t[0];									// assign non-diagonal elements
//			}
//		}
//		//this->hopping_exp.print("hopping before exponentiation");
//		//arma::vec eigval;
//		//arma::mat eigvec;
//		//arma::eig_sym(eigval, eigvec, this->hopping_exp);
//		//stout << "eigenvalues:\n" << eigval.t() << std::endl;
//		//arma::mat jordan = eigvec.i() * this->hopping_exp * eigvec;
//		//jordan = arma::expmat_sym(jordan);
//		//this->hopping_exp = eigvec * this->hopping_exp * eigvec.i();
//#pragma omp critical
//		this->hopping_exp = arma::expmat(this->hopping_exp);													// take the exponential
//		//this->hopping_exp.print("hopping after exponentiation");
//	}
//}

//double hubbard::HubbardModel::cal_ch_correlation(int sign, int current_elem_i, int current_elem_j, const mat& g_up, const mat& g_down)
//{
//	double delta_ij = 0.0L;
//	if (current_elem_i == current_elem_j) {
//		delta_ij = 1.0L;
//	}
//	return sign * (((1 - g_up(current_elem_i, current_elem_i)) * (1 - g_up(current_elem_j, current_elem_j))					//sigma = sigma' = up
//		+ (1 - g_down(current_elem_i, current_elem_i)) * (1 - g_down(current_elem_j, current_elem_j))				//sigma = sigma' = down
//		+ (1 - g_down(current_elem_i, current_elem_i)) * (1 - g_up(current_elem_j, current_elem_j))					//sigma = down, sigma' = up
//		+ (1 - g_up(current_elem_i, current_elem_i)) * (1 - g_down(current_elem_j, current_elem_j))					//sigma = up, sigma' = down
//		+ ((delta_ij - g_up(current_elem_j, current_elem_i)) * g_up(current_elem_i, current_elem_j))				//sigma = sigma' = up
//		+ ((delta_ij - g_down(current_elem_j, current_elem_i)) * g_down(current_elem_i, current_elem_j))));			//sigma = sigma' = down
//}

//? -------------------------------------------------------- EQUAL

/*
* Calculate Green with QR decomposition using LOH. Here we calculate the Green matrix at a given time, so we need to take care of the times away from precalculated sectors
* @cite doi:10.1016/j.laa.2010.06.023
* @param which_time The time at which the Green's function is calculated
*/
//void hubbard::HubbardQR::cal_green_mat(int which_time) {
//	auto tim = which_time;
//	int sec = (which_time / this->M_0);							// which sector is used for M_0 multiplication
//	int sector_end = (sec + 1) * this->M_0 - 1;
//	// multiply those B matrices that are not yet multiplied
//	b_mat_mult_left(
//		tim + 1, sector_end,
//		this->b_mat_up[tim], this->b_mat_down[tim],
//		tempGreen_up, tempGreen_down
//	);
//	// using tempGreens to store the starting multiplication
//
//	// decomposition
//	setUDTDecomp(this->tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
//	setUDTDecomp(this->tempGreen_down, Q_down, R_down, P_down, T_down, D_down);
//
//	// multiply by new precalculated sectors
//	for (int i = 1; i < this->p - 1; i++)
//	{
//		sec++;
//		if (sec == this->p) sec = 0;
//		multiplyMatricesQrFromRight(this->b_up_condensed[sec], Q_up, R_up, P_up, T_up, D_up);
//		multiplyMatricesQrFromRight(this->b_down_condensed[sec], Q_down, R_down, P_down, T_down, D_down);
//	}
//	// we need to handle the last matrices that ale also away from M_0 cycle
//	sec++;
//	if (sec == this->p) sec = 0;
//	sector_end = myModuloEuclidean(which_time - 1, this->M);
//	tim = sec * this->M_0;
//	b_mat_mult_left(tim + 1, sector_end,
//		this->b_mat_up[tim], this->b_mat_down[tim],
//		tempGreen_up, tempGreen_down);
//	multiplyMatricesQrFromRight(tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
//	multiplyMatricesQrFromRight(tempGreen_down, Q_down, R_down, P_down, T_down, D_down);
//
//	//stout << EL;
//	//this->green_up = T_up.i() * (Q_up.t() * T_up.i() + DIAG(R_up)).i()*Q_up.t();
//	//this->green_down = T_down.i() * (Q_down.t() * T_down.i() + DIAG(R_down)).i()*Q_down.t();
//
//	// Correction terms
//	makeTwoScalesFromUDT(R_up, D_up);
//	makeTwoScalesFromUDT(R_down, D_down);
//	// calculate equal time Green
//	//this->green_up = arma::inv(DIAG(D_up) * Q_up.t() + DIAG(R_up) * T_up) * DIAG(D_up) * Q_up.t();
//	//this->green_down = arma::inv(DIAG(D_down) * Q_down.t() + DIAG(R_down) * T_down) * DIAG(D_down) * Q_down.t();
//	this->green_up = arma::solve(DIAG(D_up) * Q_up.t() + DIAG(R_up) * T_up, DIAG(D_up) * Q_up.t());
//	this->green_down = arma::solve(DIAG(D_down) * Q_down.t() + DIAG(R_down) * T_down, DIAG(D_down) * Q_down.t());
//}

/*
* Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023 with premultiplied B matrices.
* For more look into :
* @copydetails "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems".
* In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices.
* @param sector Which sector does the Green's function starrts at
*/
//void hubbard::HubbardQR::cal_green_mat_cycle(int sector) {
//	auto sec = sector;
//	setUDTDecomp(this->b_up_condensed[sec], Q_up, R_up, P_up, T_up, D_up);
//	setUDTDecomp(this->b_down_condensed[sec], Q_down, R_down, P_down, T_down, D_down);
//	for (int i = 1; i < this->p; i++) {
//		sec++;
//		if (sec == this->p) sec = 0;
//		multiplyMatricesQrFromRight(this->b_up_condensed[sec], Q_up, R_up, P_up, T_up, D_up);
//		multiplyMatricesQrFromRight(this->b_down_condensed[sec], Q_down, R_down, P_down, T_down, D_down);
//	}
//	// making two scales for the decomposition following Loh
//	//makeTwoScalesFromUDT(R_up, D_up);
//	//makeTwoScalesFromUDT(R_down, D_down);
//	makeTwoScalesFromUDT(R_up, D_min_up, D_max_up);
//	makeTwoScalesFromUDT(R_down, D_min_down, D_max_down);
//
//	//this->green_up = arma::inv(DIAG(D_up) * Q_up.t() + DIAG(R_up) * T_up) * DIAG(D_up) * Q_up.t();
//	//this->green_down = arma::inv(DIAG(D_down) * Q_down.t() + DIAG(R_down) * T_down) * DIAG(D_down) * Q_down.t();
//
//	this->green_up = arma::solve(arma::inv(DIAG(D_min_up)) * Q_up.t() + DIAG(D_max_up) * T_up, arma::inv(DIAG(D_min_up)) * Q_up.t());
//	this->green_down = arma::solve(arma::inv(DIAG(D_min_down)) * Q_down.t() + DIAG(D_max_down) * T_down, arma::inv(DIAG(D_min_down)) * Q_down.t());
//
//}


/*
//* @brief Calculating unequal time Green's functions given by Bl_1*...*B_{l2+1}*G_{l2+1} \\rightarrow [B_{l2+1}^{-1}...B_l1^{-1} + B_l2...B_1B_{M-1}...B_{l1+1}]^{-1}.
//* Make inverse of function of type (Ql*diag(Rl)*Tl + Qr*diag(Rr)*Tr)^(-1) using:
//* @cite SciPost Phys. Core 2, 011 (2020)
//* @param t1 left time t1>t2
//* @param t2 right time t2<t1
//* @param inv_series_up precalculated inverse matrices multiplication for spin up
//* @param inv_series_down precalculated inverse matrices multiplication for spin down
//*/
//void hubbard::HubbardQR::uneqG_t1gtt2(int t1, int t2, const mat& inv_up, const mat& inv_down, const mat& up, const mat& down)
//{
//	assert("t1 should be higher than t2" && t1 >= t2);
//	const auto row = t1 * this->Ns;
//	const auto col = t2 * this->Ns;
//
//	//! ------------------------------------ up ------------------------------------ 
//	//? USE DOWN MATRICES AS HELPERS FOR RIGHT SUM TO SAVE PRECIOUS MEMORY!
//	//! B(t2 + 1)^(-1)...B(t1)^(-1)
//	setUDTDecomp(inv_up, Q_up, R_up, P_up, T_up, D_up);												// decompose the premultiplied inversions to up temporaries
//
//	//! B(M-1)...B(t1 + 1)
//	setUDTDecomp(up, Q_down, R_down, P_down, T_down, D_down);										// decompose and use down matrices as temporaries + equal time Green at [0]
//
//	//! SET MATRIX ELEMENT
//	setSubmatrixFromMatrix(this->g_up_time,
//		inv_left_plus_right_qr(
//			Q_up, R_up, P_up, T_up, D_up,
//			Q_down, R_down, P_down, T_down, D_down,
//			D_tmp
//		),
//		row, col, this->Ns, this->Ns, false);
//
//	//! ------------------------------------ down ------------------------------------
//	//? USE UP MATRICES AS HELPERS FOR RIGHT SUM TO SAVE PRECIOUS MEMORY!
//	//! B(t2 + 1)^(-1)...B(t1)^(-1)
//	setUDTDecomp(inv_down, Q_up, R_up, P_up, T_up, D_up);											// decompose the premultiplied inversions to up temporaries
//
//	//! B(M-1)...B(t1 + 1)
//	setUDTDecomp(down, Q_down, R_down, P_down, T_down, D_down);
//
//	//! SET MATRIX ELEMENT
//	setSubmatrixFromMatrix(this->g_down_time,
//		inv_left_plus_right_qr(
//			Q_up, R_up, P_up, T_up, D_up,
//			Q_down, R_down, P_down, T_down, D_down,
//			D_tmp
//		),
//		row, col, this->Ns, this->Ns, false);
//}
//
////TODO ----------------------->
///**
//* @param t1
//* @param t2
//*/
//void hubbard::HubbardQR::uneqG_t1ltt2(int t1, int t2)
//{
//	if (t2 <= t1) throw "can't do that m8\n";
//	// make inverse of function of type (Ql*diag(Rl)*Tl + Qr*diag(Rr)*Tr)^(-1) using SciPost Phys. Core 2, 011 (2020)
//	const auto row = t1 * this->Ns;
//	const auto col = t2 * this->Ns;
//
//	// ------------------------------------ up ------------------------------------ USE DOWN MATRICES AS HELPERS FOR RIGHT SUM!
//	// B(l2)...B(l1+1)
//	//setUDTDecomp(inv_series_up, Q_up, R_up, P_up, T_up, D_up);
//	this->tempGreen_up = arma::inv(T_up) * DIAG(D_up) * Q_up.t();
//	// B(M-1)...B(t1 + 1)
//	//setUDTDecomp(this->g_up_eq[t1], Q_down, R_down, P_down, T_down, D_down);
//	//setUDTDecomp(this->g_up_eq[0], Q_down, R_down, P_down, T_down, D_down);
//	// B(t2)...B(0)
//	//multiplyMatricesQrFromRight(this->g_up_tim[t2], Q_down, R_down, P_down, T_down, D_down);
//	// SET MATRIX ELEMENT
//	setUDTDecomp(inv_left_plus_right_qr(Q_up, R_up, P_up, T_up, D_up, \
//		Q_down, R_down, P_down, T_down, D_down, D_tmp), Q_up, R_up, P_up, T_up, D_up);
//	setUDTDecomp(this->tempGreen_up, Q_down, R_down, P_down, T_down, D_down);
//	setUDTDecomp(DIAG(R_up) * T_up * T_down.i() - Q_up.t() * Q_down * DIAG(R_down), Q_down, R_up, P_up, T_up, D_up);
//
//	setSubmatrixFromMatrix(this->g_up_time, (Q_up * Q_down) * DIAG(R_up) * (T_up * T_down), row, col, this->Ns, this->Ns, false);
//
//	// ------------------ down ------------------
//	// B(l2)...B(l1+1)
//	//setUDTDecomp(inv_series_down, Q_down, R_down, P_down, T_down, D_down);
//	this->tempGreen_down = arma::inv(T_down) * DIAG(D_down) * Q_down.t();
//	// B(M-1)...B(t1 + 1)
//	//setUDTDecomp(this->g_down_eq[t1], Q_up, R_up, P_up, T_up, D_up);
//	//setUDTDecomp(this->g_down_eq[0], Q_up, R_up, P_up, T_up, D_up);
//	// B(t2)...B(0)
//	//multiplyMatricesQrFromRight(this->g_down_tim[t2], Q_up, R_up, P_up, T_up, D_up);
//	// SET MATRIX ELEMENT
//	setUDTDecomp(inv_left_plus_right_qr(Q_down, R_down, P_down, T_down, D_down, \
//		Q_up, R_up, P_up, T_up, D_up, D_tmp), Q_down, R_down, P_down, T_down, D_down);
//	setUDTDecomp(this->tempGreen_up, Q_up, R_up, P_up, T_up, D_up);
//	setUDTDecomp(DIAG(R_down) * T_down * T_up.i() - Q_down.t() * Q_up * DIAG(R_up), Q_up, R_down, P_down, T_down, D_down);
//
//	setSubmatrixFromMatrix(this->g_down_time, Q_down * Q_up * DIAG(R_down) * T_down * T_up, row, col, this->Ns, this->Ns, false);
//}


/*
* @brief Calculate time displaced Greens. NOW ONLY t1>t2
* @TODO make t2>t1
*/
//void hubbard::HubbardQR::cal_green_mat_times()
//{
//	// -------------------------------- calculate non-inverses condensed --------------------------------
//	auto tim = 0;
//	this->b_downs[tim] = this->b_mat_down[tim];
//	this->b_ups[tim] = this->b_mat_up[tim];
//	this->b_downs_i[tim] = this->b_mat_down_inv[tim];
//	this->b_ups_i[tim] = this->b_mat_up_inv[tim];
//	for (int i = 1; i < this->M_0; i++) {
//		this->b_downs[tim + i] = this->b_mat_down[tim + i] * this->b_downs[tim + i - 1];
//		this->b_ups[tim + i] = this->b_mat_up[tim + i] * this->b_ups[tim + i - 1];
//		this->b_downs_i[tim + i] = this->b_downs_i[tim + i - 1] * this->b_mat_down_inv[tim + i];
//		this->b_ups_i[tim + i] = this->b_ups_i[tim + i - 1] * this->b_mat_up_inv[tim + i];
//	}
//
//	// stable multiply it again to get the correct ones
//	//setUDTDecomp(this->b_downs[this->M_0 - 1], Q_down, R_down, P_down, T_down);
//	//setUDTDecomp(this->b_ups[this->M_0 - 1], Q_up, R_up, P_up, T_up);
//	//this->tempGreen_down = T_down;
//	//this->tempGreen_up = T_up;
//
//	arma::svd(Q_down, D_down, T_down, this->b_downs[this->M_0 - 1]);
//	arma::svd(Q_up, D_up, T_up, this->b_ups[this->M_0 - 1]);
//	this->tempGreen_down = T_down.t();
//	this->tempGreen_up = T_up.t();
//
//	for (int i = this->M_0; i < this->M; i++) {
//		//setUDTDecomp(this->b_mat_down[i] * Q_down * DIAG(R_down), Q_down, R_down, P_down, T_down, D_down);
//		//setUDTDecomp(this->b_mat_up[i] * Q_up * DIAG(R_up), Q_up, R_up, P_up, T_up, D_up);
//		//this->tempGreen_down = T_down * this->tempGreen_down;
//		//this->tempGreen_up = T_up * this->tempGreen_up;
//		//this->b_downs[i] = (Q_down * DIAG(R_down)) * this->tempGreen_down;
//		//this->b_ups[i] = (Q_up * DIAG(R_up)) * this->tempGreen_up;
//
//
//		arma::svd(Q_down, D_down, T_down, this->b_mat_down[i] * Q_down * DIAG(D_down));
//		arma::svd(Q_up, D_up, T_up, this->b_mat_up[i] * Q_up * DIAG(D_up));
//		this->tempGreen_down = T_down.t() * this->tempGreen_down;
//		this->tempGreen_up = T_up.t() * this->tempGreen_up;
//
//		this->b_downs[i] = (Q_down * DIAG(D_down)) * this->tempGreen_down;
//		this->b_ups[i] = (Q_up * DIAG(D_up)) * this->tempGreen_up;
//
//		makeTwoScalesFromUDT(DIAG(D_down), this->D_min_down, this->D_max_down);
//		makeTwoScalesFromUDT(DIAG(D_up), this->D_min_up, this->D_max_up);
//
//		this->b_downs_i[i] = arma::inv(DIAG(D_max_down) * this->tempGreen_down) * arma::solve(DIAG(D_min_down), EYE(Ns), arma::solve_opts::refine) * Q_down.t();
//		this->b_ups_i[i] = arma::inv(DIAG(D_max_up) * this->tempGreen_up) * arma::solve(DIAG(D_min_up), EYE(Ns), arma::solve_opts::refine) * Q_up.t();
//
//	}