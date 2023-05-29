#include "../include/Models/hubbard.h"

#include <execution>
#include <numeric>
#include <utility>

// ################################################# I N I T I A L I Z E R S ######################################################

/*
* @brief initializes the memory for all of the matrices used later
*/
void Hubbard::init()
{
	// set the write lock
	WriteLock lock(this->Mutex);

	// hopping exponent
	this->TExp_.zeros(this->Ns_, this->Ns_);

	// HS transformation fields
	this->HSFields_.ones(this->M_, this->Ns_);

	// all the spin matrices
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
	{
		// Green's matrix
		this->G_		[_SPIN_].zeros(this->Ns_, this->Ns_);
		// interaction
		this->IExp_		[_SPIN_].zeros(this->Ns_, this->M_);
		// propagators
		this->B_		[_SPIN_]	=	v_1d<arma::mat>(this->M_, ZEROM(this->Ns_));
		this->iB_		[_SPIN_]	=	v_1d<arma::mat>(this->M_, ZEROM(this->Ns_));
		this->Bcond_	[_SPIN_]	=	v_1d<arma::mat>(this->p_, ZEROM(this->Ns_));
		// initialize UDT decomposition
		this->udt_		[_SPIN_].reset(new algebra::UDT_QR(this->G_[_SPIN_]));

#ifdef DQMC_CAL_TIMES

#endif // CAL_TIMES
	}
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
		arma::mat _tmpG		=	arma::eye(this->Ns_, this->Ns_);

		// calculate the Green's function directly
		for (int _t = 0; _t < this->M_; _t++)
		{
			_tmpG			=	this->B_[_SPIN_][_tau] * _tmpG;
			_tau			=	(_tau + 1) % this->M_;
		}
		_tmpG				=	(EYE(this->Ns_) + _tmpG).i();
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
	LOGINFO(LOG_TYPES::TRACE, 2);
}

// ################################################## C A L C U L A T O R S #######################################################

/*
* @brief Function to calculate the change in the interaction exponent
* @param _site site on which the change has been made
* @returns A pair for gammas for two spin channels, 0 is spin up, 1 is spin down
*/
auto Hubbard::calGamma(uint _site) -> void
{
	if (this->REPULSIVE_)
		this->currentGamma_ = (this->HSFields_(this->tau_, _site) == 1) ? this->gammaExp_[0] : this->gammaExp_[1];
	else
		this->currentGamma_ = { this->gammaExp_[0][0], this->gammaExp_[0][0] };
}

/*
* @brief Allows to calculate the change in the interaction exponent
*/
auto Hubbard::calDelta() -> spinTuple_
{
	spinTuple_ _out;
	std::transform	(		
						this->currentGamma_.begin(),
						this->currentGamma_.end(),
						_out.begin(),
						[&](auto& i) { return i + 1; }
					);
	return _out;
}

/*
* @brief Return probabilities of spin flip for both spin channels
* @param _site flipping candidate site
* @param gii gammas for both spin channels
* @returns tuple for probabilities on both spin channels, 
* @warning remember, 0 is spin up, 1 is spin down
*/
auto Hubbard::calProba(uint _site) -> spinTuple_
{
	return	{
				1.0		+	currentGamma_[_UP_]	*	(1.0 - this->G_[SPINNUM::_UP_](_site, _site)),
				1.0		+	currentGamma_[_DN_]	*	(1.0 - this->G_[SPINNUM::_DN_](_site, _site))
			};
}

// ################################################### H A M I L T O N I A N ######################################################

/*
* @brief Function to calculate the hopping matrix exponential.
*/
auto Hubbard::calQuadratic() -> void
{
	// cacluate the hopping matrix
	this->TExp_.zeros(this->Ns_, this->Ns_);
	for (int _site = 0; _site < this->Ns_; _site++)
	{
		const auto neiSize			=	this->lat_->get_nn(_site);
		for (int neiNum = 0; neiNum < neiSize; neiNum++) {
			const auto nei			=	this->lat_->get_nn(_site, neiNum);								// get given nn
			this->TExp_(_site, nei) +=	this->dtau_ * this->t_[_site];									// assign non-diagonal elements
		}
	}
#pragma omp critical
	this->TExp_						=	arma::expmat(this->TExp_);
}

/*
* @brief Function to calculate the interaction exponential at all Trotter times, each column represents the given Trotter time.
*/
auto Hubbard::calInteracts() -> void
{
	const arma::Col<double> _dtauVec		=		arma::ones(this->Ns_) * this->dtau_ * (this->mu_);
	if (this->U_ > 0)
		// Repulsive case
		for (int l = 0; l < this->M_; l++) {
			// Trotter times
			this->IExp_[_UP_].col(l)		=		arma::exp(_dtauVec + this->HSFields_.row(l).t() * (	this->lambda_));
			this->IExp_[_DN_].col(l)		=		arma::exp(_dtauVec + this->HSFields_.row(l).t() * (-this->lambda_));
		}
	else if (this->U_ < 0)
		// Attractive case
		for (int l = 0; l < this->M_; l++) {
			// Trotter times
			this->IExp_[_UP_].col(l)		=		arma::exp(_dtauVec + this->HSFields_.row(l).t() * (	this->lambda_));
			this->IExp_[_DN_].col(l)		=		this->IExp_[_UP_].col(l);
		}
	else 
	{
		this->IExp_[_UP_]					=		arma::eye(this->Ns_, this->Ns_);
		this->IExp_[_DN_]					=		arma::eye(this->Ns_, this->Ns_);
	}
}

/*
* @brief Function to calculate all B propagators for a Hubbard model. Those are used for the Gibbs weights.
*/
auto Hubbard::calPropagatB() -> void
{
	for(auto _spin = 0; _spin < this->spinNumber_; _spin++)
		for (int l = 0; l < this->M_; l++) {
			// Trotter times
			this->B_[_spin][l]		=		this->TExp_ * DIAG(this->IExp_[_spin].col(l));
			this->iB_[_spin][l]		=		this->B_[_spin][l].i();
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
		this->iB_[_spin][_tau]		=		this->B_[_spin][_tau].i();
	}
}

/*
* @brief Precalculate the multiplications of B matrices according to M0 stable ones
* @param _sec the sector to calculate the stable multiplication
*/
void Hubbard::calPropagatBC(uint _sec)
{
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++) {
		auto _time						=		_sec * this->M0_;
		this->Bcond_[_SPIN_][_sec]		=		this->B_[_SPIN_][_time];
		for (int i = _time + 1; i < _time + this->M0_; i++)
			this->Bcond_[_SPIN_][_sec]	=		this->B_[_SPIN_][i] * this->Bcond_[_SPIN_][_sec];
	}
	
}

/*
* Calculate Green with QR decomposition using LOH : doi:10.1016/j.laa.2010.06.023 with premultiplied B matrices.
* For more look into :
* @copydetails "Advancing Large Scale Many-Body QMC Simulations on GPU Accelerated Multicore Systems".
* In order to do that the M_0 and p variables will be used to divide the multiplication into smaller chunks of matrices.
* @param _tau starting time sector - always marks the beginning of the sector
*/
void Hubbard::calGreensFun(uint _tau)
{
	auto _time		[[maybe_unused]]	= _tau;
	auto _sector	[[maybe_unused]]	= _tau / this->M0_;

	// decompose the matrices
	this->udt_[SPINNUM::_UP_]->decompose(this->Bcond_[SPINNUM::_UP_][_sector]);
	this->udt_[SPINNUM::_DN_]->decompose(this->Bcond_[SPINNUM::_DN_][_sector]);

	// go through each sector
	for (int i = 1; i < this->p_; i++) {
		_sector++;
		if (_sector == this->p_)
			_sector				= 0;
		this->udt_[SPINNUM::_UP_]->factMult(this->Bcond_[SPINNUM::_UP_][_sector]);
		this->udt_[SPINNUM::_DN_]->factMult(this->Bcond_[SPINNUM::_DN_][_sector]);
	}
	// making two scales for the decomposition following Loh
	this->udt_[SPINNUM::_UP_]->loh_inplace();
	this->udt_[SPINNUM::_DN_]->loh_inplace();

	// save the Green's
	this->udt_[SPINNUM::_UP_]->inv1P(this->G_[SPINNUM::_UP_]);
	this->udt_[SPINNUM::_DN_]->inv1P(this->G_[SPINNUM::_DN_]);
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
		algebra::setSubMFromM(this->Gtime_[_SPIN_], this->Bcond_[_SPIN_][this->p_ - 1], 0, (this->M_ - 1) * this->Ns_, this->Ns_, this->Ns_, true, false);
		// other sectors
		for (int _sec = 0; _sec < this->p_ - 1; _sec++) {
			const auto row	=	(_sec + 1	) * this->Ns_;
			const auto col	=	(_sec		) * this->Ns_;
			algebra::setSubMFromM(this->Gtime_[_SPIN_], this->Bcond_[_SPIN_][_sec], row, col, this->Ns_, this->Ns_, true, true);
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
		algebra::setSubMFromM(this->Gtime_[_SPIN_], this->B_[_SPIN_][this->M_ - 1], 0, (this->M_ - 1) * this->Ns_, this->Ns_, this->Ns_, true, false);
		// other sectors
		for (int _sec = 0; _sec < this->M_ - 1; _sec++) {
			const auto row = (_sec + 1) * this->Ns_;
			const auto col = (_sec)*this->Ns_;
			algebra::setSubMFromM(this->Gtime_[_SPIN_], this->B_[_SPIN_][_sec], row, col, this->Ns_, this->Ns_, true, true);
		}
		arma::inv(this->Gtime_[_SPIN_], this->Gtime_[_SPIN_]);
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
		for (int i = 0; i < this->Ns_; i++) 
			for (int l = 0; l < this->M_; l++) 
				this->HSFields_(l, i) = this->ran_.random(0.0, 1.0) > 0.5 ? 1 : -1;
		break;
	case LOW_T:
		this->HSFields_.ones(this->M_, this->Ns_);
		break;
	}
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
	this->info_ = "Hubbard,";
	this->info_ += VEQV(M,		M_);
	this->info_ += "," + VEQV(M0, M0_);
	this->info_ += "," + VEQV(beta, beta_);
	this->info_ += "," + VEQV(U, U_);
	this->info_ += "," + VEQV(mu, mu_);

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
	for(auto _spin = 0; _spin < this->spinNumber_; _spin++)
		this->IExp_[_spin](_site, _t)	*=	_delta[_spin];
}

/*
* @brief After accepting spin change update the B matrix by multiplying it by the diagonal element (the delta)
* @param _site current lattice site for the update
*/
void Hubbard::updPropagatB(uint _site, uint _t)
{
	auto _delta = this->calDelta();
	for (int i = 0; i < this->Ns_; i++) 
	{
		this->B_[0][_t]	(i, _site)	*=	_delta[0];
		this->B_[1][_t]	(i, _site)	*=	_delta[1];

		this->iB_[0][_t](i, _site)	*=	_delta[1];
		this->iB_[1][_t](i, _site)	*=	_delta[0];
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
		this->udt_[_SPIN_]->D	=	this->G_[_SPIN_].row(_site).as_col();
		const auto gammaOverP	=	this->currentGamma_[_SPIN_] / p[_SPIN_];
		for (int _a = 0; _a < this->Ns_; _a++) {
			const auto _kron [[maybe_unused]]		=	(_a == _site) ? 1 : 0;
			const auto G_ai							=	this->G_[_SPIN_](_a, _site);
			for (int _b = 0; _b < this->Ns_; _b++)
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
	for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
		this->G_[_SPIN_]		=	(this->iB_[_SPIN_][_t - 1] * this->G_[_SPIN_]) * this->iB_[_SPIN_][_t - 1];
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
		const auto sectorToUpdate = modEUC<int>(static_cast<int>(_t / double(this->M0_)) - 1, this->p_);
		this->calPropagatBC(sectorToUpdate);
		this->calGreensFun(sectorToUpdate * this->M0_);
	}
	else
		this->updNextGreen(_t);
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
	auto _probaTuple			=	this->calProba(_site);
	this->proba_				=	1.0;
	for(auto& p : _probaTuple)
		this->proba_			*= p;
	this->proba_				*=	this->proba_ / (1.0 + this->proba_);
	const int _sign				=	(this->proba_ >= 0) ? 1 : -1;
	if (this->ran_.random<double>() <= _sign * this->proba_)
	{
		this->HSFields_(this->tau_, _site) *= -1;
		this->updPropagatB(_site, this->tau_);
		this->updEqlGreens(_site, _probaTuple);
	}
	return _sign;
}

/*
* @brief sweep space-time forward in time
*/
double Hubbard::sweepForward()
{
	this->configSign_	=	{};
	this->configSign_	=	1;
	for (int _tau = 0; _tau < this->M_; _tau++)
	{
		this->tau_		=	_tau;
		this->updGreenStep(_tau);
		configSign_		=	(this->sweepLattice() > 0) ? +this->configSign_ : -this->configSign_;
		configSigns_.push_back(configSign_);
	}
	return std::reduce(configSigns_.begin(), configSigns_.end()) / configSigns_.size();
}

// ##################################################### S I M U L A T I O N #######################################################

/*
* @brief Drive the system to equilibrium
* @param MCs number of Monte Carlo steps
* @param _quiet wanna talk?
*/
void Hubbard::equalibrate(uint MCs, bool _quiet)
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
	this->pBar_ = pBar(20, MCs);

	// sweep all
	for (int step = 0; step < MCs; step++) {
		this->sweepForward();
#ifdef DQMC_SAVE_CONF
		this->saveConfig("\t");
#endif
		if (!_quiet)
		{
			this->configSign_ > 0 ? this->posNum_++ : this->negNum_++;
			if (step % this->pBar_.percentageSteps == 0)
				this->pBar_.printWithTime(LOG_LVL2 + SSTR("PROGRESS RELAXATION"));
		}
	}
}

/*
* @brief Average the system physical measurements in the equilibrium
* @param corrTime time after which the new measurements are uncorrelated
* @param avNum number of averages to be taken
* @param quiet quiet?
*/
void Hubbard::averaging(uint MCs, uint corrTime, uint avNum, uint bootStraps, bool _quiet)
{
#pragma omp critical
	LOGINFO("Starting the averaging for " + this->info_, LOG_TYPES::TRACE, 2);
	auto start						=		std::chrono::high_resolution_clock::now();
	
	// initialize stuff
	// this->configSigns_			=		{};
	this->negNum_					=		0;
	this->posNum_					=		0;
	this->avs_->reset();
	this->pBar_						=		pBar(25, avNum);

	// check if this saved already
	for (int step = 1; step < avNum; step++) {
#ifdef DQMC_CAL_TIMES
	#ifdef DQMC_USE_HIRSH
		this->calGreensFunTHirsh();
	#else
		this->calGreensFunT();
	#endif
#endif
		for (auto _tau = 0; _tau < this->M_; _tau++) 
		{
			this->tau_				=		_tau;
#if !defined DQMC_CAL_TIMES || defined DQMC_CAL_TIMES && !defined DQMC_USE_HIRSH
			this->updGreenStep(this->tau_);
#endif
			// save the diagonal part of the Green's function on the fly
#ifdef DQMC_CAL_TIMES
			const uint _element		=		this->tau_ * this->Ns_;
			for (int _SPIN_ = 0; _SPIN_ < this->spinNumber_; _SPIN_++)
	#ifdef DQMC_USE_HIRSH
				algebra::setMFromSubM(this->G_[_SPIN_], this->Gtime_[_SPIN_], _element, _element, Ns_, Ns_, false);
	#else
				algebra::setSubMFromM(this->Gtime_[_SPIN_], this->G_[_SPIN_], _element, _element, Ns_, Ns_, false);
	#endif
#endif
			for (int _site = 0; _site < this->Ns_; _site++)
				this->avSingleStep(_site, this->configSign_);
		}
		this->configSigns_.push_back(this->configSign_);
		this->configSign_ > 0 ? this->posNum_++ : this->negNum_++;
#ifdef DQMC_CAL_TIMES
		if (step % DQMC_BUCKET_NUM == (DQMC_BUCKET_NUM - 1)) {
			LOGINFO("Saving " + STR(step / DQMC_BUCKET_NUM) + ". " + VEQ(DQMC_BUCKET_NUM) + ":" + TMS(start), LOG_TYPES::TRACE, 3);
			this->saveGreensT(step);
		}
#endif
		// kill correlations
		for (int _cor = 0; _cor < corrTime; _cor++)
			this->sweepForward();
		if(!_quiet && step % this->pBar_.percentageSteps == 0)
			this->pBar_.printWithTime(LOG_LVL2 + SSTR("PROGRESS AVERAGES"));
	}
	auto avSign = (this->posNum_ - this->negNum_) / (this->posNum_ + this->negNum_);
	this->avs_->normalize(avNum, this->M_ * this->lat_->get_Ns(), avSign);
}

// ####################################################### A V E R A G E S #########################################################

/*
* @brief A single step for calculating averages inside a simulation loop.
* @param _currI current HS spin
* @param _sign current sign to multiply the averages by
*/
void Hubbard::avSingleStep(int _currI, int _sign)
{
	// mz2
	INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mz2);
	// mx2
	INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Mx2);
	// n
	INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Occupation);
	// Ek
	INVOKE_SINGLE_PARTICLE_CAL(this->avs_, Ek);

	// save i'th point coordinates
	const auto xi			=	this->lat_->get_coordinates(_currI, Lattice::X);
	const auto yi			=	this->lat_->get_coordinates(_currI, Lattice::Y);
	const auto zi			=	this->lat_->get_coordinates(_currI, Lattice::Z);
	const auto ith_coord	=	std::make_tuple(xi, yi, zi);

	// -------------------------------- CORRELATIONS ----------------------------------------
	for (int _currJ = 0; _currJ < this->lat_->get_Ns(); _currJ++)
	{
		auto [x, y, z]		=	this->lat_->getSiteDifference(ith_coord, _currJ);
		auto [xx, yy, zz]	=	this->lat_->getSymPos(x, y, z);

		INVOKE_TWO_PARTICLE_CAL(this->avs_, Mz2, xx, yy, zz);
		INVOKE_TWO_PARTICLE_CAL(this->avs_, Occupation, xx, yy, zz);
#ifdef DQMC_CAL_TIMES
		this->avSingleStepUneq(xx, yy, zz, _currI, _currJ, _sign);
#endif
	}
}

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
		this->avs_->av_GTimeDiff_[_SPIN_][0](xx, yy)		+=		_s * this->G_[_SPIN_](_i, _j);
		this->avs_->av_GTimeDiff_[_SPIN_][0](xx, yy)		+=		this->G_[_SPIN_](_i, _j) * this->G_[_SPIN_](_i, _j);
#ifdef DQMC_CAL_TIMES_ALL
		for (int tim2 = 0; tim2 < this->M_; tim2++) {
#else
		for (int tim2 = 0; tim2 < this->tau_; tim2++) {
#endif
			auto tim	=	this->tau_ - tim2;
			if (tim == 0)
				continue;
			auto xk		=	_s;
			// handle antiperiodicity
#ifdef DQMC_CAL_TIMES_ALL
			if (tim < 0) {
				xk *= -1;
				tim += this->M_;
			}
#endif
			const uint col		=	tim2 * this->Ns_;
			const uint row		=	tau_ * this->Ns_;
			const double elem	=	xk * this->G_[_SPIN_](row + _i, col + _j);
			// save only the positive first half
			this->avs_->av_GTimeDiff_[_SPIN_][tim](xx, yy)	+=		elem;
			this->avs_->sd_GTimeDiff_[_SPIN_][tim](xx, yy)	+=		elem * elem;

		}
	}
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