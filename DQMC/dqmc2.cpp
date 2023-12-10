#include "./include/dqmc2.h"

// ######################################################### S A V E R S ###########################################################

/*
* @brief Saves the unequal times Green's functions in a special form.
* @param _step current binning step in the simulation.
*/
void DQMC2::saveGreensT(uint _step)
{
	std::unique_lock(this->fileMutex_);
	// normalize the Green's function
	this->avs_->normalizeG();
	// get number of elements
	// get the configuration sign
	const std::string _signStr	= this->configSign_ == 1 ? "+" : "-";
	//const auto _dim				= this->lat_->get_Dim();
#ifndef DQMC_SAVE_H5
	std::string information =	" Some version\n\n This is the file that contains real space Green's functions for different times.\n";
	information				+=	" The structure of each is we average over time differences and first row\n";
	information				+=	" before each Green matrix <cicj^+(t1, t2)> is an information about the difference\n";
	auto [x_num, y_num, z_num]	= this->lat_->getNumElems();

	std::ofstream _file;
	openFile(_file, this->dir_->unequalGDir +  "G_t_" + STR(_step) + "_" + _signStr + "_" + this->dir_->randomSampleStr + ".dat");

	_file					<<	"Different time real-space Green's functions\n" << information;
	std::string misc		=	"IDK";
	this->GtimeInf_ = 
	"n =\t"			+	STR(lat_->get_Lx())		+	"\n"	+
	"l =\t"			+	STR(this->M_)			+	"\n"	+
	"tausk =\t"		+	STR(this->p_)			+	"\n"	+
	"doall =\t"		+	misc					+	"\n"	+
	"denswp =\t"	+	STR(DQMC_BUCKET_NUM)	+	"\n"	+
	"histn =\t"		+	misc					+	"\n"	+
	"iran =\t"		+	misc					+	"\n"	+
	"t  =\t"		+	misc					+	"\n"	+
	"mu =\t"		+	misc					+	"\n"	+
	"delmu =\t"		+	misc					+	"\n"	+
	"bpar  =\t"		+	misc					+	"\n"	+
	"dtau = \t"		+	STRP(this->tau_,5)		+	"\n"	+
	"warms  =\t"	+	STR(currentWarmups_)	+	"\n"	+
	"sweeps =\t"	+	STR(currentAverages_)	+	"\n"	+
	"u =\t"			+	misc					+	"\n"	+
	"nwrap =\t"		+	misc					+	"\n"	+
	"difflim =\t"	+	misc					+	"\n"	+
	"errrat =\t"	+	misc					+	"\n"	+
	"doauto = \t0"	+							+	"\n"	+
	"orthlen =\t"	+	misc					+	"\n"	+
	"eorth =\t"		+	misc					+	"\n"	+
	"dopair =\t"	+	misc					+	"\n"	+
	"numpair =\t"	+	misc					+	"\n"	+
	"lambda=\t"		+	misc					+	"\n"	+
	"start = \t0"								+	"\n";
	file_			<<	this->GtimeInf_;
	file_			<<	str_p(this->configSigns_);
	const auto width = 12;
	printSeparated(file_, '\t', width, true, "G(nx,ny,ti)");
	for (int nx = 0; nx < x_num; nx++)
		for (int ny = 0; ny < y_num; ny++)
		{
			auto [x, y, z] = this->lat_->getSymPosInv(nx, ny, 0);
			printSeparated(file_, '\t', 6, true, VEQ(x), VEQ(y));
			for (int tau1 = 0; tau1 < this->M_; tau1++)
			{
				printSeparated(file_, '\t', 4, false, tau1);
				//printSeparated(file_, '\t', width + 5, false, STRP(this->avs->g_up_diffs[tau1](nx, ny), width));
				printSeparated(file_, '\t', 5, false, "+-");
				//printSeparated(file_, '\t', width + 5, true, STRP(this->avs->sd_g_up_diffs[tau1](nx, ny), width));
			}
		}
#else
	// go through imaginary times
	WriteLock lock(this->Mutex);
	std::string _filename		= this->dir_->unequalGDir + "G_t_" + STR(_step) + "_" + _signStr + "_" + this->dir_->randomSampleStr + ".h5";
	for (int _tau = 0; _tau < this->M_; _tau++) 
	{
		this->tmpG_[0]	= ((this->avs_->av_GTimeDiff_[_UP_][_tau] + this->avs_->av_GTimeDiff_[_DN_][_tau]) / 2.0).slice();
		this->tmpG_[1]	= ((this->avs_->sd_GTimeDiff_[_UP_][_tau] + this->avs_->sd_GTimeDiff_[_DN_][_tau]) / 2.0).slice();

		// append if necessary
		if (_tau != 0) 
			this->tmpG_[0].save(arma::hdf5_name(_filename, STR(_tau), arma::hdf5_opts::append));
		else
			this->tmpG_[0].save(arma::hdf5_name(_filename, STR(_tau)));
		
		// SPINS
		this->avs_->av_GTimeDiff_[_UP_][_tau].slice().save(arma::hdf5_name(_filename, "UP_" + STR(_tau), arma::hdf5_opts::append));
		this->avs_->av_GTimeDiff_[_DN_][_tau].slice().save(arma::hdf5_name(_filename, "DN_" + STR(_tau), arma::hdf5_opts::append));
		
		// STD
		this->tmpG_[1].save(arma::hdf5_name(_filename, "SD_" + STR(_tau), arma::hdf5_opts::append));
	}
#endif
}

// ########################################################## G R E E N S ###########################################################

/*
* @brief Save the equal time Green's functions.
* @param _step step of the save
*/
void DQMC2::saveGreens(uint _t)
{
	std::unique_lock(this->fileMutex_);
	LOGINFO("Saving Green's at " + VEQ(_t), LOG_TYPES::TRACE, 4);
	const std::string _signStr	= this->configSign_ == 1 ? "+" : "-";
#ifndef DQMC_USE_HIRSH
	this->tmpG_[0]				= (this->G_[_UP_] + this->G_[_DN_]) / 2.0;
	this->tmpG_[0].save	(arma::hdf5_name(this->dir_->equalGDir + "G_" + _signStr + "_" 
		+ this->dir_->randomSampleStr, "G("	  + STR(_t) + ")", arma::hdf5_opts::append));
	this->G_[_UP_].save	(arma::hdf5_name(this->dir_->equalGDir + "G_" + _signStr + "_"
		+ this->dir_->randomSampleStr, "G_UP(" + STR(_t) + ")", arma::hdf5_opts::append));
	this->G_[_DN_].save	(arma::hdf5_name(this->dir_->equalGDir + "G_" + _signStr + "_" 
		+ this->dir_->randomSampleStr, "G_DN(" + STR(_t) + ")", arma::hdf5_opts::append));
#else
	this->tmpG_[0]				= (	SUBM(this->G_[_UP_], _t * transformSize_,
														 _t * transformSize_, 
															  transformSize_,
														      transformSize_) + 
									SUBM(this->G_[_DN_], _t * transformSize_,
														 _t * transformSize_, 
															  transformSize_,
															  transformSize_)) / 2.0;
	this->tmpG_[0].save	(arma::hdf5_name(this->dir_->equalGDir + "G_" + _signStr + "_" 
		+ this->dir_->randomSampleStr, "G("	  + STR(_t) + ")", arma::hdf5_opts::append));
	DMAT(SUBM(this->G_[_UP_], _t * transformSize_,
							 _t * transformSize_,
							 transformSize_,
							 transformSize_)).save(arma::hdf5_name(this->dir_->equalGDir + "G_" + _signStr + "_"
											 + this->dir_->randomSampleStr, "G_UP(" + STR(_t) + ")",
											 arma::hdf5_opts::append));
	DMAT(SUBM(this->G_[_DN_], _t * transformSize_,
							 _t * transformSize_,
							 transformSize_,
							 transformSize_)).save(arma::hdf5_name(this->dir_->equalGDir + "G_" + _signStr + "_"
											 + this->dir_->randomSampleStr, "G_DN(" + STR(_t) + ")",
											 arma::hdf5_opts::append));
#endif
}

/*
* @brief Saves the averages after finishing the simulation
*/
void DQMC2::saveAverages(uint _step)
{
	LOGINFO("Saving single site averages at " + VEQ(_step), LOG_TYPES::TRACE, 4);
	std::ofstream fileLog, fileSigns;
	auto _logN = this->dir_->mainDir + "HubbardLog.csv";
	auto _logS = this->dir_->mainDir + "HubbardSign.csv";
	auto _date = prettyTime();

	// check if the file already exists - if not create header
	if (!fs::exists(_logN))
	{
		openFile(fileLog, _logN, std::ios::in | std::ios::app);
		printSeparatedP(fileLog, ';', 70, true, 5, 
				"RANDOM",
				"LATTICE",
				"MODEL",
				"<s>",
				"<n>",
				"dn",
				"Ek_0",
				"dEk_0",
				"mz2",
				"dmz2",
				"mx2",
				"dmx2",
				"STEP",
				"DATE");
		fileLog.close();
	}

	// open log file
	openFile(fileLog, _logN, std::ios::in | std::ios::app);
	openFile(fileSigns, _logS, std::ios::in | std::ios::app);

	printSeparatedP(fileLog, ';', 70, true, 5,
					this->dir_->randomSampleStr,
					this->lat_->get_info(), 
					this->getInfo(),
					this->avs_->av_sign_, 
					this->avs_->av_Occupation_0, 
					this->avs_->sd_Occupation_0,
					this->avs_->av_Ek_0, 
					this->avs_->sd_Ek_0,
					this->avs_->av_Mz2_0, 
					this->avs_->sd_Mz2_0,
					this->avs_->av_Mx2_0, 
					this->avs_->sd_Mx2_0,
					_step,
					_date);

	printSeparatedP(fileSigns	, '\t', 25, true, 5, 
		this->getInfo(), this->avs_->av_Occupation_0, this->avs_->av_sign_, _step, _date);
	fileLog.close();
	fileSigns.close();

	// save the Green's functions
	for (int _t = 0; _t < this->M_; ++_t)
	{
#ifndef DQMC_USE_HIRSH
		this->updGreenStep(_t);
		this->saveGreens(_step);
#else
		this->saveGreens(_step);
#endif
	}
	this->saveCorrelations(_step);
}

/*
* @brief Save the correlations
*/
void DQMC2::saveCorrelations(uint _step)
{
	LOGINFO("Saving correlations at " + VEQ(_step), LOG_TYPES::FINISH, 4);
	std::string _filename		= this->dir_->equalCorrDir + "corr_" + STR(_step) + "_" + this->dir_->randomSampleStr;
	std::string _filenameNoStep = this->dir_->equalCorrDir + "corr_" + this->dir_->randomSampleStr;

	auto [x_num, y_num, z_num]	= this->lat_->getNumElems();
	MAT<double> _outMat(x_num * y_num * z_num, 3 + 2);

	auto _iterator = 0;
	for (int x = 0; x < x_num; x++)
	{
		for (int y = 0; y < y_num; y++)
		{
			for (int z = 0; z < z_num; z++) {
				auto [xx, yy, zz] = this->lat_->getSymPosInv(x, y, z);
				//printSeparated	(file, ',', 2 * prec, false	, xx, yy, zz);
				//printSeparatedP	(file, ',', 2 * prec, true	, prec	, avs_->avC_Mz2_0[x][y][z], avs_->avC_Occupation_0[x][y][z]);

				_outMat(_iterator, 0) = xx;
				_outMat(_iterator, 1) = yy;
				_outMat(_iterator, 2) = zz;
				_outMat(_iterator, 3) = avs_->avC_Mz2_0[x][y][z];
				_outMat(_iterator, 4) = avs_->avC_Occupation_0[x][y][z];
				//if (times) {
					//	for (int i = 0; i < M; i++) {
							//fileP_time << x << "\t" << y << "\t" << z << "\t" << i << "\t" << (avs.av_M2z_corr_uneqTime[x_pos][y_pos][z_pos][i]) << "\t" << (avs.av_Charge2_corr_uneqTime[x_pos][y_pos][z_pos][i]) << endl;
							//fileP_time << x << "\t" << y << "\t" << z << "\t" << i << "\t" << this->gree << "\t" << (avs.av_Charge2_corr_uneqTime[x_pos][y_pos][z_pos][i]) << endl;
					//	}
				_iterator++;
			}
		}
	}
	std::unique_lock(this->fileMutex_);

	_outMat.save(arma::hdf5_name(_filenameNoStep + ".h5", STR(_step)));
	_outMat.save(_filename + ".dat", arma::arma_ascii);
}