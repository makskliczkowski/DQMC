#include "include/dqmc2.h"

/*
* @brief Saves the unequal times Green's functions in a special form.
* @param _step current binning step in the simulation.
*/
void DQMC2::saveGreensT(uint _step)
{
	this->avs_->normalizeG();
	auto [x_num, y_num, z_num] = this->lat_->getNumElems();
	const std::string _signStr	= this->configSign_ == 1 ? "+" : "-";

#ifndef DQMC_SAVE_H5
	std::string information =	" Some version\n\n This is the file that contains real space Green's functions for different times.\n";
	information				+=	" The structure of each is we average over time differences and first row\n";
	information				+=	" before each Green matrix <cicj^+(t1, t2)> is an information about the difference\n";

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
	for (int _tau = 0; _tau < this->M_; _tau++) {
		this->tmpG_[0]	= ((this->avs_->av_GTimeDiff_[_UP_][_tau] + this->avs_->av_GTimeDiff_[_DN_][_tau]) / 2.0);
		this->tmpG_[0].save(arma::hdf5_name(this->dir_->unequalGDir + "G_t_" + STR(_step) + "_" + _signStr + "_" + this->dir_->randomSampleStr, 
			STR(_tau), arma::hdf5_opts::append));
	}
#endif
}

/*
* @brief Save the equal time Green's functions.
* @param _step step of the save
*/
void DQMC2::saveGreens(uint _step)
{
	const std::string _signStr	= this->configSign_ == 1 ? "+" : "-";
	this->tmpG_[0]				= (this->G_[_UP_] + this->G_[_DN_]) / 2.0;
	this->tmpG_[0].save(arma::hdf5_name(this->dir_->equalGDir + "G_" + STR(_step) + "_" + _signStr + "_" + this->dir_->randomSampleStr, "G(t)", arma::hdf5_opts::append));
}
