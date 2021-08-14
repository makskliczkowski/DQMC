#pragma once
#ifndef UI_H
#define UI_H

#include "common.h"
#include "../include/general_model.h"
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <functional>
#include <filesystem>
#include <omp.h>
#include <iostream>
/// <summary>
/// Here we will state all the already implemented definitions that will help us building the user interfrace
/// </summary>
namespace impDef{
	/// <summary>
	/// Different Monte Carlo algorithms that can be provided inside the classes (for simplicity in enum form)
	/// </summary>
	enum class algMC {
		metropolis,
		heat_bath,
		self_learning
	};
	/// <summary>
	/// Types of implemented lattice types
	/// </summary>
	enum class lattice_types {
		square
		//triangle,
		//hexagonal
	};
}

// Make a User interface class

class user_interface{
protected:
	int thread_number;																						// number of threads
	int boundary_conditions;																				// boundary conditions - 0 - PBC, 1 - OBC, 2 - ABC,... 
	std::string saving_dir;
public:
	std::unique_ptr<LatticeModel> model;																	// a unique pointer to the model used
	
	virtual void exit_with_help() = 0;
/* REAL PARSING */
	virtual void parseModel(int argc, v_1d<std::string> argv) = 0;											// the function to parse the command line
/* HELPING FUNCIONS */
	virtual void set_default() = 0;																			// set default parameters
/* NON-VIRTUALS */
	v_1d<std::string> parseInputFile(std::string filename);													// if the input is taken from file we need to make it look the same way as the command line does
};




namespace hubbard{
	// MAP AND PARSER FOR HUBBARD MODEL */
	enum class parsers {
		m,									// Monte Carlo steps
		d,									// dimension
		l,									// lattice type ( not implemented yet )
		t,									// hopping coefficients
		a,									// averages number
		c,									// correlation time
		M0,									// how many slices in Trotter times are used in QR or HF (Hirsh Fye)
		dt,									// Trotter time inverval
		dtn,								// Trotter time interval step
		dts,								// Trotter time intervals number
		lx,									// x-direction length
		lxs,								// x-step
		lxn,								// x-num
		ly,									// 
		lys,								//
		lyn,								//
		lz,									//
		lzs,								//
		lzn,								//
		b,									// inverse temperature beta
		bs,									// beta step
		bn,									// betas number
		u,									// Coloumb interaction strength
		us,									// Coloumb interaction step
		un,									// U number
		mu,									// chemical potential
		mus,								// chemical potential step
		mun,								// number of chemical potentials
		th,									// outer threads used
		ti,									// inner threads used
		q,									// quiet mode
		qr,									// use qr decomposition?
		config,								// save configuration?
		times,								// calculate non-equal time properties?
		self_learn,							// create files for network teaching for self learning
		self_learn_n						// number of configurations saved for training
	};
	std::unordered_map <std::string, hubbard::parsers> const table = {
		{"m",hubbard::parsers::m},
		{"d",hubbard::parsers::d},
		{"l",hubbard::parsers::l},
		{"t",hubbard::parsers::t},
		{"a",hubbard::parsers::a},
		{"c",hubbard::parsers::c},
		{"m0",hubbard::parsers::M0},
		{"dt",hubbard::parsers::dt},
		{"dtn",hubbard::parsers::dtn},
		{"dts",hubbard::parsers::dts},
		{"lx",hubbard::parsers::lx},
		{"lxs",hubbard::parsers::lxs},
		{"lxn",hubbard::parsers::lxn},
		{"ly",hubbard::parsers::ly},
		{"lys",hubbard::parsers::lys},
		{"lyn",hubbard::parsers::lyn},
		{"lz",hubbard::parsers::lz},
		{"lzs",hubbard::parsers::lzs},
		{"lzn",hubbard::parsers::lzn},
		{"b",hubbard::parsers::b},
		{"bs",hubbard::parsers::bs},
		{"bn",hubbard::parsers::bn},
		{"u",hubbard::parsers::u},
		{"us",hubbard::parsers::us},
		{"un",hubbard::parsers::un},
		{"mu",hubbard::parsers::mu},
		{"mus",hubbard::parsers::mus},
		{"mun",hubbard::parsers::mun},
		{"th",hubbard::parsers::th},
		{"ti",hubbard::parsers::ti},
		{"q",hubbard::parsers::q},
		{"qr",hubbard::parsers::qr },
		{"cg",hubbard::parsers::config},
		{"ct",hubbard::parsers::times},
		{"sf",hubbard::parsers::self_learn},
		{"sfn",hubbard::parsers::self_learn_n}
	};
	
	class ui: public user_interface{
	private:
		v_1d<double> t;																						// hopping coefficients
		int inner_threads, outer_threads;																	// thread parameters
		int sf, sfn;																						// self learning parameters
		bool quiet, save_conf, qr_dec, cal_times;															// bool flags
		int dim, lx, ly, lz, lx_step, ly_step, lz_step, lx_num, ly_num, lz_num;								// real space proprties
		double beta, beta_step, U, U_step, mu, mu_step, dtau, dtau_step;									// physical params
		int U_num, mu_num, dtau_num, beta_num;	
		int M_0, p, M, mcSteps, avsNum, corrTime;															// time properties
	public:
		// CONSTRUCTORS
		ui() = default;
		ui(int argc, char** argv);
		// PARSER FOR HELP
		void exit_with_help() override;
		// REAL PARSER
		void parseModel(int argc, v_1d<std::string> argv) override;											// the function to parse the command line
		// HELPERS
		void set_default() override;																		// set default parameters
	};

}


#endif // !UI_H
