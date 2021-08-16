#pragma once
#ifndef UI_H
#define UI_H

#include "common.h"
#include "../include/general_model.h"
#include "../include/plog/Log.h"
#include "../include/plog/Initializers/RollingFileInitializer.h"
#include "../include/lattices.h"
#include "../include/hubbard.h"
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
	int thread_number;																				 		// number of threads
	int boundary_conditions;																		 		// boundary conditions - 0 - PBC, 1 - OBC, 2 - ABC,... 
	std::string saving_dir;

	std::string getCmdOption(const v_1d<std::string>& vec, std::string option) const;				 		// get the option from cmd input
	template <typename T>
	void set_option(T& value,const v_1d<std::string>& argv, std::string choosen_option, bool geq_0 = true);	// set an option
		
	template <typename T>
	void set_default_msg(T& value,std::string option, std::string message,\
		const std::unordered_map <std::string, std::string>& map) const;									// setting value to default and sending a message							 		
	// std::unique_ptr<LatticeModel> model;															 		// a unique pointer to the model used


public:
	virtual void make_simulation() = 0;

	virtual void exit_with_help() = 0;
/* REAL PARSING */
	virtual void parseModel(int argc, const v_1d<std::string>& argv) = 0;									 // the function to parse the command line
/* HELPING FUNCIONS */
	virtual void set_default() = 0;																	 		// set default parameters
/* NON-VIRTUALS */
	v_1d<std::string> parseInputFile(std::string filename);													// if the input is taken from file we need to make it look the same way as the command line does


};




namespace hubbard{
	// MAP OF DEFAULTS FOR HUBBARD
	std::unordered_map <std::string, std::string> const default_params = {
		{"m","300"},
		{"d","2"},
		{"l","0"},
		{"t","1"},
		{"a","50"},
		{"c","1"},
		{"m0","10"},
		{"dt","0.1"},
		{"dtn","1"},
		{"dts","0"},
		{"lx","4"},
		{"lxs","0"},
		{"lxn","1"},
		{"ly","4"},
		{"lys","0"},
		{"lyn","1"},
		{"lz","1"},
		{"lzs","0"},
		{"lzn","1"},
		{"b","6"},
		{"bs","0"},
		{"bn","1"},
		{"u","2"},
		{"us","0"},
		{"un","1"},
		{"mu","0"},
		{"mus","0"},
		{"mun","1"},
		{"th","1"},
		{"ti","1"},
		{"q","0"},
		{"qr","1" },
		{"cg","0"},
		{"ct","0"},
		{"sf","0"},
		{"sfn","1"}
	};
	
	class ui: public user_interface{
	private:
		v_1d<double> t;																						// hopping coefficients
		int lattice_type; 																					// for non_numeric data
		double t_fill;
		int inner_threads, outer_threads;																	// thread parameters
		int sf, sfn;																						// self learning parameters
		bool quiet, save_conf, qr_dec, cal_times;															// bool flags
		int dim, lx, ly, lz, lx_step, ly_step, lz_step, lx_num, ly_num, lz_num;								// real space proprties
		double beta, beta_step, U, U_step, mu, mu_step, dtau, dtau_step;									// physical params
		int U_num, mu_num, dtau_num, beta_num;	
		int M_0, p, M, mcSteps, avsNum, corrTime;															// time properties
	
		// HELPER FUNCTIONS
		// void create_directories(std::string dir, int Lx, int Ly, int Lz, double U, double beta, double dtau, double mu);
		void collectAvs(double U, int M_0, double dtau, int p, double beta, double mu, int Lx, int Ly, int Lz);
		void collectFouriers(std::string name_times, std::string name, int Lx, int Ly, int Lz, int M, std::shared_ptr<averages_par> avs);
	public:
		// CONSTRUCTORS
		ui() = default;
		ui(int argc, char** argv);
		// PARSER FOR HELP
		void exit_with_help() override;
		// REAL PARSER
		void parseModel(int argc, const v_1d<std::string>& argv) override;									// the function to parse the command line
		// HELPERS
		void set_default() override;																		// set default parameters
		// SIMULATION
		void make_simulation() override;
	
	};

}


#endif // !UI_H


