#pragma once
#ifndef UI_H
#define UI_H

#include "../common.h"

#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <functional>
#include <omp.h>

// -------------------------------------------------------- Make a User interface class --------------------------------------------------------

class user_interface {
protected:
	int thread_number;																				 		// number of threads
	int boundary_conditions;																		 		// boundary conditions - 0 - PBC, 1 - OBC, 2 - ABC,...
	std::string saving_dir;

	std::string getCmdOption(const v_1d<std::string>& vec, std::string option) const;				 		// get the option from cmd input
	template <typename T>
	void set_option(T& value, const v_1d<std::string>& argv, std::string choosen_option, bool geq_0 = true);	// set an option

	template <typename T>
	void set_default_msg(T& value, std::string option, std::string message, \
		const std::unordered_map <std::string, std::string>& map) const;									// setting value to default and sending a message
	// std::unique_ptr<LatticeModel> model;															 			// a unique pointer to the model used

public:
	virtual ~user_interface() = default;

	virtual void make_simulation() = 0;

	virtual void exit_with_help() = 0;
	// ----------------------- REAL PARSING
	virtual void parseModel(int argc, const v_1d<std::string>& argv) = 0;									 // the function to parse the command line
	// ----------------------- HELPING FUNCIONS
	virtual void set_default() = 0;																	 		// set default parameters
	// ----------------------- NON-VIRTUALS
	v_1d<std::string> parseInputFile(std::string filename);													// if the input is taken from file we need to make it look the same way as the command line does
};


#endif // !UI_H


