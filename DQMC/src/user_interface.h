#pragma once
#ifndef UI_H
#define UI_H

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




#endif // !UI_H
