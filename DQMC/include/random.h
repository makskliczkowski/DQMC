#pragma once
#ifndef RANDOM_H
#define RANDOM_H

#include "../src/xoshiro_pp.h"
#include <random>
#include <ctime>


/// <summary>
/// Random number generator class
/// </summary>
class randomGen{
private:
	XoshiroCpp::Xoshiro256PlusPlus engine;
public:
	explicit randomGen(const std::uint64_t seed = static_cast<uint64_t>(std::time(nullptr))){
		this->engine = XoshiroCpp::Xoshiro256PlusPlus(this->SeedInit(seed));
	}
	uint64_t SeedInit(uint64_t n) const
    {
    	std::vector<uint64_t> s(16,0);
		for (int i = 0; i < 16; i++)
		{
			n ^= n >> 12;   // a
			n ^= n << 25;   // b
			n ^= n >> 27;   // c
			s[i] = n * 2685821657736338717LL;                               // 2685821657736338717 = 72821711 * 36882155347, from Pierre L'Ecuyer's paper
		}
        return std::accumulate(s.begin(),s.end(), 0.0);
	}

	/* WRAPPERS ON RANDOM FUNCTIONS */
	double randomReal_uni(double _min=0, double _max=1){
		return std::uniform_real_distribution<double>(_min,_max)(engine);
	}
	uint64_t randomInt_uni(int _min, int _max){
		return std::uniform_int_distribution<uint64_t>(_min,_max)(engine);
	}
};



#endif // !RANDOM_H
