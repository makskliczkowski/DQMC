#pragma once
#include "common.h"


// -------------------------------------------------------------- binning --------------------------------------------------------------
/* 
* bin the data to calculate the correlation time approximation 
* @param seriesData data to be binned
* @param nBins number of bins to be used
* @param binSize the size of a given single bin
* @returns average in each bin
*/
//template<typename T>
//inline v_1d<T> binning(const v_1d<T>& seriesData, size_t nBins, size_t binSize){
//    v_1d<T> bins(nBins, 0);
//    if(binSize * nBins > seriesData.size()) throw "Cannot create bins of insufficient elements";
//    for(int i = 0; i < bins.size(); i++)
//        bins[i] = std::accumulate(seriesData.begin() + binSize * i, seriesData.end() + binSize * (i+1) - 1,  decltype(vector)::value_type(0))/binSize;
//    return bins;
//}
//
///* 
//* bin the data to calculate the correlation time approximation 
//* @param seriesData data to be binned
//* @param bins the vector to save the average into
//* @param binSize the size of a given single bin
//*/
//template<typename T>
//inline v_1d<T> binning(const v_1d<T>& seriesData, v_1d<T>& bins, size_t binSize){
//    if(binSize * bins.size() > seriesData.size()) throw "Cannot create bins of insufficient elements";
//    for(int i = 0; i < bins.size(); i++)
//        bins[i] = std::accumulate(seriesData.begin() + binSize * i, seriesData.end() + binSize * (i+1) - 1,  decltype(vector)::value_type(0))/binSize;
//}

/* 
* bin the data to calculate the correlation time approximation 
* @param seriesData data to be binned
* @param bins the vector to save the average into
* @param binSize the size of a given single bin
*/
template<typename T>
inline v_1d<T> binning(const arma::Col<T>& seriesData, arma::Col<T>& bins, size_t binSize){
    if(binSize * bins.size() > seriesData.size()) throw "Cannot create bins of insufficient elements";
    for(int i = 0; i < bins.size(); i++)
        bins(i) = arma::mean(seriesData.subvec(binSize * i, binSize*(i+1) - 1));
}


// -------------------------------------------------------------- statistical meassures --------------------------------------------------------------

/*
* Approximate the correlation error
* @param bins the bin average of the data
* @returns approximation of a statistical correlation error
*/
template<typename T>
inline double correlationError(const arma::Col<T>& bins){
    return arma::real(sqrt(arma::var(bins)/bins.size()));
}

/*
* Calculate the variance of a given value
*/
template <typename T>
inline T variance(T value, T average, int norm) {
	return std::sqrt((value / norm - average * average) / norm);
}

