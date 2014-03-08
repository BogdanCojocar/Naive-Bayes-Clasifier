/* 
 * File:   NaiveBayesClasifier.h
 * Author: Bogdan Cojocar
 *
 */
#ifndef NAIVEBAYESCLASIFIER_H
#define	NAIVEBAYESCLASIFIER_H

#include <cassert>
#include <cmath>
#include <functional>
#include <numeric>
#include <utility>
#include <unordered_map>

#include "TrainingSet.h"

template <typename T, unsigned int N = 2 >
class NaiveBayesClasifier {
public:
    typedef std::array<std::pair<double, double>, N + 1 > ClasifierData;
    typedef std::unordered_map<T, ClasifierData, std::hash<double >> Clasifier;

    NaiveBayesClasifier();
    ~NaiveBayesClasifier();
    bool train(TrainingSet<T, N>& trainingSet);
    T clasify(const typename TrainingSet<T, N>::Feature& sample);
private:
    static const double PI;
    Clasifier clasifier;

    double accum(const ColectedData& feature, const double init,
            const std::function<double(const double, const double) >& op);
    double mean(const ColectedData& feature);
    double variance(const ColectedData& feature, const double m);
    double normalDistribution(const double x, const double m, const double v);
};

template <typename T, unsigned int N>
const double NaiveBayesClasifier<T, N>::PI = std::atan(1) * 4;

template <typename T, unsigned int N>
NaiveBayesClasifier<T, N>::NaiveBayesClasifier() {
}

template <typename T, unsigned int N>
NaiveBayesClasifier<T, N>::~NaiveBayesClasifier() {
}

template <typename T, unsigned int N>
bool NaiveBayesClasifier<T, N>::train(TrainingSet<T, N>& trainingSet) {
    if (!trainingSet.isEqual(N)) {
        return false;
    }
    if (trainingSet.size() == 0) {
        return false;
    }
    while (trainingSet.hasMoreTypes()) {
        const T type = trainingSet.getNextType();
        ClasifierData clasifierData;
        unsigned int numberOfFeatures = 0;
        while (trainingSet.hasMoreFeatures()) {
            auto feature = trainingSet.getNextFeature();
            double m = mean(feature);
            double v = variance(feature, m);
            clasifierData[numberOfFeatures++] = std::make_pair(m, v);
        }
        // On the last position in the array store the probability of the current type.
        clasifierData[numberOfFeatures] = std::make_pair(
                static_cast<double> (numberOfFeatures + 1) / trainingSet.size(), 0);
        clasifier.insert(std::make_pair(type, clasifierData));
    }
    return true;
}

template <typename T, unsigned int N>
T NaiveBayesClasifier<T, N>::clasify(const typename TrainingSet<T, N>::Feature& sample) {
    std::pair<T, double> bestValue = std::make_pair(T(), 0);
    for (auto& clasifierDataIt : clasifier) {
        ClasifierData clasifierData = clasifierDataIt.second;
        double posteriorPobability = 1;
        for (unsigned int i = 0; i < clasifierData.size(); ++i) {
            if (i == clasifierData.size() - 1) {
                posteriorPobability *= clasifierData[i].first;
            } else {
                posteriorPobability *= normalDistribution(sample[i], clasifierData[i].first,
                        clasifierData[i].second);
            }
        }
        if (posteriorPobability > bestValue.second) {
            bestValue.first = clasifierDataIt.first;
            bestValue.second = posteriorPobability;
        }
    }
    return bestValue.first;
}

template <typename T, unsigned int N>
double NaiveBayesClasifier<T, N>::accum(const ColectedData& feature, const double init,
        const std::function<double(const double, const double) >& op) {

    return std::accumulate(feature.begin(), feature.end(), init, op);
}

template <typename T, unsigned int N>
double NaiveBayesClasifier<T, N>::mean(const ColectedData& feature) {
    assert(feature.size() > 0);
    return accum(feature, 0, std::plus<double>()) / feature.size();
}

template <typename T, unsigned int N>
double NaiveBayesClasifier<T, N>::variance(const ColectedData& feature, const double m) {
    assert(feature.size() > 0);
    return accum(feature, 0,
            [&m](const double result, const double x) {
                return result + pow(x-m, 2);
            }) / feature.size();
}

template <typename T, unsigned int N>
double NaiveBayesClasifier<T, N>::normalDistribution(const double x, const double m, const double v) {
    return (1 / sqrt(2 * PI * v)) * exp(-0.5 * (pow(x - m, 2) / v));
}

#endif	/* NAIVEBAYESCLASIFIER_H */

