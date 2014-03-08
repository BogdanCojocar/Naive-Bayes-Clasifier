/* 
 * File:   TrainingSet.h
 * Author: Bogdan Cojocar
 *
 */

#ifndef TRAININGSET_H
#define	TRAININGSET_H

#include <array>
#include <cassert>
#include <map>
#include <vector>

typedef std::vector<double> ColectedData;

template <typename T, unsigned int N = 2 >
class TrainingSet {
public:
    typedef std::array<double, N> Feature;
    typedef std::multimap<T, Feature> TrainingDataSet;

    TrainingSet();
    ~TrainingSet();
    void add(const T& type, const Feature& feature);
    bool hasMoreTypes() const;
    const T getNextType();
    bool hasMoreFeatures();
    ColectedData getNextFeature();
    bool isEqual(const unsigned int numOfFeatures) const;
    const unsigned int size() const;
public:
    unsigned int featureIndex;
    TrainingDataSet trainingSet;
    typename TrainingDataSet::iterator trainingSet_type_it;
    typename TrainingDataSet::iterator trainingSet_feature_it;
};

template <typename T, unsigned int N>
TrainingSet<T, N>::TrainingSet()
: featureIndex(0) {
}

template <typename T, unsigned int N>
TrainingSet<T, N>::~TrainingSet() {
}

template <typename T, unsigned int N>
void TrainingSet<T, N>::add(const T& type, const Feature& feature) {
    assert(N == feature.size());
    trainingSet.insert(std::make_pair<const T&, const Feature&>(type, feature));
    trainingSet_type_it = trainingSet.begin();
    trainingSet_feature_it = trainingSet.begin();
}

template <typename T, unsigned int N>
bool TrainingSet<T, N>::hasMoreTypes() const {
    return trainingSet_type_it != trainingSet.end();
}

template <typename T, unsigned int N>
const T TrainingSet<T, N>::getNextType() {
    auto next_type = trainingSet_type_it->first;
    do {
        trainingSet_type_it++;
    } while (trainingSet_type_it->first == next_type);
    return next_type;
}

template <typename T, unsigned int N>
bool TrainingSet<T, N>::hasMoreFeatures() {
    if (featureIndex < N) {
        return true;
    } else {
        featureIndex = 0;
        trainingSet_feature_it = trainingSet_type_it;
        return false;
    }
}

template <typename T, unsigned int N>
ColectedData TrainingSet<T, N>::getNextFeature() {
    ColectedData featureVec;
    if (hasMoreFeatures()) {
        auto next_feature_it = trainingSet_feature_it;
        while (next_feature_it->first != trainingSet_type_it->first) {
            featureVec.push_back(next_feature_it->second[featureIndex]);
            next_feature_it++;
        }
        featureIndex++;
    }
    return featureVec;
}

template <typename T, unsigned int N>
bool TrainingSet<T, N>::isEqual(const unsigned int numOfFeatures) const {
    return N == numOfFeatures;
}

template <typename T, unsigned int N>
const unsigned int TrainingSet<T, N>::size() const {
    return trainingSet.size();
}

#endif	/* TRAININGSET_H */
