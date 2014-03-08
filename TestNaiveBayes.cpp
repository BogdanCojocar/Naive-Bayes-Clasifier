/* 
 * File:   TestNaiveBayes.cpp
 * Author: Bogdan Cojocar
 *
 */

#include <gtest/gtest.h>
#include "NaiveBayesClasifier.h"

TEST(TrainingSetTest, test1) { 
    enum Bike {
        MOUNTAIN_BIKE,
        FAST_BIKE
    };
    TrainingSet<Bike, 2> trainingSet;
    std::array<double, 2> t1 = {1, 2};
    std::array<double, 2> t2 = {3, 4};
    std::array<double, 2> t3 = {5, 6};
    std::array<double, 2> t4 = {7, 8};
    trainingSet.add(MOUNTAIN_BIKE, t1);
    trainingSet.add(MOUNTAIN_BIKE, t2);
    trainingSet.add(FAST_BIKE, t3);
    trainingSet.add(FAST_BIKE, t4);
    
    std::vector<double> expected1 = {1, 3};
    std::vector<double> expected2 = {2, 4};
    
    EXPECT_TRUE(trainingSet.hasMoreTypes());
    EXPECT_EQ(trainingSet.getNextType(), MOUNTAIN_BIKE);
    EXPECT_TRUE(trainingSet.getNextFeature() == expected1);
    EXPECT_TRUE(trainingSet.getNextFeature() == expected2);
    EXPECT_TRUE(trainingSet.hasMoreTypes());
    EXPECT_EQ(trainingSet.getNextType(), FAST_BIKE);
    EXPECT_FALSE(trainingSet.hasMoreFeatures());
}

TEST(NaiveBayesTest, test1) {
  enum Type {
    MALE,
    FEMALE 
  };
  TrainingSet<Type, 3> trainingSet;
  
  // feature contains height, weight and foot size
  std::array<double, 3> m1 = {6, 180, 12};
  std::array<double, 3> m2 = {5.92, 190, 11};
  std::array<double, 3> m3 = {5.58, 170, 12};
  std::array<double, 3> m4 = {5.92, 165, 10};
  trainingSet.add(MALE, m1);
  trainingSet.add(MALE, m2);
  trainingSet.add(MALE, m3);
  trainingSet.add(MALE, m4);
  
  std::array<double, 3> f1 = {5, 100, 6};
  std::array<double, 3> f2 = {5.5, 150, 8};
  std::array<double, 3> f3 = {5.42, 130, 7};
  std::array<double, 3> f4 = {5.75, 150, 9};
  trainingSet.add(FEMALE, f1);
  trainingSet.add(FEMALE, f2);
  trainingSet.add(FEMALE, f3);
  trainingSet.add(FEMALE, f4);
  
  std::array<double, 3> sample1 = {5, 120, 7.7};
  std::array<double, 3> sample2 = {6.3, 172, 11.5};
  
  NaiveBayesClasifier<Type, 3> clasifier;
  EXPECT_TRUE(clasifier.train(trainingSet));
  
  EXPECT_EQ(clasifier.clasify(sample1), FEMALE);
  EXPECT_EQ(clasifier.clasify(sample2), MALE);
}

TEST(NaiveBayesTest, test2) {
   enum Bird {
    SMALL,
    MIDDLE,
    BIG
  }; 
  TrainingSet<Bird, 2> trainingSet;
  
  // weight, height
  std::array<double, 2> s1 = {2, 10};
  std::array<double, 2> s2 = {2.3, 12};
  trainingSet.add(SMALL, s1);
  trainingSet.add(SMALL, s2);
  
  std::array<double, 2> m1 = {4, 15};
  std::array<double, 2> m2 = {4.7, 17.2};
  trainingSet.add(MIDDLE, m1);
  trainingSet.add(MIDDLE, m2);
  
  std::array<double, 2> b1 = {7, 23};
  std::array<double, 2> b2 = {8.5, 22.5};
  trainingSet.add(BIG, b1);
  trainingSet.add(BIG, b2);
  
  NaiveBayesClasifier<Bird, 2> clasifier;
  EXPECT_TRUE(clasifier.train(trainingSet));
  
  std::array<double, 2> sample1 = {1.5, 9};
  std::array<double, 2> sample2 = {4.9, 16};
  std::array<double, 2> sample3 = {9, 20};
  
  EXPECT_EQ(clasifier.clasify(sample1), SMALL);
  EXPECT_EQ(clasifier.clasify(sample2), MIDDLE);
  EXPECT_EQ(clasifier.clasify(sample3), BIG);
}

