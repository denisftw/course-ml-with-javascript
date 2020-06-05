import * as tf from '@tensorflow/tfjs-node';
import loadCSV from './load-csv';
import { LinearRegression } from './LinearRegression';

let {features, labels, testFeatures, testLabels} =
 loadCSV('./cars.csv', {
     shuffle: true,
     splitTest: 50,
     dataColumns: ['horsepower'],
     labelColumns: ['mpg'],
 });

const linearRegression = new LinearRegression(features,
  labels, {
    learningRate: 0.0000001,
    iterations: 100000,
  });

linearRegression.train();

console.log(`M = ${linearRegression.m}, B = ${linearRegression.b}`)
