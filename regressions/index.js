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
    learningRate: 0.0001,
    iterations: 100,
  });

linearRegression.train();

const weights = linearRegression.weights.bufferSync();
console.log(`M = ${weights.get(1,0)}, B = ${weights.get(0,0)}`)
