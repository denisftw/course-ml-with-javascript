import * as tf from '@tensorflow/tfjs-node';
import loadCSV from './load-csv';
import { LinearRegression } from './LinearRegression';
import plot from 'node-remote-plot';

let {features, labels, testFeatures, testLabels} =
 loadCSV('./cars.csv', {
     shuffle: true,
     splitTest: 50,
     dataColumns: ['horsepower', 'displacement', 'weight'],
     labelColumns: ['mpg'],
 });

const linearRegression = new LinearRegression(features,
  labels, {
    learningRate: 0.1,
    iterations: 100,
  });

linearRegression.train();
const r2 = linearRegression.test(testFeatures, testLabels);

plot({
  x: linearRegression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error',
});

console.log('R2 = ', r2);
const weights = linearRegression.weights.bufferSync();
// console.log(`M = ${weights.get(1,0)}, B = ${weights.get(0,0)}`);

