import _ from 'lodash';
import { LogisticRegression } from './LogisticRegression';
import plot from 'node-remote-plot';
import mnist from 'mnist-data';

/** @returns {{features: number[][], labels: number[][]}} */
function loadData() {
  const mnistData = mnist.training(0, 60000);

  const features = mnistData.images.values.map(
    image => _.flatten(image));
  const encodedLabels = mnistData.labels.values.map(label => {
    const row = new Array(10).fill(0);
    row[label] = 1;
    return row;
  });
  return { features, labels: encodedLabels };
}

const { features, labels } = loadData();

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 40,
  batchSize: 100,
});

regression.train();
const testMnistData = mnist.training(0, 1000);
const testFeatures = testMnistData.images.values.map(
  image => _.flatten(image));
const testEncodedLabels = testMnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

console.log(regression.test(testFeatures, testEncodedLabels));

plot({
  x: regression.costHistory.reverse(),
});
