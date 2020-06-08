import loadCSV from '../load-csv';
import _ from 'lodash';
import { LogisticRegression } from './LogisticRegression';
import plot from 'node-remote-plot';

let { features, labels, testFeatures, testLabels } =
  loadCSV('./data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    converters: {
      mpg: (value) => {
        const mpg = parseFloat(value);
        if (mpg < 15) {
          return [1,0,0];
        } else if (mpg < 30) {
          return [0,1,0];
        } else {
          return [0,0,1];
        }
      },
    }
  });

labels = _.flatten(labels);


const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
});


regression.train();
console.log(regression.test(
  testFeatures, _.flatten(testLabels)));
