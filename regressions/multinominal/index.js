import loadCSV from '../load-csv';
import { LogisticRegression } from './LogisticRegression';
import plot from 'node-remote-plot';

let { features, labels, testFeatures, testLabels } =
 loadCSV('./data/cars.csv', {
     shuffle: true,
     splitTest: 50,
     dataColumns: ['horsepower', 'displacement', 'weight'],
     labelColumns: ['passedemissions'],
     converters: {
      passedemissions: (value) => value === 'TRUE' ? 1 : 0,
     }
 });

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.6,
});

regression.train();
console.log(regression.test(testFeatures, testLabels));

plot({
  x: regression.costHistory.reverse(),

})
