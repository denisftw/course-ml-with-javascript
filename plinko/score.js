const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 100;
  const k = 10;
  _.range(0, 3).forEach(featureIndex => {
    const data = _.map(outputs, row => [row[featureIndex], _.last(row)])
    const [testSet, trainingSet] = splitDataset(
      minMax(data, 1), testSetSize);
    const accuracy = _.chain(testSet)
      .filter(testPoint => {
        return knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint);
      })
      .size()
      .divide(testSetSize)
      .value();
    console.log('For feature index of ', featureIndex, ' accuracy: ', accuracy);
  })
}

function knn(trainingData, predictionPoint, k) {
  const normalized = _.map(trainingData, row => {
    return [
      distance(_.initial(row), predictionPoint), 
      _.last(row)
    ];
  });
  const sorted = _.sortBy(normalized, row => row[0]);
  const topK = _.slice(sorted, 0, k);
  const counted = _.countBy(topK, row => row[1]);
  const pairs = _.toPairs(counted);
  const sorted2 = _.sortBy(pairs, r => -r[1]);
  const bestStr = _.first(sorted2[0]);
  const predictedBucket = _.parseInt(bestStr);
  return predictedBucket;
}

function distance(pointA, pointB) {
  return _.chain(pointA)
    .zip(pointB)
    .map(([a, b]) => {
        return (a - b)**2;
    })
    .sum()
    .value() ** 0.5;
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);
  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}

/**
 * @param {number[]} data 
 * @param {number} featureCount 
 */
function minMax(data, featureCount) {
  const clonedData = _.cloneDeep(data);

  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map(row => row[i]);
    const [min, max] = [_.min(column), _.max(column)];
    const divider = max - min;
    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / divider;
    }
  }
  return clonedData;
}