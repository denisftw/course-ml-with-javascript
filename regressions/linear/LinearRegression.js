import * as tf from '@tensorflow/tfjs-node';
import _ from 'lodash';

export class LinearRegression {
  /**
   *
   * @param {number[][]} features
   * @param {number[][]} labels
   * @param {{learningRate: number, iterations: number, batchSize: number}} options
   */
  constructor(featuresA, labels, options) {
    /** @type {tf.Tensor} */
    this.labels = tf.tensor(labels);
    this.features = this.processFeatures(featuresA);
    this.mseHistory = [];

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 1000,
    }, options);

    /** @type {tf.Tensor} */
    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  /**
   *
   * @param {tf.Tensor} features
   * @param {tf.Tensor} labels
   */
  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);
    const slopes = features.transpose()
      .matMul(differences)
      .div(features.shape[0]);
    this.weights = this.weights.sub(
      slopes.mul(this.options.learningRate));
  }

  train() {
    const { batchSize } = this.options;
    const batchQuantity = Math.floor(
      this.features.shape[0] / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        const featureSlice = this.features.
          slice([startIndex, 0], [batchSize, -1]);
        const labelSlice = this.labels.
          slice([startIndex, 0], [batchSize, -1]);
        this.gradientDescent(featureSlice, labelSlice);
      }
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  /**
   *
   * @param {number[][]} observations
   * @returns {tf.Tensor}
   */
  predict(observations) {
    return this.processFeatures(observations)
      .matMul(this.weights);
  }

  /** @returns {number} */
  test(testFeaturesA, testLabelsA) {
    /** @type {tf.Tensor} */
    const testLabels = tf.tensor(testLabelsA);
    const testFeatures = this.processFeatures(testFeaturesA);
    const predictions = testFeatures.matMul(this.weights);

    const res = testLabels.sub(predictions).pow(2).sum().bufferSync().get();
    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().bufferSync().get();
    return 1 - res / tot;
  }

  /**
   * @param {number[][]} sourceFeatures
   * @returns {tf.Tensor} */
  processFeatures(sourceFeatures) {
    /** @type {tf.Tensor} */
    let features = tf.tensor(sourceFeatures);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  /** @param {tf.Tensor} features */
  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);
    this.mean = mean;
    this.variance = variance;
    return features.sub(mean).div(variance.pow(0.5));
  }

  recordMSE() {
    const mse = this.features.matMul(this.weights)
      .sub(this.labels).pow(2).sum()
      .div(this.features.shape[0])
      .bufferSync()
      .get();
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}

/*
gradientDescentManual() {
  const { learningRate } = this.options;
  const currentGuessesForMPG = this.features.map(row => {
    return this.m * row[0] + this.b
  });
  const N = this.features.length;
  const bSlope = _.sum(currentGuessesForMPG.map((guess, index) => {
    return guess - this.labels[index][0];
  })) * 2 / N;
  const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
    return -1 * this.features[i][0] * (this.labels[i][0] - guess);
  })) * 2 / N;

  this.m = this.m - mSlope * learningRate;
  this.b = this.b - bSlope * learningRate;
}
*/
