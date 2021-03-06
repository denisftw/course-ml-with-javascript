import * as tf from '@tensorflow/tfjs-node';
import _ from 'lodash';

export class LogisticRegression {
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
    this.costHistory = [];

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 1000,
    }, options);

    /** @type {tf.Tensor} */
    this.weights = tf.zeros([this.features.shape[1],
      this.labels.shape[1]]);
  }

  /**
   *
   * @param {tf.Tensor} features
   * @param {tf.Tensor} labels
   */
  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();
    const differences = currentGuesses.sub(labels);
    const slopes = features.transpose()
      .matMul(differences)
      .div(features.shape[0]);
    return this.weights.sub(
      slopes.mul(this.options.learningRate));
  }

  train() {
    const { batchSize } = this.options;
    const batchQuantity = Math.floor(
      this.features.shape[0] / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;
        this.weights = tf.tidy(() => {
          const featureSlice = this.features.
            slice([startIndex, 0], [batchSize, -1]);
          const labelSlice = this.labels.
            slice([startIndex, 0], [batchSize, -1]);
          return this.gradientDescent(featureSlice, labelSlice);
        })
      }
      this.recordCost();
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
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  /** @returns {number} */
  test(testFeaturesA, testLabels) {
    const predictions = this.predict(testFeaturesA);
    testLabels = tf.tensor(testLabels).argMax(1);
    const incorrect = predictions
      .notEqual(testLabels)
      .sum()
      .bufferSync()
      .get();
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
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

    const filler = variance.cast('bool').logicalNot().cast('float32');

    this.mean = mean;
    this.variance = variance.add(filler);
    return features.sub(mean).div(this.variance.pow(0.5));
  }

  recordCost() {
    const guesses = this.features.matMul(this.weights).softmax();
    const term1 = this.labels.transpose().matMul(guesses.add(1e-7).log());
    const term2 = this.labels.mul(-1).add(1).transpose()
      .matMul(
        guesses.mul(-1).add(1).add(1e-7).log()
      );
    const cost = term1.add(term2).div(this.features.shape[0])
      .mul(-1).bufferSync().get(0, 0);
    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      this.options.learningRate *= 1.05;
    }
  }
}
