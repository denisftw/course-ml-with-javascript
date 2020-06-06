import * as tf from '@tensorflow/tfjs-node';
import _ from 'lodash';

export class LinearRegression {
  /**
   *
   * @param {number[][]} features
   * @param {number[][]} labels
   * @param {{learningRate: number, iterations: number}} options
   */
  constructor(features, labels, options) {
    /** @type {tf.Tensor} */
    this.features = tf.tensor(features);
    /** @type {tf.Tensor} */
    this.labels = tf.tensor(labels);

    this.features = tf.ones([this.features.shape[0], 1]).
      concat(this.features, 1);

    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 1000,
    }, options);

    /** @type {tf.Tensor} */
    this.weights = tf.zeros([2, 1]);
  }

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights);
    const differences = currentGuesses.sub(this.labels);
    const slopes = this.features.transpose()
      .matMul(differences)
      .div(this.features.shape[0]);
    this.weights = this.weights.sub(
      slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
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
