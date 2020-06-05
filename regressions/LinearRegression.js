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
    this.features = features;
    this.labels = labels;
    this.options = Object.assign({
      learningRate: 0.1,
      iterations: 1000,
    }, options);

    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
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

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}
