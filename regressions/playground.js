import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const features = tf.tidy(() => {
  return tf.tensor([
    [0, 20, 40],
    [0, 20.5, 80],
    [0, 20.3, 90],
  ]);
})

const { mean, variance } = tf.moments(features, 0);

// mean.print();
// variance.print();

variance.cast('bool')
  .logicalNot().cast('float32').print();

// features.sub(mean).div(variance.pow(0.5)).print();

// features.isNaN().print();
