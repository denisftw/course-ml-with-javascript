import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const features = tf.tensor([
  [1,2],
  [3,4],
  [5,6],
  [7,8],
]);

const weights = tf.tensor([
  [9,11,13],
  [10,12,14],
])

const result = features.matMul(weights);

result.print();
