import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const features = tf.tensor([
  [1,2,3,9],
  [10,20,35,95],
  [100,200,350,950],
]);

const sliced = features.slice([1, 0], [1, -1]);

sliced.print();
