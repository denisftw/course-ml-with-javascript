import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const guesses = tf.tensor([
  [15, 20, 23],
]);

guesses.softmax().print();

