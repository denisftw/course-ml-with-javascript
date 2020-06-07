import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const features = tf.tensor([
  [0.01],
  [0.1],
  [0.2],
  [0.3],
  [0.4],
  [0.5],
  [0.6],
  [0.7],
  [0.8],
]);

const result = features.greater(0.4).cast('int32');

result.print();
