import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const features = tf.tensor([10,20,35,95]);
const { mean, variance } = tf.moments(features, 0);
const std = features.sub(mean).div(variance.pow(0.5));


const arr = [1,2,3];
arr.unshift(0);
console.log(arr);

// ones.transpose().print();
console.log(std.bufferSync());
