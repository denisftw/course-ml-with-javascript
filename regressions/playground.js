import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const ones = tf.zeros([2,1]);

ones.transpose().print();
