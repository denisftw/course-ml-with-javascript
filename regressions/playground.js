import * as tf from '@tensorflow/tfjs-node';
import { Matrix, multiply } from './matrix';

const a = new Matrix([
  [1, 5],
  [2, 6],
  [3, 7],
  [4, 8],
]);

const b = new Matrix([
  [10, 30, 50],
  [20, 40, 60],
])

const c = multiply(a, b);

c.print();
