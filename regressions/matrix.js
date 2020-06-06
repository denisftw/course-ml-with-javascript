import * as tf from '@tensorflow/tfjs-node';
import _ from 'lodash';

export class Matrix {
  /**
   * @param {number[][]} numbers
   */
  constructor(numbers) {
    this.numbers = numbers;
    this.rows = numbers.length;
    this.columns = numbers[0].length;
  }
  print() {
    console.log('[')
    this.numbers.forEach(row => {
      const rowStr = row.join(" ");
      console.log(' [' + rowStr + ']')
    })
    console.log(']')
  }
}

/**
 *
 * @param {Matrix} a
 * @param {Matrix} b
 * @returns {Matrix}
 */
export function multiply(a, b) {
  if (a.columns !== b.rows) throw "Dimensions do not match!"
  const c = [];
  for (let i = 0; i < a.rows; i++) {
    const cRow = [];
    for (let j = 0; j < b.columns; j++) {
      const row = a.numbers[i];
      const products = row.map((aij, jj) => {
        return aij * b.numbers[jj][j];
      })
      cRow[j] = _.sum(products);
    }
    c.push(cRow);
  }
  return new Matrix(c);
}
