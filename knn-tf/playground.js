import * as tf from '@tensorflow/tfjs-node';

const features = tf.tensor([
    [-121, 47],
    [-121.2, 46.5],
    [-122, 46.4],
    [-120.9, 46.7],
]);

const labels = tf.tensor([
    [200],
    [250],
    [215],
    [240],
]);

const predictionPoint = tf.tensor([-121, 47]);

const distances = features
    .sub(predictionPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1);

const result = distances.concat(labels, 1);

// result.print();

const unstucked = result
    .unstack()
    .sort((a, b) => {
        const first = a.bufferSync().get(0);
        const second = b.bufferSync().get(0);
        return first < second ? 1 : -1;
    });

unstucked.forEach(s => s.print());

const k = 2;

const topK = unstucked
    .slice(0, k)
    .reduce((acc, pair) => {
        return acc + pair.bufferSync().get(1);
    }, 0) / k;

console.log(topK);