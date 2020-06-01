import _ from 'lodash';

const points = [200, 150, 650, 430];

const [min, max] = [_.min(points), _.max(points)];
const divider = max - min;
const result = _.map(points, point => {
    return (point - min) / divider;
});


console.log(result);