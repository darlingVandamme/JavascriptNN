// const { Neuron } = require('../src/neuron.js');
// import { jest } from '@jest/globals';
import Neuron  from '../lib/neuron.js';

describe('Neuron Class', () => {
    let hiddenNeuron;
    let inputNeuron;

    beforeEach(() => {
        inputNeuron = new Neuron();
        hiddenNeuron = new Neuron();
        hiddenNeuron.bias = 0.2
        let weights = [0.5,0.6,0.7]
        let values = [1,0,1]
        let input = [new Neuron(), new Neuron(), new Neuron()]
        input.forEach((n,i)=>n.value=values[i])
        hiddenNeuron.connectLayer(input,weights)
    });
    
    test('it should instantiate the Neuron class correctly', () => {
        expect(hiddenNeuron).toBeInstanceOf(Neuron);
    });

    test('reset method should reset value and delta', () => {
        hiddenNeuron.value = 5;
        hiddenNeuron.delta = 3;
        hiddenNeuron.reset();
        expect(hiddenNeuron.value).toBeNull();
        expect(hiddenNeuron.delta).toBeNull();
    });

    /*test('connect method connects neurons', () => {
        const otherNeuron = new Neuron();
        neuron.connect(otherNeuron);
        expect(neuron.in).toHaveLength(1);
    }); */

    test('setWeights method sets weights', () => {
        const weights = [1, 2, 3];
        hiddenNeuron.setWeights(weights);
        hiddenNeuron.in.forEach((c, i) => {
            expect(c.weight).toBe(weights[i]);
        });
    });

    test('getWeights method returns array of weights', () => {
        const weights = [1, 2, 3];
        hiddenNeuron.in.forEach((c, i) => {
            c.weight = weights[i];
        });
        const result = hiddenNeuron.getWeights();
        expect(result).toEqual(weights);
    });

    test('isInput returns false if neuron has input', () => {
        expect(hiddenNeuron.isInput()).toBeFalsy();
    });
    test('isInput returns true if neuron has no input', () => {
        expect(inputNeuron.isInput()).toBeTruthy();
    });
    test('isOutput returns true if neuron has no output', () => {
        expect(hiddenNeuron.isOutput()).toBeTruthy();
    });

    it('should calculate correct value', () => {
        hiddenNeuron.ff();
        let expectedZ = 1.4 // (0.5×1+0.7×1+0.2)
        let expectedA= 0.8021838885585817 // 1/(1+ exp(-(0.5×1+0.7×1+0.2)))
        expect(hiddenNeuron.z).toEqual(expectedZ);
        expect(hiddenNeuron.value).toEqual(expectedA);
    });


});