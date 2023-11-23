import fs from 'fs';
import  {createCanvas} from 'canvas';

// https://stackoverflow.com/questions/25024179/reading-mnist-dataset-with-javascript-node-js

import { URL } from 'url';
const __filename = new URL('', import.meta.url).pathname;
const __dirname = new URL('.', import.meta.url).pathname;
let mnistRoot = '../../MNIST/'

let dataFileBuffer
let labelFileBuffer


function readMNIST(start, end) {
    if (!dataFileBuffer) {
        console.log("reading images")
        dataFileBuffer = fs.readFileSync(__dirname + mnistRoot+"train-images.idx3-ubyte");
        labelFileBuffer = fs.readFileSync(__dirname + mnistRoot+'train-labels.idx1-ubyte');
    }
    let pixelValues = [];

    for (let image = start; image < end; image++)
    {
        let pixels = [];
        for (let y = 0; y <= 27; y++)
        {
            for (let x = 0; x <= 27; x++)
            {
                pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
            }
        }
        // todo use typed arrays???
        // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Typed_arrays

        let imageData  = {};
        imageData["index"] = image;
        imageData["label"] = labelFileBuffer[image + 8];
        imageData["pixels"] = pixels;
        pixelValues.push(imageData);
    }
    return pixelValues;
}

function saveMNIST(start, end) {
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');
    var pixelValues = readMNIST(start, end);
    pixelValues.forEach(function(image)
    {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (var y = 0; y <= 27; y++)
        {
            for (var x = 0; x <= 27; x++)
            {
                var pixel = image.pixels[x + (y * 28)];
                var colour = 255 - pixel;
                ctx.fillStyle = `rgb(${colour}, ${colour}, ${colour})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        const buffer = canvas.toBuffer('image/png')
        fs.writeFileSync(__dirname + mnistRoot+`images/image${image.index}-${image.label}.png`, buffer)
    })
}

function printImage(index){
    let image = readMNIST(index,index+1)[0]
    console.log(JSON.stringify(image))
}

export {readMNIST}

// saveMNIST(20, 50);
// printImage(20)
