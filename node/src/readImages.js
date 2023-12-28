import fs from 'fs';
import path, {dirname} from 'path'
import  {createCanvas} from 'canvas';

// https://stackoverflow.com/questions/25024179/reading-mnist-dataset-with-javascript-node-js

import {fileURLToPath, URL} from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let mnistRoot = '/../../../MNIST/'

const images = []

function getImages(start, end) {
    if (images.length==0){
        readMNIST("train")
        readMNIST("t10k")
    }
    return images.slice(start,end)
}
function getTestImages(start, end) {
    return getImages(start+60000,end+60000)
}

function readMNIST(imageType, start, end) {
    console.log("reading images "+imageType)
    if(!fs.existsSync(__dirname + mnistRoot+imageType+"-images.idx3-ubyte")){
        console.log("MNIST files not found")
        console.log("looking for MNIST files at location "+ path.resolve(__dirname + mnistRoot))
        console.log(__dirname + mnistRoot+imageType+"-images.idx3-ubyte")
        throw new Error("MNIST files not found");
    }
    let dataFileBuffer = fs.readFileSync(__dirname + mnistRoot+imageType+"-images.idx3-ubyte");
    let labelFileBuffer = fs.readFileSync(__dirname + mnistRoot+imageType+'-labels.idx1-ubyte');

    console.log("length "+dataFileBuffer.length)
    const max = labelFileBuffer.length-8
    console.log("max "+max)
    const imgData = new Uint8Array(dataFileBuffer,0)
    const labelData = new Uint8Array(labelFileBuffer,0)

    for (let i = 0; i < max; i++){
        let pixels = Float32Array.from(imgData.subarray(16+(i*28*28), 16+((i+1)*28*28))) //  [...imgData.slice(i*(28*28),(i+1)*(28*28))]
        images.push({
            index: i,
            label: labelData[i+8],
            pixels : pixels,
            doublePixels: pixels.map(p => ( p *1.0)/256 )
        })
    }
    return images
}

function saveMNIST(start, end) {
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');
    //var pixelValues = getImages( start, end);
    var pixelValues = getTestImages( start, end);
    pixelValues.forEach(function(image)
    {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (var y = 0; y < 28; y++)
        {
            for (var x = 0; x < 28; x++)
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
    let image = getImages(index,index+1)[0]
    console.log(JSON.stringify(image))
}

export {getImages,getTestImages}

 saveMNIST(9980,10000);
/*const startTime = Date.now()
printImage(20)
console.log(Date.now()-startTime)*/
