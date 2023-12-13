import {NNet,Neuron} from "../index.js";
import {getImages} from "./readImages.js";
import {countlog} from "../lib/counter.js"
import {Worker, isMainThread, parentPort, workerData,threadId} from 'worker_threads';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let net

function resultArray(label){
    let result = new Array(10).fill(0,0,10)
    result[label] = 1
    return result
}

function run() {
    //console.time("total")
    net = new NNet(28*28)
    net.step = 2
    net.batchSize = 10
    net.epochs = 3
    //net.translateInput=  (v)=> v.map(item=> item/256 )
    net.calculateCosts = true
    //net.addLayer(35)
    net.addLayer(30)
    net.addLayer(10)  // output

    let images = getImages(false,0,50000)
    // console.log(JSON.stringify(images[0]))
    train(images)
    check(40000,true)
    //weights()
}

function train(images){
    console.time("train")
    let startTime = Date.now()
    for (let epochs=0;epochs<net.epochs ; epochs++) {
        //net.step = net.step*0.9
        for (let i = 0; i < images.length; i++) {
            net.train(images[i].pixels, resultArray(images[i].label), images[i].label)
            if (net.trainings%5000==0){
                console.log("train "+net.trainings+"  Cost "+net.getAverageCost(10).toFixed(5))
            }

        }
        //check(1000,false)
        console.log("epoch "+epochs+"  Cost "+net.getAverageCost().toFixed(5) + " step "+net.step.toFixed(2))
        net.trainTime = Date.now() - startTime
        console.timeLog("train")
    }
}

function checkImages() {
    let startTime = Date.now()
    let correct = 0
    net  = workerData.network;
    const images = workerData.images
    const show = workerData.show
    // net.prototype = NNet.prototype
    Object.setPrototypeOf(net, NNet.prototype);
    net.neurons.forEach(n=>Object.setPrototypeOf(n, Neuron.prototype));

    console.log("start Checking images " + threadId +" "+ show+" "+images.length+" "+net.layers+" "+net.input.length+" "+(Date.now()-startTime))
    for (let i = 0; i < images.length; i++) {
        net.feed(images[i].pixels)
        let output = net.getOutput()
        let result = {label: net.getHighest(output), score: 1}  //
        //let result= net.getResult(output)
        if (images[i].label == result.label) {
            correct++
        }
        /*if (show) { parentPort.postMessage(((images[i].label == result.label) ? "check " : "MISS ")
            + i + " " + images[i].label + "  <=> " + result.label + "  " + result.score.toFixed(4) + "   " + output.map(r => r.toFixed(3)))//send Message
        }*/
    }
    console.log("Stop checking for thread "+ threadId +" total "+images.length+" ratio "+correct/images.length)
    console.log("prof "+(Date.now()-startTime))
    console.log("speed "+1000*images.length/(Date.now()-startTime))
    return correct;
}

function check(count,show){
    if (show) console.time("check")
    let startTime = Date.now()
    let start = Math.floor(Math.random()*(50000 - count))
    if (show) console.log("test images "+start+" "+count )
    let testImages = getImages(false,start,start+count)
    let correct = 0
    let numCPUs = 4
    const imagesPerWorker = testImages.length/numCPUs
    const workers = [];
    for (let i = 0; i < numCPUs; i++) {
        //console.log("Start worker "+net.neurons.length)
        workers.push(new Worker(__filename, {
            workerData: {
                network:net,
                images:testImages.slice(i*imagesPerWorker,(i+1)*imagesPerWorker),
                show:show
            },
        }));
    }
    for (const worker of workers) {
        worker.on('message', (msg) => console.log(msg));
        worker.on('error', (err) => console.error(`Worker error: ${err}`));
        worker.on('exit', (code) => {
            if (code !== 0)
                console.error(`Worker stopped with exit code ${code}`);
        });
    }
    console.log("Set up "+numCPUs+" threads "+(Date.now()-startTime))
    if (show) console.log("Network "+net.layers+" layers "+ net.hoNeurons.length+" neurons  ("+net.neurons.length+") "+net.hoNeurons.reduce((prev,n)=>(prev+n.in.length),0)+" weights" )
    if (show) console.log("Training iterations "+ net.trainings+"  TrainTime "+net.trainTime+" "+(net.trainings/(net.trainTime/1000)).toFixed(2)+" Trainings/s " + net.step +" step")
    if (show) console.log("Training step:"+  net.step +"   BatchSize: "+net.batchSize)
    //if (show) console.log("labels ",net.patterns)

}




function weights(){
    net.hoNeurons.forEach(n=>{
        console.log("Neuron "+n.id+" bias "+n.bias)
        n.in.forEach(con=>{
            console.log("Weight "+con.in.id+" \t"+con.weight)
        })
    })

}

// console.log(resultArray(4))
if(isMainThread) {
    run()
} else {
    // check the number of images
    checkImages()
}
