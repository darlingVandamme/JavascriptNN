import {NNet} from "../index.js";
import {getImages} from "./readImages.js";
import * as tf from '@tensorflow/tfjs';

const model = tf.sequential();
model.add(tf.layers.dense({units: 30, activation: 'sigmoid', inputShape: [28*28]}));
model.add(tf.layers.dense({units: 10, activation: 'sigmoid'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
let startTime = Date.now()
let trainTime = 50000
let epochs = 3
let trainSize = 50000
let checkTime

function resultArray(label){
    let result = new Array(10).fill(0,0,10)
    result[label] = 1
    return result
}

async function run() {
    //console.time("total")

    let images = getImages(false, 0, 50000)
    //console.log(JSON.stringify(images[0]))
    await train(images)
    check(10000, true)
    //weights()
}

function onBatchEnd(batch, logs) {
    console.log("batch "+(Date.now()-startTime))
    // console.log('Accuracy', logs.acc);
}

function train(images){
    console.time("train")
    let startTime = Date.now()

    let imageData = tf.tensor( images.filter((d,i)=>i<trainSize).map((d)=> d.pixels.map(p=>p/256) ))
    let labelData = tf.tensor(images.filter((d,i)=>i<trainSize).map((d)=> resultArray(d.label)))

    return model.fit(imageData, labelData, {
        epochs: epochs,
        batchSize: 10,
        callbacks: {
        onTrainBegin: async () => {
            console.log("onTrainBegin")
        },
            onTrainEnd: async (epoch, logs) => {
            console.log("onTrainEnd" + epoch + JSON.stringify(logs))
        },
            onEpochBegin: async (epoch, logs) => {
            console.log("onEpochBegin" + epoch + JSON.stringify(logs))
        },
            onEpochEnd: async (epoch, logs) => {
            console.log("batch "+(Date.now()-startTime))
            console.log("onEpochEnd" + epoch + JSON.stringify(logs))
        }/*,
            onBatchBegin: async (epoch, logs) => {
            console.log("onBatchBegin" + epoch + JSON.stringify(logs))
        },
            onBatchEnd: async (epoch, logs) => {
                console.log("batch "+(Date.now()-startTime))
                console.log("onBatchEnd" + epoch + JSON.stringify(logs))
            }*/
        }
    }).then(info => {
         console.log('Final accuracy', info.history.acc);
        trainTime = Date.now()-startTime
        // check(1000,false)
        console.timeLog("train")
        return
    });

}


function check(size,show){

    if (show) console.time("check")
    let startTime = Date.now()
    let start = Math.floor(Math.random()*(10000 - size))
    if (show) console.log("test images "+start+" "+size )
    let testImages = getImages(true,start,start+size)
    let correct = 0
    for (let i=0;i<testImages.length;i++) {
        let img = testImages[i].pixels.map((d)=> d/256)
        let output = model.predict(tf.tensor(1,img))
        if (show) console.log(output)

        // if(testImages[i].label == result.label) correct++

        /*if (show) console.log(((testImages[i].label == result.label)?"check ":"MISS ")
            +i+" "+testImages[i].label+"  <=> "+result.label+"  "+result.score.toFixed(4)+"   "+ output.map(r=>r.toFixed(3)))*/
    }

    checkTime = Date.now()-startTime
    if (show) console.log("Training iterations "+ (trainSize*epochs)+"  TrainTime "+trainTime+" "+((trainSize*epochs)/(trainTime/1000)).toFixed(2)+" Trainings/s " )
    if (show) console.log("Check iterations "+ size+"  Time "+checkTime+"   "+(1000*size/checkTime).toFixed(2)+" Checks/s ")
    console.log("success rate "+ (correct/testImages.length).toFixed(3))
    if (show) console.timeLog("check")

}

// console.log(resultArray(4))
run()
