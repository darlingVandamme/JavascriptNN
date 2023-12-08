import {NNet} from "../index.js";
import {getImages} from "./readImages.js";
import {countlog} from "../lib/counter.js"

const net = new NNet(28*28)

function resultArray(label){
    let result = new Array(10).fill(0,0,10)
    result[label] = 1
    return result
}

function run() {
    //console.time("total")
    net.step = 3
    net.batchSize = 10
    net.epochs = 3
    //net.translateInput=  (v)=> v.map(item=> item/256 )
    net.calculateCosts = true
    //net.addLayer(35)
    net.addLayer(30)
    net.addLayer(10)  // output

    let images = getImages(false,0,50000)
    //console.log(JSON.stringify(images[0]))
    train(images)
    check(10000,true)
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
        check(1000,false)
        console.log("epoch "+epochs+"  Cost "+net.getAverageCost().toFixed(5) + " step "+net.step.toFixed(2))
        net.trainTime = Date.now() - startTime
        console.timeLog("train")
    }
}

function check(count,show){
    if (show) console.time("check")
    let startTime = Date.now()
    let start = Math.floor(Math.random()*(10000 - count))
    if (show) console.log("test images "+start+" "+count )
    let testImages = getImages(true,start,start+count)
    let correct = 0
    for (let i=0;i<testImages.length;i++) {
        let output = net.check(testImages[i].pixels)
        let result = net.getResult(output)
        if(testImages[i].label == result.label) correct++

        if (show) console.log(((testImages[i].label == result.label)?"check ":"MISS ")
            +i+" "+testImages[i].label+"  <=> "+result.label+"  "+result.score.toFixed(4)+"   "+ output.map(r=>r.toFixed(3)))
    }

    if (show) console.log("Network "+net.layers+" layers "+ net.neurons.length+" neurons  ("+net.allNeurons().length+") "+net.neurons.reduce((prev,n)=>(prev+n.in.length),0)+" weights" )
    if (show) console.log("Training iterations "+ net.trainings+"  TrainTime "+net.trainTime+" "+(net.trainings/(net.trainTime/1000)).toFixed(2)+" Trainings/s " + net.step +" step")
    if (show) console.log("Training step:"+  net.step +"   BatchSize: "+net.batchSize)
    if (show) console.log("Check iterations "+ count+" "+(1000*count/(Date.now()-startTime)).toFixed(2)+" Checks/s ")
    console.log("success rate "+ (correct/testImages.length).toFixed(3))
    if (show) console.log("Avg Cost "+ net.getAverageCost().toFixed(5))
    //if (show) console.log("labels ",net.patterns)
    if (show) console.timeLog("check")
    countlog()

}

function weights(){
    net.neurons.forEach(n=>{
        console.log("Neuron "+n.id+"  layer "+n.layer+"  bias "+n.bias)
        n.in.forEach(con=>{
            console.log("Weight "+con.in.id+" \t"+con.weight)
        })
    })

}

// console.log(resultArray(4))
run()
