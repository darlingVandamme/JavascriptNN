import {NNet} from "../index.js";
import {getImages,getTestImages} from "./readImages.js";
import {countlog} from "../lib/counter.js"

let net = new NNet(28*28)


function run() {
    //console.time("total")
    net.step = 3
    net.batchSize = 10
    net.epochs = 3
    //net.translateInput=  (v)=> v.map(item=> item/256 )
    net.translateExpected = (expexted) =>{
        let result = new Array(10).fill(0,0,10)
        result[expexted] = 1
        return result
    }
    net.translateOutput = (output)=> {
        let result = {}
        result.label = output.reduce((prev, val, i, arr) => val > arr[prev] ? i : prev, 0)
        result.score = output[result.label]
        result.output = output
        return result
    }

    net.calculateCosts = true
    //net.addLayer(35)
    net.addLayer(30)
    net.addLayer(10)  // output

    let images = getImages(0,50000)
    // console.log(JSON.stringify(images[0]))
    train(images)
    check(10000,true)
    //weights()
}

function train(images){
    console.time("train")
    let startTime = Date.now()
    for (let epochs=0;epochs<net.epochs ; epochs++) {
        //net.step = net.step*0.9
        images.forEach(image=>{
            net.train(image.doublePixels, image.label, image.label)
            if (net.trainings%10000==0){
                console.log("train "+net.trainings+"  Cost "+net.getAverageCost(100).toFixed(6))
            }
        })
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
    let testImages = getTestImages(start,start+count)
    let correct = 0
    testImages.forEach((image,i)=> {
        let result = net.check(image.doublePixels)
        //let result = {label: net.getHighest(output), score:1}  //
        if(image.label == result.label) correct++

        /*if (show) console.log(((image.label == result.label)?"check ":"MISS ")
            +i+" "+image.label+"  <=> "+result.label+"  "+result.score.toFixed(4)+"   "+ output.map(r=>r.toFixed(3)))*/
    })

    if (show) console.log("Network "+net.layers+" layers "+ net.neurons.length+" neurons  ("+net.allNeurons.length+") "+net.neurons.reduce((prev, n)=>(prev+n.in.length),0)+" weights" )
    if (show) console.log("Training iterations "+ net.trainings+"  TrainTime "+net.trainTime+" "+(net.trainings/(net.trainTime/1000)).toFixed(2)+" Trainings/s " + net.step +" step")
    if (show) console.log("Training step:"+  net.step +"   BatchSize: "+net.batchSize)
    if (show) console.log("Check iterations "+ count+" "+(1000*count/(Date.now()-startTime)).toFixed(2)+" Checks/s ")
    console.log("success rate "+ (correct/testImages.length).toFixed(3))
    if (show) console.log("Avg Cost "+ net.getAverageCost(100).toFixed(5))
    //if (show) console.log("labels ",net.patterns)
    if (show) console.timeLog("check")
    countlog()

}




function weights(){
    net.neurons.forEach(n=>{
        console.log("Neuron "+n.id+"  bias "+n.bias)
        n.in.forEach(con=>{
            console.log("Weight "+con.in.id+" \t"+con.weight)
        })
    })

}

// console.log(resultArray(4))
run()
