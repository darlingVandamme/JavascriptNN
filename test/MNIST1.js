import {NNet} from "../index.js";
import {readMNIST} from "./readImages.js";


const net = new NNet(28*28)


function resultArray(label){
    let result = new Array(10).fill(0,0,10)
    result[label] = 1
    return result
}

function test() {
    //console.time("total")
    net.step = 3
    net.batchSize = 1
    net.translateInput=  (v)=> v.map(item=> item/256 )
    //net.addLayer(30)
    net.addLayer(30)
    net.addLayer(10)  // output

    let images = readMNIST(0,20000)
    console.log(JSON.stringify(images[0]))
    train(images)
    check(1000)
    //weights()
}

function train(images){
    console.time("train")
    let startTime = Date.now()
    for (let epochs=0;epochs<30 ; epochs++) {
        net.step = net.step*0.9
        for (let i = 0; i < images.length; i++) {
            net.train(images[i].pixels, resultArray(images[i].label), images[i].label)
            if (net.trainings%5000==0){
                console.log("train "+net.trainings+"  Cost "+net.getAverageCost().toFixed(5))
            }

        }
        console.log("epoch "+epochs+"  Cost "+net.getAverageCost().toFixed(5) + " step "+net.step)
        net.trainTime = Date.now() - startTime
        console.timeLog("train")
    }
}

function check(count){
    console.time("check")
    let startTime = Date.now()
    let testImages = readMNIST(50000,50000+count)
    let correct = 0
    for (let i=0;i<testImages.length;i++) {
        let output = net.check(testImages[i].pixels)
        //let max = Math.max(...result)
        //let r= result.indexOf(max)
        let result = net.getResult(output)
        if(testImages[i].label == result.label) correct++
        console.log("check "+i+" "+testImages[i].label+"  <=> "+result.label+"  "+result.score.toFixed(4)+"   "+ output.map(r=>r.toFixed(3)))
    }
    console.log("Training iterations "+ net.trainings+"  TrainTime "+net.trainTime+" "+(net.trainings/(net.trainTime/1000)).toFixed(3)+" Trainings/s ")
    console.log("Check iterations "+ count+" "+(1000*count/(Date.now()-startTime)).toFixed(3)+" Checks/s ")
    console.log("success rate "+ (correct/testImages.length).toFixed(2))
    console.timeLog("check")

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
test()
