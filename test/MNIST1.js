import {NNet} from "../index.js";
import {readMNIST} from "./readImages.js";


function resultArray(label){
    let result = new Array(10).fill(0,0,10)
    result[label] = 1
    return result
}

function test() {
    //console.time("total")
    const net = new NNet(28*28)
    net.step = 3
    net.batchSize = 10
    net.translateInput=  (v)=> v.map(item=> item/256 )
    //net.addLayer(30)
    net.addLayer(30)
    net.addLayer(10)  // output

    let images = readMNIST(1,20000)
    console.log(images[0])

    console.time("train")
    for (let epochs=0;epochs<20; epochs++) {

        for (let i = 0; i < images.length; i++) {
            net.train(images[i].pixels, resultArray(images[i].label))
        }
        console.log("epoch "+epochs)
        console.timeLog("train")

    }

    let testImages = readMNIST(15000,15020)
    for (let i=0;i<testImages.length;i++) {
        let result = net.check(testImages[i].pixels)
        let max = Math.max(...result)
        let r= result.indexOf(max)

        console.log("check "+i+" "+testImages[i].label+"  "+r+"   "+result)
    }


}

// console.log(resultArray(4))
test()
