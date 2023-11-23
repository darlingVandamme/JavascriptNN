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
    net.step = 0.1
    net.addLayer(16)
    net.addLayer(10)  // output

    let images = readMNIST(100,1820)
    console.log(images[0])

    for (let i=0;i<images.length;i++) {
        net.train(images[i].pixels, resultArray(images[i].label))
    }

    let testImages = readMNIST(5000,5020)
    for (let i=0;i<testImages.length;i++) {
        let result = net.check(testImages[i].pixels)
        console.log("check "+i+" "+testImages[i].label+"  "+result)
    }


}

// console.log(resultArray(4))
test()
