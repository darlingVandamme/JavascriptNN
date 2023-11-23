import NNet from "../lib/NNet.js";

function test(){
    //console.time("total")
    const net = new NNet(5)
    net.addLayer(4)
    net.addLayer(2)
    net.batchSize = 1
    net.step = 0.5
    net.feed([1,0,0,2,1])
    console.log(net.getOutput())
    console.log(net)
    console.time()

    net.train([1,0,0,2,1],[0,1])
    console.log(net.getOutput())
    net.train([0, 1, 0, 0, 0], [1, 0])
    console.log(net.getOutput())
    console.log(net)
    console.time("train")

    for (let i=1 ; i<1000;i++) {

        net.train([1, 0, 0, 2, 1], [0, 1])
        net.train([1, 0, 0, 2, 0], [0, 1])
        net.train([0, 1, 0, 0, 1], [1, 0])
        net.train([0, 1, 0, 0, 0], [1, 0])
    }
    console.timeLog("train")
    console.log(net)
    net.feed([1,0,0,2,2])
    console.log("expected  [0 , 1] ",net.getOutput())
    net.feed([0,1,0,0,0])
    console.log("expected  [1 , 0] ",net.getOutput())
    net.feed([0,0,0,0,0])
    console.log("expected  [????] ",net.getOutput())



    console.timeEnd()

}

test()
