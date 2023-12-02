import NNet from "../lib/NNet.js";

function test(){
    //console.time("total")
    const net = new NNet(5)
    net.addLayer(50)
    net.addLayer(2)
    net.batchSize = 10
    net.step = 10
    net.feed([1,0,0,2,1])
    console.log(net.getOutput())
    //console.log(net)
    console.time()

    net.train([1,0,0,2,1],[0,1])
    console.log(net.getOutput())
    net.train([0, 1, 0, 0, 0], [1, 0])
    console.log(net.getOutput())
    //console.log(net)
    console.time("train")

    for (let i=1 ; i<100000;i++) {

        net.train([1, 0, 0, 2, 1], [0, 1])
        net.train([1, 0, 0, 2, 0], [0, 1])
        net.train([0, 1, 0, 0, 1], [1, 0])
        net.train([0, 1, 0, 0, 0], [1, 0])
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

    console.log("Network "+net.layers+" layers "+ net.neurons.length+" neurons  ("+net.allNeurons().length+") "+net.neurons.reduce((prev,n)=>(prev+n.in.length),0)+" weights" )

    console.log("Avg Cost "+ net.getAverageCost().toFixed(5))

    console.time("test")
    for (let i=0;i<1000000;i++){
        net.feed([0,1,0,0,0]);
        net.getOutput();
    }
    console.timeLog("test")


    console.timeEnd()

}

test()
