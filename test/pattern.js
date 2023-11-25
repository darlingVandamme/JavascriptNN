import NNet from "../lib/NNet.js";



function test(){
    let n = new NNet(5)
    n.patterns[3] = {label :3, count:5, pattern:[0,0,0,1,0,0,0,0,0,0]}
    n.patterns[4] = {label :4, count:5, pattern:[0,0,0,0,1,0,0,0,0,0]}
    n.patterns[5] = {label :5, count:5, pattern:[0,0,0,0,0,1,0,0,0,0]}

    console.log("found ",n.getResult([0,0,0,0.1,0.2,1,0,0,0,0]))
}


test()