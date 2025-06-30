// Generates a basic neural network diagram with specified layers
import {matrix} from "./matrix.js"
import {NNet} from "./NNet.js"
import {NGraph} from "./NGraph.js"

function createImageData(layerConfig){
    let data = {
        layerConfig,
        width : 800,
        height : 600,
        neuronRadius : 15,
        neuronSpacing : 50
    }
    data.layerSpacing = data.width / (data.layerConfig.length + 1);
    data.get = function(prop,...context){
        let v = this[prop]
        if (v instanceof Function ){
            return v(context)
        } else {
            return v
        }
    }
    return data
}

function init(d,container){
    d3.select(container).selectAll("*").remove();
    d.svg = d3.select(container).append("svg")
        .attr("width", d.width)
        .attr("height", d.height);
}

function border(d){
    d.svg.append("rect")
        .attr("x", 2)
        .attr("y", 2)
        .attr("width", d.width-2)
        .attr("height", d.height-2)
        .style("fill", "none")           // Transparent fill
        .style("stroke", "black")        // Border color
        .style("stroke-width", 2);
}

// Main: Generate the neural network diagram
function SmallNetwork() {
    // Configure the network: [Input Neurons, Hidden Layer Neurons, Output Neurons]
    const d = createImageData([5, 8, 2]); // Example: 4 inputs, 6 hidden, 3 outputs
    // Generate the visualization in the div with id="chart"
    d.output=[0.975,0.121]
    d.input=[1,0,0,1,2]

    let v = [
        {input:[1, 0, 0, 2, 1], output:[0.01,0.994]},
        {input:[1, 0, 0, 2, 0], output:[0.001,0.993]},
        {input:[0, 1, 0, 0, 1], output:[.995,0.003]},
        {input:[0, 1, 0, 0, 0], output:[.993,0.004]},
        {input:[1,1,0,2,0], output:[.135,0.03]},
        {input:[1,1,3,1,1], output:[.324,0.65]}
    ].sort((a,b)=>Math.random()-0.5)
    d.input = v[0].input
    d.output = v[0].output
    init(d,"#chart")
    NNet.generate(d);
    border(d)
}

function PrunedNetwork() {
    // Configure the network: [Input Neurons, Hidden Layer Neurons, Output Neurons]
    const d = createImageData([5, 12, 12, 5]);
    d.height=600
    d.neuronRadius = 10
    d.neuronSpacing = 30
    d.output=false
    d.input=false
    d.connectionFilter = (d,i,j)=>{
        let dist =  Math.abs(i/d.layerConfig[d.layerIndex] - j/d.layerConfig[d.layerIndex+1] )
        return Math.random() * dist < 0.2
    }
    init(d,"#chart")
    NNet.generate(d);
    border(d)
}


function ConvolvNetwork() {
    // Configure the network: [Input Neurons, Hidden Layer Neurons, Output Neurons]
    const d = createImageData([18, 12, 12, 5]);
    d.height=600
    d.neuronRadius = 10
    d.neuronSpacing = 30
    d.output=false
    d.input=false
    d.connectionFilter = (d,i,j)=>{
        if (d.layerIndex > 0) return true
        let dist =  Math.abs(i/d.layerConfig[d.layerIndex] - j/d.layerConfig[d.layerIndex+1] )
        return (dist<0.1)
    }
    init(d,"#chart")
    NNet.generate(d);
    border(d)
}

function CrossLayerNetwork() {
    // Configure the network: [Input Neurons, Hidden Layer Neurons, Output Neurons]
    const d = createImageData([5, 40, 5]);
    d.height=600
    d.neuronRadius = 10
    d.neuronSpacing = 30
    d.connectTreshold = 0.20
    d.layers = 4
    d.connect = function(from,to){
        if (from.layer == "output") return false
        if (to.layer == "input") return false
        if (from.i >= to.i) return false
        if (from.x >= to.x) return false
        let dist = Math.sqrt((from.x-to.x)**2 +(from.y-to.y)**2 )
        dist = dist / d.graphWidth
        // console.log(from, to, dist)
        //return true // fully connected
        return ( Math.random() * dist < d.connectTreshold ) // distance based connections
        // todo assert that every neuron has at least some inputs and some outputs
    }
    init(d,"#chart")
    NGraph.generate(d);
    border(d)
}

function GraphNetwork() {
    // Configure the network: [Input Neurons, Hidden Layer Neurons, Output Neurons]
    const d = createImageData([5, 60, 5]);
    d.height=600
    d.neuronRadius = 10
    d.neuronSpacing = 30
    d.connectTreshold = 0.10
    d.layers = 30
    d.connect = function(from,to){
        if (from.layer == "output") return false
        if (to.layer == "input") return false
        if (from.i >= to.i) return false
        if (from.x > to.x) return false
        let dist = Math.sqrt((from.x-to.x)**2 +(from.y-to.y)**2 )
        dist = dist / d.graphWidth
        // console.log(from, to, dist)
        //return true // fully connected
        return ( Math.random() * dist < d.connectTreshold ) // distance based connections
        // todo assert that every neuron has at least some inputs and some outputs
    }
    init(d,"#chart")
    NGraph.generate(d);
    border(d)
}

function LoopNetwork() {
    // Configure the network: [Input Neurons, Hidden Layer Neurons, Output Neurons]
    const d = createImageData([5, 60, 5]);
    d.height=600
    d.neuronRadius = 10
    d.neuronSpacing = 30
    d.connectTreshold = 0.10
    d.layers = 30
    d.connect = function(from,to){
        if (from.layer == "output") return false
        if (to.layer == "input") return false
        let dist = Math.sqrt((from.x-to.x)**2 +(from.y-to.y)**2 )
        dist = dist / d.graphWidth
        if (from.i >= to.i) {
            // backloop
            return (Math.random() * dist < d.connectTreshold / 20)
        } else {
            return (Math.random() * dist < d.connectTreshold) // distance based connections
        }
        // todo assert that every neuron has at least some inputs and some outputs
    }
    init(d,"#chart")
    NGraph.generate(d);
    border(d)
}

function Matrix() {
    const d = {
        width : 800,
        height : 400,
        rows:3,
        columns:10,
    }
    init(d,"#chart")
    matrix.generate(d);
    // border(d)
}


window.SmallNetwork = SmallNetwork;
window.PrunedNetwork = PrunedNetwork;
window.ConvolvNetwork = ConvolvNetwork;
window.GraphNetwork = GraphNetwork
window.LoopNetwork = LoopNetwork
window.CrossLayerNetwork = CrossLayerNetwork
window.Matrix = Matrix;
window.glow = NNet.glow
// default show pruned
PrunedNetwork()
// Call main to execute the program
// SmallNetwork();
// ConvolvNetwork()
// Matrix()