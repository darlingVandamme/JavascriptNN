// Generates a basic neural network diagram with specified layers
import {matrix} from "./matrix.js"
import {NNet} from "./NNet.js"

function createImageData(layerConfig){
    let data = {
        layerConfig,
        width : 800,
        height : 400,
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
    d.output=true
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
    d.output=true
    d.input=true
    d.connectionFilter = (d,i,j)=>{
        if (d.layerIndex > 0) return true
        let dist =  Math.abs(i/d.layerConfig[d.layerIndex] - j/d.layerConfig[d.layerIndex+1] )
        return (dist<0.1)
    }
    init(d,"#chart")
    NNet.generate(d);
    border(d)
}

function Matrix() {
    const d = {
        width : 800,
        height : 400,
        rows:3,
        columns:3,
    }
    init(d,"#chart")
    matrix.generate(d);
    border(d)
}


window.SmallNetwork = SmallNetwork;
window.ConvolvNetwork = ConvolvNetwork;
window.Matrix = Matrix;
// Call main to execute the program
// SmallNetwork();
// ConvolvNetwork()
// Matrix()