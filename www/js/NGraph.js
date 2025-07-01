
export const NGraph = {}

NGraph.generate = function(d) {
    // Draw layers
    // input
    if (!d.d3) d.d3 = d3 // store a ref to d3 inside the d data
    const data=new Array(d.layerConfig[1]).fill(0)
    d.graphWidth = d.width*0.6
    d.graphX = (d.width - d.graphWidth) / 2
    d.graphHeight = d.height*0.7
    d.graphY = (d.height - d.graphHeight) / 2
    d.layerWidth = d.graphWidth / (d.layers || 100)
    d.data = data.map((v,i)=>{
        let x = d.graphX + (Math.random() * d.graphWidth)
        // round to layerWidth
        x = Math.floor(x / d.layerWidth) * d.layerWidth
        let y = d.graphY + (Math.random() * d.graphHeight)
        return {x,y, from:[], to:[]}
    }).sort((a,b)=> (a.x - b.x))
    // check collisions?
    preventCollisions(d.data,d.neuronRadius)
    let startY =  (d.height - (d.neuronSpacing * (d.layerConfig[0] - 1))) / 2;
    let input = Array.from({length:(d.layerConfig[0])},(v,i)=>
    {
        return {x:30,y:startY + i*d.neuronSpacing, layer : "input",from:[], to:[]}
    })
    startY =  (d.height - (d.neuronSpacing * (d.layerConfig[2] - 1))) / 2;
    let output = Array.from({length:(d.layerConfig[2])},(v,i)=> {
        return {x:d.width-50,y:startY + i*d.neuronSpacing, layer : "output",from:[], to:[]}
    })
    d.data = [...input,...d.data,...output]
    d.data.forEach((v,i)=>{v.i = i})
    //console.log(d.data)
    //drawLayer(d, d.layerConfig[0], 0);
    //output
    //drawLayer(d, d.layerConfig[d.layerConfig.length-1], d.layerConfig.length-1);
    drawConnections(d);
    drawGraph(d, d.data)
    drawInOutput(d)
}

// Function to draw a input or output)
function drawLayer(d, layerIndex) {
    const x = d.layerSpacing * (layerIndex + 1); // Horizontal position of current layer
    const neurons = d.layerConfig[layerIndex]
    const yStart = (d.height - (d.neuronSpacing * (neurons - 1))) / 2; // Center neurons vertically

    for (let i = 0; i < neurons; i++) {
        const y = yStart + i * d.neuronSpacing;

        // Create a circle for each neuron
        d.svg.append("circle")
            .attr("cx", x)
            .attr("cy", y)
            .attr("r", d.neuronRadius)
            .style("fill", "steelblue")
            .style("stroke", "black")
            .style("stroke-width", d3.randomUniform(1,6));

        // Add neuron labels
        d.svg.append("text")
            .attr("x", x)
            .attr("y", y + 4) // Offset to center the label inside the circle
            .attr("text-anchor", "middle")
            .attr("fill", "white")
            .style("font-size", "10px")
            .text(i + 1);
    }
}

function drawGraph(d, neurons) {
    const enter = d.svg.selectAll("g")
        .data(d.data)
        .enter().append("g")

    enter.append("circle")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", d.neuronRadius)
        .style("fill", "steelblue")
        .style("stroke", "black")
        .style("stroke-width", 1) // d3.randomUniform(1,6));
        // .on("click" , highlight)
        .on("click" , function(event, d1) {
          highlight( event, d1, d.d3);  // `this` is the DOM element
        })
    enter.append("text")
        .attr("x", d=>d.x)
        .attr("y", d=>d.y + 4) // Offset to center the label inside the circle
        .attr("text-anchor", "middle")
        .attr("fill", "white")
        .style("font-size", "10px")
        .text(((d,i)=>i + 1))
        // .on("click" , highlight);
        .on("click" , function(event, d1) {
            highlight(event, d1, d.d3);  // `this` is the DOM element
    })

/*    for (let i = 0; i < neurons.length; i++) {
        // Create a circle for each neuron
        d.svg.append("circle")
            .attr("cx", d.data[i].x)
            .attr("cy", d.data[i].y)
            .attr("r", d.neuronRadius)
            .style("fill", "steelblue")
            .style("stroke", "black")
            .style("stroke-width", 1) // d3.randomUniform(1,6));

        // Add neuron labels
        d.svg.append("text")
            .attr("x", d.data[i].x)
            .attr("y", d.data[i].y + 4) // Offset to center the label inside the circle
            .attr("text-anchor", "middle")
            .attr("fill", "white")
            .style("font-size", "10px")
            .text(i + 1);
    }*/
}

function preventCollisions(data, neuronRadius) {
    const threshold = neuronRadius * 4; // Minimum distance between nodes
    const maxIterations = 100; // Limit iterations to avoid infinite loops
    for (let iteration = 0; iteration < maxIterations; iteration++) {
        let collisions = false;
        for (let i = 0; i < data.length; i++) {
            for (let j = i + 1; j < data.length; j++) {
                const dx = data[i].x - data[j].x;
                const dy = data[i].y - data[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < threshold) {
                    // Nodes are overlapping; adjust positions
                    collisions = true;
                    const overlap = (threshold - distance) / 2;
                    const angle = Math.atan2(dy, dx);
                    // Move nodes apart proportionally
                    data[i].x += Math.cos(angle) * overlap;
                    data[i].y += Math.sin(angle) * overlap;
                    data[j].x -= Math.cos(angle) * overlap;
                    data[j].y -= Math.sin(angle) * overlap;
                }
            }
        }
        if (!collisions) break; // Exit if no collisions remain
    }
}

// Function to draw connections between layers
function drawConnections(d) {
    /*function connect(from,to){
        if (from.layer == "output") return false
        if (to.layer == "input") return false
        if (from.i >= to.i) return false
        let dist = Math.sqrt((from.x-to.x)**2 +(from.y-to.y)**2 )
        dist = dist / d.graphWidth
        // console.log(from, to, dist)
        //return true // fully connected
        return ( Math.random() * dist < d.connectTreshold ) // distance based connections
        // todo assert that every neuron has at least some inputs and some outputs
    }*/
    d.data.forEach(from => {
        d.data.forEach(to =>{
            if (d.connect(from, to)){
                // add connection to the data
                from.to.push(to)
                to.from.push(from)
                d.svg.append("line")
                    .attr("x1", from.x+d.neuronRadius)
                    .attr("y1", from.y)
                    .attr("x2", to.x-d.neuronRadius)
                    .attr("y2", to.y)
                    .style("stroke", from.i >= to.i?"red":"black")
                    .style("stroke-width", from.i >= to.i?1:0.3)
                    .attr("data-from",from.i)
                    .attr("data-to",to.i);
            }
        })
    })

}



function drawInOutput(d) {
    let layer = d.layerConfig[0]
    let x = d.layerSpacing - 50;
    let yStart = (d.height - (d.neuronSpacing * (layer - 1))) / 2;
    if (d.input) {
        // console.log(x + "   " + yStart + lastLayer)
        for (let i = 0; i < layer; i++) {
            let v = d3.randomUniform(0, 1)().toFixed(3)
            if (Array.isArray(d.input)){
                v=d.input[i]
            }
            d.svg.append("text")
                .attr("x", x )
                .attr("y", yStart + i * d.neuronSpacing +5)
                .style("stroke", "black")
                .style("stroke-width", .3)
                .text(v);
        }
    }
    layer = d.layerConfig[d.layerConfig.length - 1]
    x = d.layerSpacing * (d.layerConfig.length) + 30;
    yStart = (d.height - (d.neuronSpacing * (layer - 1))) / 2;
    // console.log(x + "   " + yStart + lastLayer)
    if (d.output) {
        for (let i = 0; i < layer; i++) {
            let v = d3.randomUniform(0, 1)().toFixed(3)
            if (Array.isArray(d.output)){
                v=d.output[i]
            }
            d.svg.append("text")
                .attr("x", x)
                .attr("y", yStart + i * d.neuronSpacing +5)
                .style("stroke", "black")
                .style("stroke-width", .3)
                .text(v);
        }
    }
/*    d.svg.append("path")
        .attr("x", x)
        .attr("y", yStart + 3 * d.neuronSpacing)
        .attr("transform", "translate(50, 50) scale(0.1)")
        .attr("d","M160 48a48 48 0 1 1 96 0 48 48 0 1 1 -96 0zM126.5 199.3c-1 .4-1.9 .8-2.9 1.2l-8 3.5c-16.4 7.3-29 21.2-34.7 38.2l-2.6 7.8c-5.6 16.8-23.7 25.8-40.5 20.2s-25.8-23.7-20.2-40.5l2.6-7.8c11.4-34.1 36.6-61.9 69.4-76.5l8-3.5c20.8-9.2 43.3-14 66.1-14c44.6 0 84.8 26.8 101.9 67.9L281 232.7l21.4 10.7c15.8 7.9 22.2 27.1 14.3 42.9s-27.1 22.2-42.9 14.3L247 287.3c-10.3-5.2-18.4-13.8-22.8-24.5l-9.6-23-19.3 65.5 49.5 54c5.4 5.9 9.2 13 11.2 20.8l23 92.1c4.3 17.1-6.1 34.5-23.3 38.8s-34.5-6.1-38.8-23.3l-22-88.1-70.7-77.1c-14.8-16.1-20.3-38.6-14.7-59.7l16.9-63.5zM68.7 398l25-62.4c2.1 3 4.5 5.8 7 8.6l40.7 44.4-14.5 36.2c-2.4 6-6 11.5-10.6 16.1L54.6 502.6c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L68.7 398z")
 */
}

function highlight(event, d, d3) {
    const link = d3.select(this)
    console.log("click "+d.from.length+" "+d.i)
    const delay = 2000
    d3.selectAll("line[data-from='"+d.i+"']")
        .transition()
        .ease(d3.easeCubicOut)
        .duration(delay)
        .style("stroke", "red")
        .style("stroke-width",2)
        .transition()
        .ease(d3.easeCubicIn)
        .duration(delay)
        .style("stroke", "black")
        .style("stroke-width",0.2)
    d3.selectAll("line[data-to='"+d.i+"']")
        .transition()
        .duration(delay)
        .ease(d3.easeCubicOut)
        .style("stroke", "green")
        .style("stroke-width",2)
        .transition()
        .ease(d3.easeCubicIn)
        .duration(delay)
        .style("stroke", "black")
        .style("stroke-width",0.2)
    /*link.transition()
        .duration(delay)
        .style("fill", "orange")
        .transition()
        .duration(delay)
        .style("fill", "steelblue");

     */
    /*
    original values
    data.forEach(d => {
    d3.selectAll("line[data-from='" + d + "']")
        .each(function() {
            // Save original styles
            const originalStroke = d3.select(this).style("stroke");
            const originalStrokeWidth = d3.select(this).style("stroke-width");
            // Transitions
            d3.select(this)
                .transition()
                .duration(delay)
                .style("stroke", "orange")
                .style("stroke-width", 2)
                .transition()
                .duration(delay)
                .style("stroke", originalStroke)
                .style("stroke-width", originalStrokeWidth);
        });
    });
*/
}
