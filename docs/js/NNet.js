
export const NNet = {}

NNet.generate = function(d) {

    // Draw layers
    d.layerConfig.forEach((neurons, index) => {
        drawLayer(d, index);
    });
    // Connect neurons across layers
    drawConnections(d);
    drawInOutput(d)
}

// Function to draw a single layer (input, hidden, or output)
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

// Function to draw connections between layers
function drawConnections(d) {
    for (let layerIndex = 0; layerIndex < d.layerConfig.length - 1; layerIndex++) {
        d.layerIndex = layerIndex
        const currentLayerNeurons = d.layerConfig[layerIndex];
        const nextLayerNeurons = d.layerConfig[layerIndex + 1];

        const x1 = d.layerSpacing * (layerIndex + 1)+d.neuronRadius;
        const x2 = d.layerSpacing * (layerIndex + 2)-d.neuronRadius;

        const yStart1 = (d.height - (d.neuronSpacing * (currentLayerNeurons - 1))) / 2;
        const yStart2 = (d.height - (d.neuronSpacing * (nextLayerNeurons - 1))) / 2;

        for (let i = 0; i < currentLayerNeurons; i++) {
            for (let j = 0; j < nextLayerNeurons; j++) {
                const y1 = yStart1 + i * d.neuronSpacing;
                const y2 = yStart2 + j * d.neuronSpacing;
                if (!(d.connectionFilter) || (d.connectionFilter(d,i,j))){
                        // Draw a line connecting neuron (i) in this layer to neuron (j) in the next
                        d.svg.append("line")
                            .attr("x1", x1)
                            .attr("y1", y1)
                            .attr("x2", x2)
                            .attr("y2", y2)
                            .style("stroke", "black")
                            .style("stroke-width", 1);
                }
            }
        }
    }
}

function drawInOutput(d) {
    let layer = d.layerConfig[0]
    let x = d.layerSpacing - 80;
    let yStart = (d.height - (d.neuronSpacing * (layer - 1))) / 2;

    if (d.input) {
        // console.log(x + "   " + yStart + lastLayer)
        for (let i = 0; i < layer; i++) {
            d.svg.append("text")
                .attr("x", x )
                .attr("y", yStart + i * d.neuronSpacing +5)
                .style("stroke", "black")
                .style("stroke-width", .3)
                .text(d3.randomUniform(0, 1)().toFixed(3));
        }
    }
    layer = d.layerConfig[d.layerConfig.length - 1]
    x = d.layerSpacing * (d.layerConfig.length) + 30;
    yStart = (d.height - (d.neuronSpacing * (layer - 1))) / 2;
    // console.log(x + "   " + yStart + lastLayer)
    if (d.output) {
        for (let i = 0; i < layer; i++) {
            d.svg.append("text")
                .attr("x", x)
                .attr("y", yStart + i * d.neuronSpacing +5)
                .style("stroke", "black")
                .style("stroke-width", .3)
                .text(d3.randomUniform(0, 1)().toFixed(3));
        }
    }
/*
    d.svg.append("path")
        .attr("x", x)
        .attr("y", yStart + 3 * d.neuronSpacing)
        .attr("transform", "translate(50, 50) scale(0.1)")
        .attr("d","M160 48a48 48 0 1 1 96 0 48 48 0 1 1 -96 0zM126.5 199.3c-1 .4-1.9 .8-2.9 1.2l-8 3.5c-16.4 7.3-29 21.2-34.7 38.2l-2.6 7.8c-5.6 16.8-23.7 25.8-40.5 20.2s-25.8-23.7-20.2-40.5l2.6-7.8c11.4-34.1 36.6-61.9 69.4-76.5l8-3.5c20.8-9.2 43.3-14 66.1-14c44.6 0 84.8 26.8 101.9 67.9L281 232.7l21.4 10.7c15.8 7.9 22.2 27.1 14.3 42.9s-27.1 22.2-42.9 14.3L247 287.3c-10.3-5.2-18.4-13.8-22.8-24.5l-9.6-23-19.3 65.5 49.5 54c5.4 5.9 9.2 13 11.2 20.8l23 92.1c4.3 17.1-6.1 34.5-23.3 38.8s-34.5-6.1-38.8-23.3l-22-88.1-70.7-77.1c-14.8-16.1-20.3-38.6-14.7-59.7l16.9-63.5zM68.7 398l25-62.4c2.1 3 4.5 5.8 7 8.6l40.7 44.4-14.5 36.2c-2.4 6-6 11.5-10.6 16.1L54.6 502.6c-12.5 12.5-32.8 12.5-45.3 0s-12.5-32.8 0-45.3L68.7 398z")
 */
}
