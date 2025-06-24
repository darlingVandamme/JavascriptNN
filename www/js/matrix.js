
export const matrix = {}

matrix.generate = function(d){
    /*
    d.rows
    d.columns
    */
    d.data = Array.from({ length: d.rows }, () => new Array(d.columns).fill(0));

    if (!d.cellWidth) d.cellWidth = Math.min(d.width*0.8 / d.columns, d.height*0.8 / d.rows)
    d.x = d.x || 50
    d.y = d.y || 50
    d.maxX = d.x+(d.columns*d.cellWidth)
    d.maxY = d.y+(d.rows*d.cellWidth)
    for (let i=0;i<=d.columns;i++){
        let x = d.x + (i*d.cellWidth)
        d.svg.append("line")
            .attr("x1", x)
            .attr("y1", d.y)
            .attr("x2", x)
            .attr("y2", d.maxY)
            .style("stroke", "black")
            .style("stroke-width", 1);
    }
    for (let i=0;i<=d.rows;i++){
        let y = d.y + (i*d.cellWidth)
        d.svg.append("line")
            .attr("x1", d.x)
            .attr("y1", y)
            .attr("x2", d.maxX)
            .attr("y2", y)
            .style("stroke", "black")
            .style("stroke-width", 1);
    }
    for (let i=0;i<d.columns;i++){
        for (let j=0;j<d.rows;j++) {
            let x = d.x + (i*d.cellWidth)+(d.cellWidth/2)
            let y = d.y + (j*d.cellWidth)+(d.cellWidth/2)
            let number = d3.randomUniform(0,1)().toFixed(2)
            d.data[i][j]=number
            // console.log(x+" "+y+" "+number)
            d.svg.append("text")
                .attr("x", x)
                .attr("y", y)
                .attr("text-anchor", "middle")
                .attr("dy", "0.35em")          // Adjust for vertical alignment
                .style("font-size", "12px")
                .style("stroke", "black")
                .style("stroke-width", 0.3)
                .text(number)

        }
    }
}