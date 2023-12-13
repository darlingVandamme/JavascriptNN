import Neuron from "./neuron.js";

class NNet{
    constructor(inputSize){
        this.layers = 0
        this.step = 3.0
        this.batchSize = 10
        this.trainings = 0
        this.batches =0
        this.count = 0
        this.patterns = {}
        // keep a list of neurons
        // makes it easier to reset and serialize
        this.neurons = [] // Hidden and output Neurons
        this.allNeurons = []
        this.output = []
        this.input = this.addLayer(inputSize)
        this.costs = []
        this.costIndex=0
        this.costsSize = 500
        this.bestCost = 1;
        this.bestNetwork
    //    this.calculateCosts = true
    }

    // clone
    /*
     - copy input layer
     - copy Neurons
     - set output to last x neurons
     */
    clone(){
        // todo reverse? Copy from existing
        // expensive operation
        let clone = new NNet(this.input.length)
        clone.neurons = this.neurons.map(n=>{
            let c = new Neuron()
            c.bias=n.bias
            c.id=n.id
            c.index = n.index
            clone.allNeurons.push(c)
            c.connect( clone.allNeurons, n.getConnectionIDs(), n.getWeights())
            return c
        })
        clone.output = clone.neurons.slice(- this.output.length)
        clone.step = this.step
        clone.batchSize = this.batchSize
        clone.trainings = this.trainings
        clone.batches = this.batches
        clone.count= this.count
        clone.patterns = this.patterns
        clone.costs = this.costs
        clone.costIndex = this.costIndex
        clone.costsSize = this.costsSize
        clone.bestCost = this.bestCost
        //console.log("cloned network ",clone.input.length, clone.allNeurons.length, clone.neurons,clone.output.length )
        return clone
    }

    addLayer(size){  // weight[][] and bias[] optional for serialization / deserialization
        let position = 0
        let layer = Array.from({length:size}, (n1,i)=>{
            let n = new Neuron()
            n.position = position++
            n.id=this.allNeurons.length
            n.index = i
            this.allNeurons.push(n)
            /*if (this.output.length>0){
                this.neurons.push(n)
            }*/
            this.output.forEach((other)=> n.connect(other))
            return n
        })
        this.neurons.push(...layer.filter(n=> !n.isInput())) // add all non input neurons
        this.output = layer
        this.layers++
        return layer
    }

    reset(){this.neurons.forEach((n, i)=>{n.reset()})}

    feed(input){
        this.count++
        this.reset()
        //counters["feed"]++
        this.input.forEach((n,i)=> {n.value = input[i]})
        this.neurons.forEach((n, i)=>{n.ff()})
        //for (let i=0;i<this.neurons.length;i++){this.neurons[i].ff()}
    }

    getOutput(){
        return this.output.map(n=>n.getValue())
        /*let result = new Array(this.output.length)
        for(let i=0;i<this.output.length;i++){
            result[i]= (this.output[i].getValue())
        }
        return result*/
    }

    // convenience methods
    translateInput(value){
        // allow translate of any object to list of values
        // default do nothing
        return value
    }

    translateOutput(value){
        // allow translate of list of values to object
        // default do nothing
        return value
    }

    check(item){
        this.feed(this.translateInput(item))
        return this.translateOutput(this.getOutput())
    }

    //getCost(result)
    getCost(){
        return this.output.reduce((prev,n,i)=> (prev + (Math.pow((n.expected - n.value),2  ) / (2*this.output.length))) , 0)
        // return this.output.reduce((prev,n,i)=> (prev + (Math.pow((n.expected - n.value),2  ) / (this.output.length))) , 0)
    }
    getAverageCost(length){
        // average cost of the most recent trainins
        let slice
        if(!length){
            slice = this.costs
        } else {
            slice = this.costs.slice(Math.max(0,this.costIndex-length), this.costIndex+1)
            if (this.costIndex<length) {
                slice = [...slice, ...this.costs.slice(this.costIndex - length)]
            }
        }
        return slice.reduce((prev,cost)=>{return prev+cost},0)/slice.length
    }

    getResult(pattern){
        // todo cleaner
        let result = null
        let best = 0
        Object.keys(this.patterns).forEach(k=>{
            let p = this.patterns[k]
            // 1 - squared errors 
            let score = 1 - (p.pattern.reduce((prev,item,i)=>{
                return prev + Math.pow((item-pattern[i]),2)},0.0) / p.pattern.length)
            // console.log("check "+k+" "+score)
            if (score > best) {
                best = score
                result = {label:p.label,score:score, pattern:p.pattern,trained:p.count}
            }
        })
        return result
    }

    getHighest(pattern){
        /*let max = Math.max(...pattern)
        return pattern.indexOf(max)
         */
        return pattern.reduce((prev,val,i,arr)=> val>arr[prev]?i:prev,0)
    }

    train(item, expected, label){
        if (label != null && label != undefined){
            let pattern = this.patterns[label]
            if (!pattern) {
                pattern = { label:label, count:0, pattern:expected}
                this.patterns[label] = pattern
            }
            pattern.count++
        }
        let output = this.check(item)

        // store the expected values in the output neurons
        this.output.forEach((n,i)=> {n.expected = expected[i] })
        this.costIndex = this.trainings % this.costsSize
        this.costs[this.costIndex] = this.getCost()
        this.trainings ++

        // backpropagate Delta recursive
        //this.neurons.forEach((n,i)=>n.getDelta())

        // backpropagate delta iterative  backwards
        for(let i=this.neurons.length-1 ; i>=0; i--){
            this.neurons[i].getDelta()
        }

        // adjust weights
        if (this.trainings % this.batchSize == 0){
            this.batches++
            //let cost = this.getCost(result)
            // calculate batchCost?

            // recursive
            //this.input.forEach((n,i)=>n.learn(this.step))
            // iterative
            this.neurons.forEach((n, i)=>n.learn(this.step/this.batchSize))
            //console.log(" Cost  \t"+this.trainings+" \t "+this.getCost())
            // keep a reference to the best network up till now
            /*if (this.getAverageCost(100 ) < this.bestCost ){
                this.bestCost = this.getAverageCost(100)
                this.bestNetwork = this.clone();
                console.log("Clone network "+this.bestCost)
            }*/
        }
    }

    toJSON(){
        // recent cost functions or success rate
        // trainings
        // patterns?

        return {
            inputSize: this.input.length,
            outputSize: this.output.length,
            allNeurons: this.allNeurons.map(n => {
                return {
                    bias: n.bias,
                    id: n.id,
                    index: n.index,
                    connectionIDs: n.getConnectionIDs(),
                    weights: n.getWeights()
                }
            })
        }
    }
    /*printInfo(){
        console.log("counts "+counts)
    }*/

}


export default NNet