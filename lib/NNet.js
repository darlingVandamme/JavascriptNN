import Neuron from "./neuron.js";

class NNet{
    constructor(inputSize){
        this.input = new Array(inputSize).fill(0).map((n1,i)=>{
            return new Neuron()
        })
        this.output = this.input
        this.layers = 1
        this.step = 2
        this.batchSize = 10
        this.trainings = 0
        this.count = 0
        this.patterns = {}
        // keep a list of all neurons
        // makes it easier to reset and serialize
        this.neurons = []
        this.costs = []
        this.costsSize = 500
        this.calculateCosts = true
    }

    addLayer(size){
        let layer = new Array(size).fill(0).map((n1,i)=>{
            let n = new Neuron()
            this.neurons.push(n)
            n.connect(this.output)
            return n
        })
        this.output = layer
        this.layers++
    }

    translateInput(value){
        // allow translate of any object to list of values
        return value
    }

    reset(){this.neurons.forEach((n,i)=>{n.reset()})}

    feed(v){
        this.count++
        this.reset()
        let values=this.translateInput(v)
        this.input.forEach((n,i)=>{n.value=values[i]})
    }

    getOutput(){
        return this.output.map(n=>n.getValue())
    }

    check(item){
        this.feed(item)
        return this.getOutput()
    }

    //getCost(result)
    getCost(){
        return this.output.reduce((prev,n,i)=> (prev + (Math.pow((n.expected - n.value),2  ) / (2*this.output.length))) , 0)
        // return this.output.reduce((prev,n,i)=> (prev + (Math.pow((n.expected - n.value),2  ) / (this.output.length))) , 0)
    }
    getAverageCost(){
        // average cost of the most recent trainins
        return this.costs.reduce((prev,cost)=>{return prev+cost},0)/this.costs.length
    }

    getResult(pat){
        let result = null
        let best = 0
        Object.keys(this.patterns).forEach(k=>{
            let p = this.patterns[k]
            // 1 - squared errors 
            let score = 1 - (p.pattern.reduce((prev,item,i)=>{
                return prev + Math.pow((item-pat[i]),2)},0.0) / p.pattern.length)
            // console.log("check "+k+" "+score)
            if (score > best) {
                best = score
                result = {label:p.label,score:score, pattern:p.pattern,trained:p.count}
            }
        })
        return result
    }

    train(v, expected, label){
        if (label){
            let pattern = this.patterns[label]
            if (!pattern) {
                pattern = { label:label, count:0, pattern:expected}
                this.patterns[label] = pattern
            }
            pattern.count++
        }
        this.feed(v)
        let output = this.getOutput()

        // store the expected values in the output neurons
        this.output.forEach((n,i)=> {n.expected = expected[i] })
        if (this.calculateCosts){
            let index = this.trainings % this.costsSize
            this.costs[index] = this.getCost()
        }
        this.trainings ++

        // backpropagate error in a reverse way
        //this.input.forEach((n,i)=>n.getError())

        // backpropagate error the normal way
        for(let i=this.neurons.length-1 ;i>=0;i--){
            this.neurons[i].getError()
        }

        // adjust weights
        if (this.trainings % this.batchSize == 0){
            this.batches++
            //let cost = this.getCost(result)

            //this.input.forEach((n,i)=>n.learn(this.step))
            this.neurons.forEach((n,i)=>n.learn(this.step))
            //console.log(" Cost  \t"+this.trainings+" \t "+cost)
        }
        // adjust weights & Biases
    }

    toJSON(){

    }

}


export default NNet