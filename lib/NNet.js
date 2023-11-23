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
    }

    addLayer(size){
        let layer = new Array(size).fill(0).map((n1,i)=>{
            let n = new Neuron()
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

    reset(){this.input.forEach((n,i)=>{n.reset()})}

    feed(v){
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

    getCost(result){
        return this.output.reduce((prev,n,i)=> (prev + (Math.pow((n.expected - n.value),2  ) / (2*this.output.length))) , 0)
        // return this.output.reduce((prev,n,i)=> (prev + (Math.pow((n.expected - n.value),2  ) / (this.output.length))) , 0)
    }

    train(v,result){
        this.trainings ++
        this.feed(v)
        let output = this.getOutput()
        // console.log("output ",output)
        // store the expected values in the output neurons
        this.output.forEach((n,i)=> {n.expected = result[i] })

        // backpropagate error in a reverse way
        this.input.forEach((n,i)=>n.getError())
        // adjust weights
        if (this.trainings % this.batchSize == 0){
            this.batches++
            let cost = this.getCost(result)
            this.input.forEach((n,i)=>n.learn(this.step))
            console.log(" Cost  \t"+this.trainings+" \t "+cost)

            // resetErrors
        }
        // adjust weights & Biases
    }

    toJSON(){

    }

}


export default NNet