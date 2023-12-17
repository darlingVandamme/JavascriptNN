import {count} from "./counter.js"
let neurons = 0

class Neuron {

    constructor(){ // link to network?
        this.id = (neurons++)
        this.index = 0
        this.in = []
        this.out = []
        this.activate = sigmoid
        this.actDeriv = sigmoidDeriv
        this.value = null
        this.delta = null
        this.deltaSum = 0.0
        this.bias = getRandom()
        this.z = 1.0

        this.position = 0
    }

    reset(){
        this.value = null
        this.delta = null
        //this.out.forEach(c => c.out.reset())
    }

    connectLayer(layer,weights){
        if (!weights) weights = []
        layer.forEach((n,i)=>this.connect(n,weights[i]))
    }
    connect(other,weight){
        new Connection(other, this, weight )
    }

    setWeights(weights){
        weights.forEach((w,i)=>{this.in[i].weight = w})
    }

    getWeights(){
        return this.in.map((c)=>c.weight)
    }

    getConnectionIDs(){
        return this.in.map((c) => c.in.id);
    }

    ff(){
        this.z = this.in.reduce((prev, conn) => prev + (conn.weight * conn.in.getValue())
        , this.bias)
        //count("ff")

        //console.log(" z "+this.z+"  "+this.bias);
/*
        this.z = this.bias;
        for (let i=0;i<this.in.length;i++) {
            let conn = this.in[i]
            this.z +=  ( conn.weight * conn.in.getValue())
        }
*/
        /*this.z = this.bias;
        for (let conn of this.in) {
            this.z +=  ( conn.weight * conn.in.getValue())
        }*/
        /*this.z = this.bias
        this.in.forEach(conn=> {
            this.z +=  ( conn.weight * conn.in.getValue())
        })
         */
        this.value = this.activate(this.z) // sigmoid
        //return this.value
    }

    getValue(){
        //count("getvalue")
        if (this.value === null){
            this.ff()
        }
        return this.value
    }

    getDelta(){
        if (this.isInput() ) return 0 // ?
        // lazy calculate
        if (this.delta === null) {
            let zDeriv = this.actDeriv(this.z)
            if (this.isOutput() ) {
                this.delta = ((this.value - this.expected) * zDeriv)
            } else{
              // other neurons
                this.delta = this.out.reduce((prev, conn) => {
                    return prev + (conn.out.getDelta() * conn.weight )
                }, 0) * zDeriv //
            }
            this.in.forEach((conn, i) => {
                conn.setDelta(this.delta)
            })
            this.deltaSum += this.delta
        }
        return this.delta
    }

    learn(batchStep) {
        if (this.delta) {
            if (!this.isInput()) {
                // adjust bias
                this.bias -= batchStep * this.deltaSum
                // adjust input weights
                this.in.forEach((conn, i) => {
                    conn.learn(batchStep)
                })
            }
            //recursive
            /*this.out.forEach((c, i) => {
                c.out.learn(step)
            })*/
            // reset delta
            this.deltaSum = 0
            this.delta = null
        }
    }

    /*
    Extra info methods
     */

    isInput(){
        return this.in.length == 0
        // return this.layer == 0
    }

    isOutput(){
        return this.out.length == 0
    }

    isHidden(){
        return (this.out.length > 0 && this.in.length > 0);
    }
    /*

    serialize - deserialize
    - write to json structure
    - read from json structure
     */
    /*printInfo(){
        console.log("counts "+counts)
    }
*/

}

class Connection {
    /**
     * Constructor method for creating a new connection.
     * @param {Object} input - The input node of the connection.
     * @param {Object} output - The output node of the connection.
     * @param {Number} weight - The weight of the connection. If not provided, a random weight will be generated.
     */
    constructor(input , output, weight){
        this.in = input
        this.out = output
        this.weight = weight || getRandom()
        this.deltaSum=  0.0
        input.out.push(this)
        output.in.push(this)
    }
    setDelta(delta){
        this.deltaSum += delta * this.in.getValue()
    }
    learn(batchStep){
        this.weight -= batchStep * this.deltaSum
        this.deltaSum=0
    }
}


function getRandom(){
    // return (Math.random()*4)-2
    // normal distribution
    const u1 = Math.random();
    const u2 = Math.random();
    let dev = 1
    return  ( Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) / dev );
}

function sigmoid(x){
    return 1.0 / (1.0 + Math.exp(-x));
}
function sigmoidDeriv(x){
    //if (!x) return 1
    let s = sigmoid(x)
    return s * (1-s);
}


export default Neuron