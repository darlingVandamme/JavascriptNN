let neurons = 0

class Neuron {

    constructor(){ // link to network?
        this.id = (neurons++)
        this.in = []
        this.out = []
        this.activate = sigmoid
        this.actDeriv = sigmoidDeriv
        this.value = null
        this.error = null
        this.errorSum = 0
        this.errorN = 0
        this.bias = getRandom()
        this.z = 1
        this.layer = 0
        //init default values, connections
    }

    reset(){
        // better option possible???
        if (this.value !=null) {
            //console.log("reset "+this.id)
            this.value = null
            this.error = null
            this.out.forEach(c => c.out.reset())
        }
    }

    connect(other){
        this.in = other.map(other=>{
            let conn = { in:other, out:this, weight:getRandom() }
            this.layer = other.layer+1
            other.out.push(conn)
            return conn
        })
    }

    ff(){
        this.z = this.in.reduce((prev, conn)  => {
            return prev + ( conn.weight * conn.in.getValue())
        },  this.bias)
        this.value = this.activate(this.z) // sigmoid
        return this.value
    }

    getValue(){
        if (this.value == null){
            //console.log("getValue "+this.id)
            this.value=0
            this.ff()
        }
        return this.value
    }

    getError(){
        // lazy calculate
        if (this.error == null) {
            // if (this.expected  == null ){
            // output neuron
            if (this.out.length  == 0 ){
                this.error = (( this.value-this.expected )  * this.actDeriv(this.z) )
            } else {
              // other neurons
                this.error = this.out.reduce((prev, conn) => {
                    return prev + (conn.out.getError() * conn.weight)
                }, 0) * this.actDeriv(this.z)
            }
            this.errorN++
            this.errorSum += this.error
        }
        return this.error
    }

    getAvgError(){
        if (this.errorN == 0) return 0 //and errorSum = 0
        return this.errorSum / this.errorN
    }

    learn(step) {
        // adjust for minibatch
        //let m=1
        //step = step/m
        //console.log("avgError "+this.getAvgError()+"   "+this.errorSum+"   "+this.errorN)
        this.bias -= step * this.getAvgError()
        this.in.forEach( (c,i) => { c.weight -= step * this.getAvgError() * c.in.value })
        this.out.forEach( (c,i) => {  c.out.learn(step) })
        // reset errors
        this.errorN = 0
        this.errorSum = 0
        this.error = null
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

    /*
    - list of incoming connections
    - list of weights
    - bias
    - errors
    - z
    - activationF
    - a


    - calcValue
    - backtrack (learn)

    serialize - deserialize
    - id
    - write to json structure
    - read from json structure
     */


}

function getRandom(){
    return (Math.random()*4)-2
}

function sigmoid(x){
    return 1 / (1 + Math.exp(-x));
}
function sigmoidDeriv(x){
    //if (!x) return 1
    let s = sigmoid(x)
    //console.log(" sigDer "+x+" / "+s+" / "+)
    return s * (1-s);
}


export default Neuron