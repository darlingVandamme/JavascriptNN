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
        }
        return this.error
    }

    learn(step) {
        // adjust for minibatch
        let m=1
        step = step/m
        this.bias -= step * this.error
        this.in.forEach( (c,i) => { c.weight -= step * this.error * c.in.value })
        this.out.forEach( (c,i) => {  c.out.learn(step) })
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