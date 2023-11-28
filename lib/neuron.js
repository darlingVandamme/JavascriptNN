let neurons = 0

class Neuron {

    constructor(){ // link to network?
        this.id = (neurons++)
        this.in = []
        this.out = []
        this.activate = sigmoid
        this.actDeriv = sigmoidDeriv
        this.value = null
        this.delta = null
        this.deltaSum = 0
        this.deltaN = 0
        this.bias = getRandom()
        this.z = 1
        this.layer = 0
        //init default values, connections
    }

    reset(){
        //if (this.value !=null) {
            //console.log("reset "+this.id)
            this.value = null
            this.delta = null
            // Propagate the reset
            // this.out.forEach(c => c.out.reset())
        //}
    }

    connect(other){
        this.in = other.map(other=>{
            let conn = { in:other, out:this, weight:getRandom() , delta: 0}
            this.layer = other.layer+1
            other.out.push(conn)
            return conn
        })
    }

    ff(){
        this.z = this.in.reduce((prev, conn)  => {
            return prev + ( conn.weight * conn.in.getValue())
        },   this.bias)
        /*
        this.z = this.bias;
        for (let i=0;i<this.in.length;i++) {
            let conn = this.in[i]
            this.z +=  ( conn.weight * conn.in.getValue())
        }*/
        /*
        this.z = this.bias;
        for (let conn of this.in) {
            this.z +=  ( conn.weight * conn.in.getValue())
        }*/

        this.value = this.activate(this.z) // sigmoid
        return this.value

    }

    getValue(){
        if (this.value == null){
            //this.value=0
            this.ff()
        }
        return this.value
    }

    getDelta(){
        // lazy calculate
        if (this.delta == null) {
            //let zDeriv =   this.value * (1-this.value)//
            let zDeriv = this.actDeriv(this.z)
            if (this.isOutput() ){
                this.delta = ((  this.value - this.expected )  * zDeriv )
            } else if (this.isInput() ){
                // no need to calculate the error for the input layer
                // only propagate  in recursive setup
                //this.out[0].out.getError()
            } else{
              // other neurons
                // actDeriv inside reduce???
                this.delta = this.out.reduce((prev, conn) => {
                    return prev + (conn.out.getDelta() * conn.weight )
                }, 0) * zDeriv // * this.actDeriv(this.z)
            }
            this.in.forEach((c, i) => {
                c.delta += this.delta * c.in.value
            })
            this.deltaN++
            this.deltaSum += this.delta
        }
        return this.delta
    }

    getAvgDelta(){
        return this.deltaSum / this.deltaN
    }

    learn(step,batchSize) {
        if (this.deltaN > 0) {
            step = step / batchSize
            //console.log("avgError "+this.getAvgError()+"   "+this.errorSum+"   "+this.errorN)
            if (!this.isInput()) {
                // adjust bias
                this.bias -= step * this.deltaSum
                // adjust input weights
                this.in.forEach((c, i) => {
                    c.weight -= step * c.delta
                    c.delta=0
                })
            }
            //recursive
            /*this.out.forEach((c, i) => {
                c.out.learn(step)
            })*/
            // reset delta
            this.deltaN = 0
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

function getRandom(){
    // return (Math.random()*4)-2

    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0
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