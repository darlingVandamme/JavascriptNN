# Neural Net OO Javascript implementation
Although I learned the underlying basics of neural networks
since the early 1990's and already did some experiments with AI and ML. 
With the recent wave of new AI tools appearing everywhere, I wanted to really know and understand the low-level details. 
3blue1brown
Michael Nielsen explains this quite nice in his [online book](http://neuralnetworksanddeeplearning.com).
But I found the accompanying code really bad. 
The python code (of course python, what else...) is difficult to read, and doesn't reflect what the concept of a neural network really is.
A neural net is such a nice example to fit object-oriented programming, but almost all examples ...

I did a quick google search to see if I could find better, nicer looking implementations, but I was quite disappointed.
Some [articles](https://towardsdatascience.com/understanding-and-implementing-neural-networks-in-java-from-scratch-61421bb6352c) 
and [videos](https://www.youtube.com/watch?v=NWQETJz8fII) have promising titles, but as soon as you see 
first lines of code, you know that that is absolutely no clean OO implementation. 
This feeling is 'enhanced' when the article concludes 
> Being a fellow Python programmer I was not very comfortable with Java

I didn't look any further, there must be some good implementation examples somewhere, but instead decided to try it myself and share it to you all. I hope you like it.

## Like we're living in the matrix 
Most implementations of a neural network implement the calculations as matrix and vector operations.
Some people think that's nice and convenient and allows us to use speed-optimized libraries to do the calculations. 
But it doesn't exactly make the code transparent or readable. These articles have the intention to teach us something.
The code should be an illustration or a clarification of the mathematical and computational theory.
But in most cases the code reflects an alternate, mostly matrix oriented, view on the needed algorithms.

[Code example]

The program doesn't represent a neural network like it is explained in all the fancy pictures.
It models something that can be shown to work mathematically equivalent to a neural network.
I don't want an illustration of an alternative, which can be shown to be mathematically equivalent. I want to see the real thing. The working network.

## Performance
I know, I know, performance!!! 
We really need to use vectors and matrices so that we can use all the fancy libraries that do the magic. 
As if neural networks need more, obscure, hocus pocus than there already is. 
How and especially why neural networks work is already a mystery, even to most of the people programming and using them. 
We don't need an extra layer of black box obscurity.

The code examples in these articles have an educational, illustrative purpose. They shouldn't focus on performance.
There's a common belief or narrative that you absolutely need to follow the matrix route to do the neural calculations.

A very important programming guideline is `premature optimization is the source of all evil`. 
Using matrix calculations in introductory articles and tutorials on AI is a perfect example of premature optimization.
It's perfectly possible that the matrix approach (and GPU's oh yeah, GPU's an extra obscurity layer, everyone seems to love) 
is necessary to allow the large scale AI tools we see popping up everywhere the last year.
But actually, we simply don't know. It might well be possible that the matrix approach forces us in a strict .... that poses its own limitations and boundaries
and these are currently not questioned. The matrix calculations, besides giving us a supposed speed boost, also limit our creativity....
To use the typical neural lingo, the matrix approach 'has' the danger of being stuck in a local minimum in our search to solving real world problems.
Anyway, for the rest of the article I'll try to avoid the performance issue and focus on programming elegance and clarity. 
I'll focus on the (not premature) performance tuning in a next article. Spoiler: the matrix approach doesn't seem to be that much faster after all.

## The python trap
One more paragraph of critique and negativity, before we turn ....
A lot of the AI/ML applications are implemented in python. 
The unfortunate thing is that python actually is a [bad programming language](https://medium.com/nerd-for-tech/python-is-a-bad-programming-language-2ab73b0bda5) and 
bad for the [environment](https://greenlab.di.uminho.pt/wp-content/uploads/2017/10/sleFinal.pdf).
Well, apart from the remarks made in the article, I mainly think that python enables or encourages you to use bad programming habits.
If we look at the code examples from neuralnetworksanddeeplearning, apart from some strange structural choices, bad encapsulation and 
supposed but unnecessary speed optimizations, what struck me most is that the basic routine of the network, the feed forward, is implemented twice.
Once as the feedforward routine itself and again inside the backprop routine. 
This is a clear violation of the DRY principle. Especially because those few lines are the absolute center of the whole Neural Network. 
The biggest problem with this duplicated code is that a change in one place is not always replicated in the other location and could lead to difficult to find bugs. 
I also found it a bit strange that the Cost function itself, while being such a crucial part of the training process, isn't even implemented at all. Not even as an illustration.

## OO Javascript implementation
So, how did I implement this Neural Network? I won't explain the workings of the network or the meaning of the different concepts and terminology. 
You can read that in [Nielsen's book](http://neuralnetworksanddeeplearning.com). To keep things comparable, I will be focussing on the exact same example network as in the first 2 chapters of the book.
A completely connected network with 1 hidden layer (30) and 10 output neurons to recognize written numbers, trained on the [MNIST dataset](https://yann.lecun.com/exdb/mnist/).
(There seems to be something wrong with the original website, but you can download the dataset at several other locations)

At first, I'll implement this network in javascript without any other dependencies, with a focus on clean, readable and nicely encapsulated code. 
As already mentioned, we won't be focussing on performance, but it might help us in certain implementation decisions, as long as it doesn't interfere with the readability.
In the various loops and propagations, we will be mixing a recursive and an iterative approach, depending on what's most obvious or clean.
Later on, we might sometimes change this approach if it seems to be to slow or using too many useless function calls. 
Also, we prefer to use the dedicated array looping functions like forEach, map and reduce over old-fashioned for() iteration.   

Let's first focus on the main structure and the feed forward algorithm.
Obviously, at the basis of the code is the Neuron class. Well, obviously? Then why do most sample implementations don't even have a Neuron class???
Anyway, we'll use a bottom up approach and start with the Neuron itself.
The Neuron class keeps the values (value and z), the neurons bias and a list of connections to the previous and next layer.

```javascript
let neurons = 0

class Neuron {
    constructor() {
        this.id = (neurons++)
        this.in = []
        this.out = []
        this.activate = sigmoid
        this.value = null
        this.bias = getRandom()
        this.z = 1.0
        this.layer = 0
    }

    reset() {
        this.value = null
        this.delta = null
    }

    connect(layer) {
        this.in = layer.map(other => {
            let conn = new Connection(other, this)
            this.layer = other.layer + 1
            return conn
        })
    }

    ff() {
        this.z = this.in.reduce((prev, conn) => {
            return prev + (conn.weight * conn.in.getValue())
        }, this.bias)
        this.value = this.activate(this.z) // sigmoid
    }

    getValue() {
        if (this.value === null) {
            this.ff()
        }
        return this.value
    }
}
```


The connect function creates the (fully connected) network structure by connecting the Neuron to the previous layer by using a Connection object.  
You could code this with an implicit Connection (or should we call it Synapse), but, especially when we add the training code, it's cleaner if we also code the Connection class separately. 
```javascript
class Connection {
    constructor(input, output) {
        this.in = input
        this.out = output
        this.weight = getRandom()
        input.out.push(this)
    }
}
```
If a neuron is in reset state (value === null) ff() will calculate the weighted sum of the values of the previous layer (this.in) and add the bias value of the neuron.
That weighted sum is stored in this.z, after which a so-called activation function is applied to get the output value of the neuron. 
The getValue function checks if the value for this neuron is already calculated and if not, calls the ff function.
At first this mechanism might look a bit strange and backwards to people who are used to the more classical forward moving implementation.
Later on, we will show that this backwards approach can be changed to a forward one (with a certain extra speed gain) by simply adding a single line of code. 
But I do think this implementation has a certain beauty in it, that might become clear when we try out some more experimental designs.

There are several possible ways to calculate this weighted sum, but for now we think the reduce makes the code quite elegant. 
Since the getValue and ff functions can be called millions of times, every small speed gain in these functions can speed up the overall network dramatically.    

The last class we need is the Network itself to keep the whole structure and provide the input and output functions. 
I already mentioned that most implementations put everything in the Network class and don't even implement a Neuron (and certainly no Connection).
This is exactly what I mean with 'clean encapsulation'. Here we put the variables and methods in the class where they belong.
Clean code and proper encapsulation leads to better testability and more maintainable code. (move?)
```javascript
import Neuron from "./neuron.js";

class NNet {
    constructor(inputSize) {
        this.input = Array.from({length: inputSize}, (n1, i) => {
            return new Neuron()
        })
        this.output = this.input
        this.layers = 1
        this.count = 0
        // keep a list of all neurons
        // makes it easier to reset and serialize
        this.neurons = []
        this.allNeurons = (function () {
            return [...this.input, ...this.neurons]
        })
    }

    addLayer(size) {
        let layer = Array.from({length: size}, (n1, i) => {
            let n = new Neuron()
            this.neurons.push(n)
            n.connect(this.output)
            return n
        })
        this.output = layer
        this.layers++
    }
    
    reset() {
        this.neurons.forEach((n, i) => {
            n.reset()
        })
    }

    feed(input) {
        this.count++
        this.reset()
        this.input.forEach((n, i) => {
            n.value = input[i]
        })
    }

    getOutput() {
        return this.output.map(n => n.getValue())
    }

    // convenience methods
    translateInput(value) {
        // allow translate of any object to list of values
        // default do nothing
        return value
    }

    translateOutput(value) {
        // allow translate of list of values to object
        // default do nothing
        return value
    }
    
    check(item) {
        this.feed(this.translateInput(item))
        return this.translateOutput(this.getOutput())
    }
}
```
The NNet object mainly keeps an input and an output array of Neurons and some counters and network parameters.
It's possible to have the network with only the input and output and no direct access to the neurons in hidden layers, 
but it's easier for routines like reset if we do keep a list of all the neurons. 
- this.neurons lists the neurons of the hidden and output layers.
- this.allNeurons is a convenience method that gives a list of all the neurons of the whole network.

When we create a new Network we only create the input layer and make the output layer also point to the input layer. 
addLayer (you might have guessed) adds a new layer to the network, fully connected to the previous last layer. 
The last layer added is automatically the output layer.
At the low level, a feed forward is done by:
- resetting all neurons
- sending an array (with the size of the input layer) values of the neurons of the input layer
- reading the values of the output layer (getOutput) which recursively sets all the values of the network.

To decorate this a bit, there are 2 methods that default to 'do nothing' but can be used to transform an input object to a correct input array (translateInput)
or to interpret the meaning of the output array to a meaningful result.
In our example with the handwritten numbers, translateInput could convert an image to a predefined size and convert the 0-255 grayscale integers to an array of 0-1 doubles.
TranslateOutput can be used to identify the largest value in the output vector and return that index as the final answer of the network.

The check methods simply combines translateInput -> feed -> getOutput -> translateOutput.
[add illustration]

Now the neural network is in theory fully functional, but we cannot show this, because there's no training mechanism or an option to load predefined networks. 

[todo: pretrained classifier]

## Training code
Training the network means gradually adjusting all the neurons' biases and all the connections' weights until the whole network gives us the desired output for a specific input.
So, after training, we add the grayscale pixel values of a 28x28pixels image of a '3', we ideally expect [0,0,0,1,0,0,0,0,0,0] as an output. 
In reality the output might look more like [0.000,0.001,0.001,0.919,0.000,0.004,0.000,0.000,0.000,0.000] but that's fine as well, since the fourth value clearly stands out against all others.

We can train our network by feeding it (lets of) images for which we already know which number they represent (so called labeled images) and each time tweak the biases and weights a bit so that the next time the network hopefully performs a litlle bit better.
The MNIST dataset is a big list of images (70000) that are already labeled. 

To adjust all the values, we use a technique called Stochastic Gradient Descent (SGD). 
I won't explain the details about SGD here, since that is nicely laid out in Nielsen's book and in the [3blue1brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U) [videos](https://www.youtube.com/watch?v=tIeHLnjs5U8) about neural networks.
The explanation is not always easy to fully understand. It contains a lot of vectors and derivatives, things that are not easy to fully understand for everybody .
I hope my code can help in understanding these details. 
When we feed a training sample to the network we can compare the output of the network to the expected value, and adjust the weights and biases from the output layer backwards through the network.
The comparison of the output values to the expected values is what we call the cost function. 
Here we use the mean quadratic difference between output and expected value. 

```javascript
    // in class NNet
    getCost(){
        return this.output.reduce((prev,n,i)=> 
            (prev + (Math.pow((n.expected - n.value),2  ) / (2*this.output.length))) , 0)
    }
```

Finding the optimal weights and biases of the network is where this (highly dimensional) function reaches a minimum. 
Finding the minimum of a function usually involves derivatives. That's why, the explanation of SGD is also a lot about derivatives.    
The weights and biases are typically not adjusted for every training sample. 
Instead, we accumulate the so-called delta values for the weights and biases for some amount of samples (the mini batch).
When all the samples in the mini batch are evaluated, we adjust the values (learn) by stepping an amount (this.step) in the right direction and go on with the next mini batch.
The code for evaluating and adjusting the values itself is mainly done in the Neuron and Connection objects.
For the NNet, the training routine is 
- feed the training sample to the input layer.
- assign the expected values to the output layer
- calculate the cost
- have the neurons evaluate the difference of expected and actual values
- when the training batch is complete, let the neurons and connections adjust themselves.

For the NNet class we need 2 extra variables the step and the batchSize. The bare train method is only 
```javascript
    this.step = 3
    this.batchSize = 10

    train(item, expected){
        let output = this.check(item)
        // store the expected values in the output neurons
        this.output.forEach((n,i)=> {n.expected = expected[i] })
        let cost = this.getCost()
             
        for(let i=this.neurons.length-1 ;i>=0;i--){
            this.neurons[i].getDelta()
        }

        if (this.trainings % this.batchSize == 0){
            this.neurons.forEach((n,i)=>n.learn(this.step/this.batchSize))
        }
    }
```
In reality, the code is a bit longer, since we want to keep some counters and timings, and store the cost value in a circular array, but this is the main routine.
Again, I use a different approach than Nielsen uses in his code.

Instead of feeding a batch of samples to the training method, I can simply iterate over all values and call train() for every sample. 
The batchSize is a parameter that belongs to the network itself. The network itself keeps track of when a mini batch is complpete.

Now for the neurons, we need a getDelta (accumulate the differences) and a learn method (adjust weights and biases).
```javascript
class Neuron {

    constructor() {
        /*  previously defined variables */
        this.actDeriv = sigmoidDeriv
        this.delta = null
        this.deltaSum = 0.0
    }

    /* previously defined methods*/

    getDelta() {
        if (this.delta === null) {
            if (this.isOutput()) {
                // output neurons  
                this.delta = ((this.value - this.expected) * this.actDeriv(this.z))
            } else {
                // hidden neurons
                this.delta = this.out.reduce((prev, conn) => {
                    return prev + (conn.out.getDelta() * conn.weight)
                }, 0) * this.actDeriv(this.z)
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
            // adjust bias
            this.bias -= batchStep * this.deltaSum
            // adjust input weights
            this.in.forEach((conn, i) => {
                conn.learn(batchStep)
            })
            this.deltaSum = 0
            this.delta = null
        }
    }

    /*
    Extra info methods
     */

    isInput() {
        return this.in.length == 0
    }

    isOutput() {
        return this.out.length == 0
    }
}
```

The Neuron.getDelta() method calculates the delta value for every neuron.
- If the neuron is an output neuron, delta is the difference between the output value and the expected value
- For other neurons (the hidden neurons), delta is the weighted sum of the delta's of the next layer. 
- Then this delta value is sent to all the incoming connections

The learn() method adjusts the neuron's bias and calls the learn method for all incoming connections and resets the accumulated delta values

```javascript
class Connection {
    constructor(input , output){
        /* previously defined variables*/
        this.deltaSum= 0
    }
    setDelta(delta){
        this.deltaSum += delta * this.in.getValue()
    }
    learn(batchStep){
        this.weight -= batchStep * this.deltaSum
        this.deltaSum=0
    }
}
```

That's it. This is all the code needed to train a neural network.
If we want to train and use the network, the code becomes 
```javascript
import {NNet} from "../index.js";
import {getImages} from "./readImages.js";

const net = new NNet(28*28)

function resultArray(label){
    let result = new Array(10).fill(0,0,10)
    result[label] = 1
    return result
}

function run() {
    net.step = 3
    net.batchSize = 10
    net.epochs = 3
    net.translateInput=  (v)=> v.map(item=> item/256 )
    // get index of highest value
    net.translateOutput= (pattern)=> pattern.reduce((prev,val,i,arr)=> val>arr[prev]?i:prev,0)
    net.addLayer(30)
    net.addLayer(10)  // output

    let images = getImages(false,0,50000)
    train(images)
    check(10000,true)
}

function train(images){
    console.time("train")
    let startTime = Date.now()
    for (let epochs=0;epochs<net.epochs ; epochs++) {
        for (let i = 0; i < images.length; i++) {
            net.train(images[i].pixels, resultArray(images[i].label))
            // monitor decreasing cost
            if (net.trainings%5000==0){
                console.log("train "+net.trainings+"  Cost "+net.getAverageCost(10).toFixed(5))
            }
        }
        check(1000,false)
        console.log("epoch "+epochs+"  Cost "+net.getAverageCost().toFixed(5) )
        net.trainTime = Date.now() - startTime
        console.timeLog("train")
    }
}

function check(count,show){
    if (show) console.time("check")
    let startTime = Date.now()
    let start = Math.floor(Math.random()*(10000 - count))
    if (show) console.log("test images "+start+" "+count )
    let testImages = getImages(true,start,start+count)
    let correct = 0
    for (let i=0;i<testImages.length;i++) {
        let result = net.check(testImages[i].pixels)
        if(testImages[i].label == result) correct++
        if (show) console.log(((testImages[i].label == result)?"check ":"MISS ")
            +i+" "+testImages[i].label+"  <=> "+result.label+"  "+result.score.toFixed(4)+"   "+ output.map(r=>r.toFixed(3)))
    }

    if (show) console.log("Network "+net.layers+" layers "+ net.neurons.length+" neurons  ("+net.allNeurons().length+") "+net.neurons.reduce((prev,n)=>(prev+n.in.length),0)+" weights" )
    if (show) console.log("Training iterations "+ net.trainings+"  TrainTime "+net.trainTime+" "+(net.trainings/(net.trainTime/1000)).toFixed(2)+" Trainings/s " + net.step +" step")
    if (show) console.log("Training step:"+  net.step +"   BatchSize: "+net.batchSize)
    if (show) console.log("Check iterations "+ count+" "+(1000*count/(Date.now()-startTime)).toFixed(2)+" Checks/s ")
    console.log("success rate "+ (correct/testImages.length).toFixed(3))
    if (show) console.log("Avg Cost "+ net.getAverageCost().toFixed(5))
}

run()

```

## Additions

- labeling system
- cost function circular array
- cost allows for arbitrary training evaluation
- serializing / deserializing storing models in db

## Performance


## possibilities
- polymorph neurons
- not fully connected layers, convolutional 
- less strick layered approach
- blob of neurons, ff
- loops, feedback
- parallel processing
- 