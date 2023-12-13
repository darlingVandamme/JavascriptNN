# Neural Net OO Javascript implementation
Although I learned the underlying basics of neural networks
since the early 1990's and already did some experiments with AI and ML. 
With the recent wave of new AI tools appearing everywhere, I wanted to really know and understand the low-level details. 
The basic concept of a neural network is nicely explained and illustrated in the [3blue1brown](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) video series.
The links under the video led me to Michael Nielsen's [online book](http://neuralnetworksanddeeplearning.com).
If you're new to neural networks, I would advise you to first watch the video's or read the first (or first 2) chapter in the book. 
I liked the detailed explanation of how such a network works, but I found the code samples and  
the accompanying python code (of course python, what else... later more on that...) quite confusing and difficult to fully comprehend.
A crucial concept like 'neuron' doesn't even appear anywhere in the code, only in the comments.
The main feed-forward algorithm is coded as

```python
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
```
It works, that's right, but to me, this code doesn't reflect what the feed-forward algorithm really is.

[insert FF schema image ]

A neural net is such a nice example to be modelled as objects in an object-oriented language, but almost all examples use a dense procedural (ugly?) style.

I did a quick google search to see if I could find better, nicer looking implementations, but I was quite disappointed.
Some [articles](https://towardsdatascience.com/understanding-and-implementing-neural-networks-in-java-from-scratch-61421bb6352c)  and [videos](https://www.youtube.com/watch?v=NWQETJz8fII) have promising titles, but when the presented code starts with an array of arrays called 'w'
you know that that is absolutely no clean readable OO implementation. 
This feeling was further 'enhanced' when the article concludes: 
> Being a fellow Python programmer I was not very comfortable with Java

Yeah, I could have guessed.
I didn't look any further, there must be some good implementation examples somewhere,
but instead decided to just do it myself and share it to you all. I hope will clarify what bothers me on the code samples I found everywhere.

## Like we're living in the matrix 
Most implementations of a neural network implement the calculations as matrix and vector operations.
Some people think that's nice and convenient and allows us to use speed-optimized libraries to do the calculations. 
But it doesn't exactly make the code transparent or readable. 
The fact that you can multiply a and b where it doesn't matter if a and b are just numbers or vectors or matrices might 
be convenient for the programmer. It's certainly not easier to read for other people.     
These articles have the intention to teach us something. 
The code should be an illustration or a clarification of the mathematical and computational theory.
But in most cases the code reflects an alternate, matrix oriented, view on the implemented algorithms.

[Code example]

The program doesn't represent a neural network like it is explained in all the fancy pictures.
It models something that can be shown to be a mathematically equivalent model to a neural network that gives a correct output.
But, especially in a educational context like the articles and video's mentioned,
I don't want an illustration of an equivalent alternative. I want to see the real thing. 
The working network! Neurons! Synapses!

## Performance
Of course, I know, I know, performance!!!  
The strongly dominant narrative is that we really need to translate our algorithms to a vector and matrix approach
so that we can use all these fancy GPU-optimized libraries that do the magic much quicker than our lousy slow program. 
As if neural networks need more, obscure, hocus pocus than there already is. 
How and especially why neural networks work is already a mystery, even to most of the people programming and using them. 
We're not going to benefit of an extra layer of black box obscurity.

The code examples in these articles have an educational, illustrative purpose. They shouldn't focus on performance.

Using matrix calculations in introductory articles and tutorials on AI is a perfect example of [premature optimization](https://xkcd.com/1691/).
It's perfectly possible that the matrix approach (and GPU's oh yeah, GPU's an extra obscurity layer, everyone seems to love) 
is necessary to allow the large scale AI tools we see popping up everywhere recently.
But actually, we simply don't know. 
It might well be possible that the matrix idiom leads us to a programming tunnel vision that poses its own limitations and boundaries
that are currently not questioned. 
The matrix calculations, besides giving us a supposed speed boost, limit our creativity and mental freedom.

Let's say eg. that I want to see what happens if I don't want fully connected layers. 
That I want to connect the neurons to the previous layer based on some positional property of the neurons. 
After all, the example that we will use with the images is 2D, but we try to process them suboptimal as one long array. 
That's also the extra information that convolutional networks use. 
Maybe my simplistic idea of non fully connected layers might turn out to be totally useless. 
But in the matrix approach, I lack the possibility to just try it out and experiment with it. 

And I don't like to be mentally limited! 

To use the typical neural lingo, the matrix approach inherently contains the danger of being stuck in a local minimum in our search to solving real world problems.
Anyway, for the rest of the article I'll try to avoid the performance issue and focus on programming elegance and clarity. 
I'll focus on the (not premature) performance tuning in a next article. 
Spoiler: the matrix approach might not be that much faster after all. The issue is at least a bit more complicated.

## The python trap
One more paragraph of critique and negativity, before we turn to the object implementation. 
You can skip this paragraph if you like python ;-)
A lot of the AI/ML applications are implemented in python. (also a lot of data analysis)  
The unfortunate thing is that python actually is very slow, [bad programming language](https://medium.com/nerd-for-tech/python-is-a-bad-programming-language-2ab73b0bda5) and 
bad for the [environment](https://greenlab.di.uminho.pt/wp-content/uploads/2017/10/sleFinal.pdf).
It's a bit strange that python made it as the almost default language for AI and ML. 
Apart from the remarks made in the article, I mainly think that python enables or encourages you to use bad programming habits. 
Partly because python itself is so slow, programmers habitually give up code clarity over presumed speed gains.
If we look at the code examples from neuralnetworksanddeeplearning, apart from some strange structural choices, bad encapsulation and 
supposed but unnecessary speed optimizations, what struck me most is that the basic routine of the network, the feed forward, is implemented twice.
Once as the feedforward routine itself and again inside the backprop routine. 
I guess that's because in the feedForward, the Z values (the weighted sum of incoming neurons, before the activation function is applied) 
are not stored in memory, and in the backpropagation, you need the neurons' Z values anyway. So we end up with one version that stores these values
and another one that calculates the neurons' output in one go.
If you try it out, the presumed optimization by leaving out the Z values is simply non-existent.

This [duplicate code](https://en.wikipedia.org/wiki/Duplicate_code) is a clear violation of the [DRY principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself).
Especially because those few lines are the absolute center of the whole Neural Network. 
The biggest problem with this duplicated code is that a change in one place is not always replicated in the other location and could lead to bugs that are difficult to find.

I also found it a bit strange that the Cost function itself, while being such a crucial part of the training process, isn't even implemented at all. Not even as an illustration.

## OO Javascript implementation
So, how did I implement this Neural Network? 

I won't explain the details of the network or the meaning of the different concepts and terminology. 
As said before, read [the book](http://neuralnetworksanddeeplearning.com) and watch the videos. To keep things comparable, I will be focussing on the exact same example network as in the first 2 chapters of the book and the videos.
A completely connected network with 1 hidden layer (30 neurons) and 10 output neurons to recognize written numbers, trained on the [MNIST dataset](https://yann.lecun.com/exdb/mnist/).
(There seems to be something wrong with the original website, but you can download the dataset at several other locations)

At first, I'll implement this network in javascript without any other dependencies, with a focus on clean, readable and nicely encapsulated OO code. 
As already mentioned, we won't be focussing on performance, but it might guide us in certain implementation decisions, as long as it doesn't interfere with the readability.
In the various loops and propagations, we will be mixing a recursive and an iterative approach, depending on what's most obvious or clean.
Later on, we might sometimes change this approach if it seems to be to slow or using too many useless function calls. 
Also, we prefer to use the dedicated array looping functions like forEach, map and reduce over old-fashioned for() iteration.   

Let's first focus on the main structure and the feed forward algorithm.
Anyway, we'll use a bottom up approach and start with the Neuron itself. 
After all, just like in the brain analogy, the network is just an emergent concept that holds together the neurons. 
Obviously, at the basis of the code is the Neuron class, that's doing most of the work. 
Well, obviously? Then why do most sample implementations don't even have a Neuron class???
The Neuron class keeps the values (value and z), the neurons bias and a list of connections to the previous and next layer.

[illustration]


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

    connectLayer(layer) {
        this.in = layer.map(other => {
            return new Connection(other, this)
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

The connectLayer function creates the (fully connected) network structure by connecting the Neuron to the previous layer by using a Connection object.  
You could code this with an implicit Connection (or should we call it Synapse), but, especially when we add the training code, it's cleaner if we also code the Connection class separately.

```javascript
class Connection {
    constructor(input, output) {
        this.in = input
        this.out = output
        this.weight = getRandom()
        input.out.push(this)
        output.layer = input.layer+1 
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

The last class we need is the Network itself to keep the whole structure together and provide the input and output functions. 
I already mentioned that most implementations put everything in the Network class and don't even implement a Neuron (and certainly no Connection).
This is exactly what I mean with 'clean encapsulation'. Here we put the variables and methods in the class where they belong.
Clean code and proper encapsulation leads to better testability and more maintainable code. (move?)

```javascript
import Neuron from "./neuron.js";

class NNet {
    constructor(inputSize) {
        // keep a list of neurons
        // makes it easier to clone and serialize
        this.allNeurons = []
        this.input = Array.from({length: inputSize}, (n1, i) => {
            return new Neuron()
        })
        this.output = this.input
        this.layers = 1
        this.count = 0
        // keep a list of hidden and output neurons
        // makes it easier to reset and serialize
        this.neurons = []  
    }

    addLayer(size) {
        let layer = Array.from({length: size}, (n1, i) => {
            let n = new Neuron()
            this.allNeurons.push(n)
            n.connectLayer(this.output)
            return n
        })
        this.output = layer
        this.layers++
    }

    reset() {
        this.allNeurons.forEach((n, i) => {
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
- TODO I changed this in later implementations. Maybe implement the changes earlier

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

We can train our network by feeding it (lots of) images for which we already know which number they represent (so called labeled images) and each time tweak the biases and weights a bit so that the next time the network hopefully performs a little bit better.
The MNIST dataset is a big list of images (70000) that are already labeled. 

To adjust all the values, we use a technique called Stochastic Gradient Descent (SGD). 
I won't explain the details about SGD here. Once more... watch the [videos](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi).
Although the overall principle is clearly laid out, the fine details are sometimes a bit harder to fully understand (What is a mini-batch anyhow?). 
It contains a lot of vectors and derivatives, things that are not always obvious to everybody .
I hope my code can help in understanding these details. 
When we feed a training sample to the network we compare the output of the network to the expected value,
and adjust the weights and biases from the output layer backwards through the network.
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
        // or less 'backwards' this.neurons.forEach((n,i)=>n.getDelta())

        if (this.trainings % this.batchSize == 0){
            this.neurons.forEach((n,i)=>n.learn(this.step/this.batchSize))
        }
    }
```
In reality, the code is a bit longer, since we want to keep some counters and timings, and store the cost value in a circular array, but this is the main routine.
Again, I use a different approach than Nielsen uses in his code.

Instead of feeding a batch of samples to the training method, I can simply iterate over all values and call train() for every sample. 
The batchSize is a parameter that belongs to the network itself, not to the calling object. 
The network itself can keep track of when a mini batch is complpete and perform the adjustments accordingly.

Now for the neurons, we need the getDelta (accumulate the differences) and learn methods (adjust weights and biases) where the real work is done.

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
- For other neurons (the hidden layers), delta is the weighted sum of the delta's of the next layer. 
- Then this delta value is backpropagated to all the incoming connections where each connection also keeps an accumulating delta value for the connection's weight 

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

That's it. This code can now be seen as our simple Neural Network library that can train a network and use it. 
If we want to create the network to recognise the handwritten MNIST numbers, the code becomes something like

```javascript
import {NNet} from "../index.js";
import {getImages} from "./readImages.js";

const net = new NNet(28*28)

function run() {
    net.step = 3
    net.batchSize = 10
    net.epochs = 3
    net.translateInput=  (v)=> v.map(item=> item/256 ) // depends on thhe version of the reader code
    // get index of highest value
    net.translateOutput= (pattern)=> pattern.reduce((prev,val,i,arr)=> val>arr[prev]?i:prev,0)
    net.addLayer(30)   // hidden
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

// transform a number into the expected output array
function resultArray(label){
    let result = new Array(10).fill(0,0,10)
    result[label] = 1
    return result
}

run()

```

- We create a neural network with 28*28 input neurons
- add a fully connected layer of 30 neurons
- add a final layer (output) of 10 neurons
- read the labeled image data  (I actually don't like this. It looks too pythonesque, but I wanted to follow the same approach) 
- send this image data several times (epochs) to the network to train it.
- after every 5000 images, we monitor the current average cost function result
- after each epoch, we check 1000 test images to see what the current success rate is
- after x training epochs, the 10000 test images are tested and each result is shown in the console.

[screenshots]


## Additions

While I was writing this program, I added quite some features that are not present in the Nielsen's python code, but I think are quite useful.
These are not relevant for the basic working of the code, but might be interesting to ecplore.

[code repository]

###  Forward feedforward
I mentioned before that the main feedforward algorithm might seem a bit backwards / recursive.
Although I still think this approach looks nice and might have some interesting advantages in the future, if we just add 1 line of code like: 
```javascript
    // in NNet
    feed(input){
        this.count++
        this.reset()
        this.input.forEach((n,i)=> {n.value = input[i]})
    // add this:
        this.neurons.forEach((n,i)=>{n.ff()})     
    }
```
It actually speeds things up as well quite a bit, since there's less unneeded recursive method calls.

### cost funtion circular array
After each training input, the cost function is calculated and stored in a circular array.
I don't want to keep an array of millions of training samples, but simply keeping the most recent 500 cost results in an array is very informative
At first, I added a binary calculateCost variable, to specify if I should calculate the cost of every training iteration.
It turned out to be a premature optimization as well. Calculating the cost of every training iteration adds some load on the system,
but it's hardly measurable. The overall timings are in the same range whether I calculate the cost or not.
Calculating the cost function allows us to decide when extra training samples don't offer extra advantage, and so we can stop the training session.
```javascript
class NNet{
    constructor(inputSize){
        // previously defined variables
        this.costs = []
        this.costIndex=0
        this.costsSize = 500
    }

    train(item, expected, label){
        /* ..... */
        this.output.forEach((n,i)=> {n.expected = expected[i] })
        this.costIndex = this.trainings % this.costsSize
        this.costs[this.costIndex] = this.getCost()
        /* ....*/ 
    }

    getAverageCost(){
        return this.costs.reduce((prev,cost)=>{return prev+cost},0)/this.costs.length
    }


```

### labeling system
Since NNet knows how a label can be translated into an array of input values (translateInput),
we can add a label to the input values of the train routine and create an automatic selection mechanism for the translateOutput method.
By keeping a list of all trained labels with their corresponding expected output vector, we can compare a new output to all known trained outputs and select which one fits the most.
In the MNIST example, the translateOutput simply selects the index of the output value with the highest score, but it doesn't give you an estimate of how clear the results are.
The label inventory might give more subtle results. I won't show the code here but you'll find it in the repository. TODO

```javascript
    result = { 
        label:5,
        score:0.998,
        pattern:[],
        trained:5425
    }
```

### serializing / deserializing
Of course, it's a waste of time and energy if we always have to retrain our networks when we need them. 
We should have some mechanism of storing the trained network in a file or database, and use the pretrained network when we need it.
Serializing / deserializing or cloning is also important if we want to parallelize the training or checking process.
As much as I like the clean OO network approach, I have to admit that it's much simpler to serialize/save/clone a network when it's simply an array of arrays.
When the network is a real network of neurons, massively linked to each other, serialization is not that easy.


storing models in db

## possibilities
- polymorph neurons
- not fully connected layers, convolutional 
- less strick layered approach
- blob of neurons, ff
- loops, feedback
- parallel processing
- 