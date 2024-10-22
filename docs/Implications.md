# Implications of my Neural Network implementation

A few months ago my disappointment in the code quality of many AI / Neural network articles and tutorials,
was the motivating factor for developing my own, clean, object-oriented [implementation](https://github.com/darlingVandamme/OONeuralNetwork).
The [first article](https://medium.com/@geertvandamme/building-an-object-oriented-neural-network-ee3f4af085b6) focuses on the code itself and is therefore very technical.
Now, I want to explore a bit more the philosophical implications of what I found out doing these experiments.

## AI Disclaimer.
I'm not an AI expert who follows all the latest trends and technologies. 
I just wanted to experiment a bit with AI at the lowest level (ie. programming it myself) and I learned a lot from it.
My background in psychology, cognitive science, philosophy and 
software development in general give these experiments a specific personal perspective. 
So the issues I'm talking about come from my own experience and thoughts. 
They may sometimes be highly speculative and probably controversial. ;-)

* Ethical issues and consequences are popular. Not focusing on those here.

## Driven vs. limited by technology.
The recent AI boom is driven by a technological revolution that makes it possible to develop and train the systems we're seeing now.
Cheap, fast computing power allows...

However, I have the impression that most of the current systems and companies are stuck in a brute force growth.
Systems become better, and mostly bigger, by adding extra computing power, memory and GPU's.
And consuming huge amounts of energy.

Next to this brute force growth, we will also need to invest in radically different and experimental projects.
A lot of articles, videos and tutorials about AI operate within a strict straightjacket about how things should be implemented.
Things that feel too strict to me:...
  #### Python as the default language. 
In se, there should be nothing wrong in using python, but currently there's an aura around python that it's the only viable option in AI.
Most python AI sample code we see is badly written, not up to programming quality standards. 
Python is big in academics and amongst data scientists but the way it is mostly used looks quick 'n dirty.
Python is a rather slow programming language, so we rely on external libraries (numpy, Keras, tensorflow... ) to perform the real calculations.
But also in this case, these technologies can have a limiting effect on the flexibility.
  #### Matrix algorithms
In most cases, the calculation of the neural network algorithm is implemented as a matrix algorithm.
Although I find the network / graph oriented representation clearer and ..., the equivalent matrix algorithm can be done faster by using external libraries. 
These matrix operations can be calculated very fast by using a GPU.  
But again, this comes at a cost. By focussing on these matrices, we lose the flexibility of experimenting with implementations or algorithms that don't translate well to matrix calculations. 

  #### Fully connected layers
I guess, the fact that we typically implement a neural network as fully connected layers of neurons is a consequence of the matrix approach.


  #### Loops


Bio neurons https://finalspark.com/live/

* Status of AI
* Freedom of thinking / creativity
* fashion / typical ways of working / taboo 
* I haven't started experimenting with these issues, maybe it'll turn out useless, but at least, I want to be able to do it.
* Test your AI knowledge...


## Intelligent behavior
When I was implementing my OO network, I created a small dummy network, just as a quick test of my code.
The network has 5 input neurons, 10 neurons in a hidden layer and 2 outputs. 
You can think of it as a system that can check an input pattern of 5 (eg [1,0,1,0,1] of [1,2,3,4,5]) and answer through the 2 output neurons with a 'yes' ([1,0]) or a 'no' ([0,1]).
Actually, it can also respond with a 'maybe' (like [0.78,0.2]) or 'I don't know' ([0.3,0.25]).
Even this small network, only 15 neurons or so, showed certain features of what we could call intelligent behavior.
When the network is trained on only 2 specific patterns, it can correctly identify patterns that look like the ones it was trained on, although it will indicate to be less sure about its answer.

Although we can hardly call this simple recognition task 'intelligence', it does show that it can handle new, previously unseen, situations.

 
* Comparison with our thinking.
* Limitations of brain analogy
* Intelligent behavior even in very small networks
    * "ChatGPT is only a word predictor" Well, it's quite an impressive one. 
    * ChatGPT can surely be seen as intelligent. Can we call it smart?
* What is intelligence
  * Clearly more than (sensory) perception and memory. But these are necessary.   
  * handling new situations. Good 
  * Adapting to change. ok, but not impressive. Train, but no feedback learning.
  * 
* Consciousness 
  * Quite clear that chatGPT is not conscious 
  * However, I think that there's no hard clear distinction between conscious and not conscious.
* Acting as part of thinking / intelligence / consciousness

*    
* Learning / training who is the actor? feedback
* Training looks very slow. Gradient descent doesn't seem the best approach
  * training as a fundamentally different phase
  * Learning during operational phase is technically quite easy.
  * training during operation (learning) conflicts with cloning
  * 
* Serialization issue
  * classical NN (fully connected layers) are very easy to serialize
  * https://huggingface.co/docs/hub/gguf
  * https://www.ibm.com/think/topics/gguf-versus-ggml
  * When we allow more complex architectures
    * unknown territory
    * brain similarity
    * GPU and matrix library issues
    * serialization
  * cloning conflicts with learning / individualization of parallel clones
  * cloning - highly parallel computing
  * Brain copy



Tegmark:
https://www.youtube.com/watch?v=_-Xdkzi8H_o
