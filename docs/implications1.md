# Implications of my Neural Network implementation

A few months ago, my dissatisfaction with the code quality of many AI/Neural network articles and tutorials prompted me to develop my own. I wanted an object-oriented, clean, and robust implementation of a neural network. What I came up with can be seen [here](https://github.com/darlingVandamme/OONeuralNetwork).

The [first article](https://medium.com/@geertvandamme/building-an-object-oriented-neural-network-ee3f4af085b6) I published focused on technical aspects of the code itself. Now I’d like to shift gears and explore some of the philosophical implications arising from my experiences and discoveries while working on this project.

## AI Disclaimer

Before I begin, I want to make it clear that what follows are purely my own thoughts and speculations, some of which may be considered controversial. My observations are based on my personal experiences and perceptions about the current state of affairs in AI. I am not closely following all the new trends and directions, so keep in mind that the landscape may change in the (near) future.

This won't be a discussion about ethical topics and moral consequences which are currently a popular subject in AI.

## Freedom of Thinking / Creativity

In the development of AI, are we being driven by technology or limited by it? As of late, the AI boom seems to be largely driven by technological evolution. It's easy to fall into the trap of attributing every recent advancement to technology. I can't help but think that our progress is somewhat stuck in a state of brute-force growth, as, for example, mentioned [here](https://medium.com/predict/ai-is-hitting-a-hard-ceiling-it-cant-pass-851f4667d39b).

The default choice of Python as the go-to language for AI also seems to be more of a limitation than an enabling factor.

## Status of AI and Intelligent Behavior

When looking at a tool like ChatGPT, which is practically "just" a word predictor, we can't help but be impressed. The ability to understand context, learn new concepts, and generate human-like sentences certainly seems intelligent. But can we really call this "smart"?

## What is Intelligence?

Defining intelligence is an undertaking in itself. It is clearly more than just perception and memory—although these are necessary attributes. Intelligence involves handling new situations effectively and adapting to change.

In the context of AI, adaptation might occur through training but there's minimal feedback learning. The gradient descent method used for optimization seems to make the learning process quite slow and inefficient. These things make me believe that training should not be a fundamentally different phase from operation. Implementing learning during the operational phase is technically quite easy.

## Consciousness

It is quite clear that an AI like ChatGPT doesn't possess consciousness. However, the boundary between conscious and non-conscious entities might not be as clear as we think.

## Acting as Part of Thinking

Acting is an integral part of intelligence and consciousness. Learning and training AI also involve some sort of action.

## Intelligent Behavior in Small Networks

Interestingly, I've observed intelligent behavior even in very small neural networks. This makes me rethink the scale that is necessary for entities to exhibit intelligent behavior.

## Issues with Serialization and Cloning

Serialization of classical fully connected layer NNs is straightforward. However, as we move to more complex architectures, we enter an unknown territory, which brings us closer to reflecting the complexity of the human brain. However, these complex architectures present challenges related to GPU compatibility, library support, and serialization.

Cloning, while critical to parallel computing, may conflict with learning. Successful cloning entails creating identical copies, but this might prevent networks from learning and individualizing their responses. Thinking about creating a brain copy here is mind-boggling but also fascinating at the same time!

In conclusion, developing my own object-oriented neural network was an incredible journey that not only helped me understand the technical aspects but also led me to ponder profound philosophical implications related to the nature of intelligence and consciousness. 