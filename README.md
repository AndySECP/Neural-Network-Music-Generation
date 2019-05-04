# Neural-Network-Music-Generation
Leverage state of the art AI models to generate human sounding musics

This project is part of the class CS282-Designing and Visualizing Neural Networks, with professor John Canny at UC Berkeley.

## Objectives

We investigate the following research problem: “How can we architect a model to generate sequences of discrete tokens that not only mimics short term patterns, but also model longer term patterns.” We will be conducting this research in conjunction with music generation - rhythm and melody being the short term patterns and form and structure being the longer term patterns we will attempt to model.


## Background

There are already extensive researches on the challenge of artificial music generation. Recent approaches include LSTM and bi-STM architectures[1], SeqGAN architecture that trains generative adversarial nets for structured sequences generation via policy gradient[2] or GAN[3]. Yet, the most recent advancements come from OpenAI and leverage the recent breakthrough in NLP Deep Learning models. They are using a improved version of the transformer, called Musical Transformer. The model uses attention: every output element is connected to every input element, and the weightings between them are calculated dynamically. It isn’t explicitly programmed with an understanding of music, but it can discover patterns of harmony, rhythm, and style by learning to predict tokens — notes encoded in a way that combines the pitch, volume, and instrument information — in hundreds of thousands of MIDI files. See the OpenAI blog post for more information: https://openai.com/blog/musenet/



## Approach

1. As a first baseline, we are buidling a **bi-LSTM model** 
2. We are improving our model building up on the last OpenAI publication and implementing a **GPT network**
3. The model are currently evaluating using a **customized BLEU score**

## Evaluation

To have a **quantitative metric** to evaluate our generated sample, we are using blue score. Several approaches are tried. 
- Looking at different **cumulated ngrams**
- Creating a **customized BLEU score** that reflect what we want our generated samples to demonstrate

Customized BLEU socre:
To have a better metric, we can **weight differently the n_grams**. We expect 1-gram BLEU score to be equal to 1 as by using a sufficiently large reference corpus all the notes available are going to be used. Therefore, we can input a smaller weight to small grams, then a higher weights for higher grams and at end diminishing the weight again for very large grams (8-9) for the sake of diversity.

Weigths proposal:
- 1-gram: 0
- 2-gram: 1/12
- 3-gram: 1/12
- 4-gram: 1/6
- 5-gram: 1/6
- 6-gram: 1/6
- 7-gram: 1/6
- 8-gram: 1/12
- 9-gram: 1/12

For the LSTM generated samples, the results are pretty with the BLEU evaluation, and are close to real compositions for both cumulated ngrams as well as our customized metric function. For the GPT generated samples, the current samples performances are not as good at the light of the BLEU score evaluation:

![bleu_gpt50](https://user-images.githubusercontent.com/38164557/57184874-24dae680-6e77-11e9-8a59-b77a73771964.png)


## Papers

[1] DeepJ: Style Specific Music Generation - 2018
[2] Yu, L., Zhang W., Wang J., and Yu, Y. Seqgan: Sequence generative adversarial nets with policy gradient. arXiv preprint arXiv:1609.05473, 2016
[3] Midinet: A Convolutional Generative Adversarial Network for Symbolic Domain Music Generation - 2017
