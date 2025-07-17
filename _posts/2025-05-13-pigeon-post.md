---
title: "A Comprehensive Introduction to PIGEON: Predicting Image Geolocations"
date: 2025-05-13
categories: [blog]
layout: single
classes: wide
---

![Placeholder for featured image](/images/PIGEONCover.jpg)

## Motivation

GeoGuessr, a popular online game with a dedicated community, challenges players to identify locations from images provided by Google Street View, demonstrating the difficulty of accurately guessing image locations. Advances in artificial intelligence (AI), particularly in multimodal and visual AI technologies, have created opportunities to automate and enhance this task significantly. Inspired by this concept, the introduction of the PIGEON model—"Predicting Image Geolocations"—leverages advanced visual embeddings from CLIP-ViT to significantly improve accuracy in automated image geolocation. <br>

This blog post aims to provide a comprehensive overview of the key ideas behind PIGEON, place it within the broader context of multimodal AI research, and explore practical applications and ethical considerations of this technology. As we will later showcase, this approach can even outperform GeoGuessr professionals in pinpointing geolocations on the map based on Street View images.
Join us as we explore the capabilities of PIGEON!


## Table of Contents
1. [Way to Multi-Modal-AI](#way-to-multi-modal-ai)  
   - [LLM](#llm)  
   - [Vision Transformer (ViT)](#vision-transformer-vit)  
   - [CLIP](#clip)  
2. [PIGEON](#pigeon)  
   - [Geocell Division](#geocell-division)  
   - [Synthetic Image Captions](#synthetic-image-captions)    
   - [Distance-Based Label Smoothing](#distance-based-label-smoothing)
   - [Location Cluster Retrieval](#location-cluster-retrieval)  
3. [Experiments](#experiments)  
   - [Experimental Setting](#experimental-setting)  
   - [Evaluation](#evaluation)  
   - [Ablations](#ablations)  
4. [Ethical Considerations](#ethical-considerations)  
5. [Conclusion](#conclusion)


---

## Way to Multi-Modal-AI

### LLM

Large Language Models (LLMs) have found their way into our everyday lives through chatbots, with ChatGPT being one of the most prominent examples. These models are especially powerful in natural language contexts, such as generating text or answering prompts. <br> 

LLMs are based on the Transformer architecture, first proposed by Vaswani et al. in 2017 in the paper “Attention Is All You Need.” This groundbreaking architecture introduced the concept of self-attention, which allows the model to process input more efficiently and makes it highly parallelizable.

Let’s take a closer look at how Transformers work under the hood!

### Transformer

![Vaswani et al](/images/transformerArchitecture.png)

To begin this section, we can roughly break the Transformer architecture into three main components. <br>

First, the encoder, which reads and understands the input sequence (e.g., a sentence in German). Second, the decoder, which generates the output sequence one token at a time. And third, the glue that holds everything together: the multi-head self-attention mechanism, used in both the encoder and decoder. This attention mechanism allows the model to focus on different parts of the sequence at once, taking the relationships between words into account. <br>

Now that we've outlined the three main components of a Transformer, let’s take a closer look at the encoder and decoder and how they interact. The encoder takes the entire input sentence at once and processes it through a stack of identical layers. Each of these layers uses self-attention to let every word in the sentence “see” all the others, helping the model build contextual representations. The output of the encoder is a sequence of vectors that capture both the meaning of each word and how it relates to the rest of the sentence. <br>

The decoder, on the other hand, generates the output sentence step by step. It has a similar layered structure, but with an additional component: it includes an attention mechanism that lets it focus not just on previous words it has generated, but also on the output of the encoder. This way, the decoder can condition its predictions on both the previously generated words and the full input sentence. <br>

Multi-headed self-attention, as the name suggests, is based on the concept of self-attention. Even though this might sound confusing at first, it’s a concept that everyone uses intuitively. Consider the sentence: <br>

<i>“The animal didn't cross the street because it was too tired.”<i> <br>

Answering the question of what the “it” in this sentence refers to is trivial for humans, but for a program, this can be quite challenging. This is where self-attention comes into play: with self-attention, a model can associate “it” with “animal” in this example.

![Example of Attention](/images/attention_example.png)

What now begs the question is how an algorithm can compute self-attention.

To answer this, we first need to calculate three matrices that make it possible: the Query (Q), Key (K), and Value (V) matrices. These are obtained by multiplying the input sequence in matrix form (X) with three pre-trained weight matrices: WQ, WK and WV.


![alt text](/images/query_key_value.png)

These matrices can be understood through an analogy with information retrieval. Let’s take a sentence we want to translate from German to English as an example.

- Q represents what we’re looking for — like a search term. For each word in the sentence, it asks: “Which other words am I interested in?”
- K is like a catalog of representations for all documents — or in our case, all the other tokens in the sentence. It helps determine how relevant each token is to the query.
- V is the actual content we want to retrieve, i.e. the detailed information associated with each token.

With these Matrices we are now able to calculate the so called Scaled Dot-Product Attention:

![alt text](/images/scaled_dot_product.png)

By multiplying Q with Kᵀ, we get a square matrix (number of tokens × number of tokens) whose rows contain raw attention scores with all other tokens in our sentence. Then, the resulting matrix is scaled by  √dk to avoid extreme values when applying the softmax function. After applying the softmax function, the resulting matrix is multiplied with V, finally giving us the attention output.

Now the actual transformers uses multi-headed attention meaning we have N (number of heads) different weight matrices WQ, WK and WV for each head. That means we also get N different attention outputs Z. In order to combine these into one representation, the different outputs are concatenated and multiplied a final time by another weight matrix to get the final output.
### Vision Transformer (ViT)

With the rise and success of Transformer architecture in Deep Learning - more specifically in Natural Language Processing (NLP) Tasks -, there are also attempts to translate this success to the field of Computer Vision. 

- Explain how Vision Transformers (ViT) use the Transformer architecture and use it in Computer Vision
- Explain the architecture
- Explain the advantages of this approach (outperforms convolutional networks, scales better)


### CLIP

- With both previous sections done we can use both to explain what CLIP is that is used in the original PIGEON paper
- Uses both images and text as input
- Explain the how it uses two transformers, one for images, one for text
- Show advantages to previous approaches and limitations

---

## PIGEON

### Geocell Division

- Naive vs. semantic Geocell creation
- OPTICS clustering and Voronoi tessellation to even out number of samples in each cell


### Synthetic Image Captions

- Added to each image, containing information about the location (e.g., climate, region, season, etc.)
- CLIP uses these to better generalize the given images

### Distance-Based Label Smoothing

One of PIGEON’s main contributions is joining the discrete nature of geocell classification with the continuous structure of the Earth’s surface. Traditional approaches treat each geocell as an independent class, penalizing all mistakes equally even when two cells lie side by side and the true location lies in between the two cells. In reality, misclassifying neighboring regions is far less severe than misclassifying two distant continents. To address this problem, PIGEON introduces a novel Haversine-smoothed loss that explicitly models spatial relations between geocells.

At its core, the technique computes the great-circle (Haversine) distance between every geocell’s centroid and the true image location. The great-circle distance between two points on the surface of a sphere is the shortest distance along the surface, accounting for the sphere’s curvature.
Instead of a one-hot target, each training example is assigned a soft label vector $y_{n,i}$ over all cells $i$, where

$$
y_{n,i} = \exp\!\left(-\frac{\mathrm{Hav}(g_i, x_n) - \mathrm{Hav}(g_{\mathrm{true}}, x_n)}{\tau}\right)
$$

with $\mathrm{Hav}(\cdot,\cdot)$ measuring kilometers along Earth’s surface and $\tau$ a temperature hyperparameter controlling smoothing sharpness. Geocells whose centers lie closer to the ground truth receive higher weights, naturally biasing the model toward geographically plausible neighbors.

Training then minimizes a cross-entropy–style loss against these smoothed targets:

$$
L_n = - \sum_i y_{n,i}\,\log p_{n,i},
$$

where $p_{n,i}$ is the model’s predicted probability for cell $i$. This distance-aware objective not only penalizes large mistakes more heavily but also enables PIGEON to learn an implicit multi-scale hierarchy, first pinpointing coarse regions before zooming in on finer distinctions. Picking up the example used at the start of this section, PIGEON now assigns similar target variables to both cells since the distance from the true location to the center of the true geocells is the same as the distance between the true location and the center of the neighboring cell. This ultimately introduces spatial relations between the cells.



### Location Cluster Retrieval

- Inter-Geocell: Predict the top-K most likely geocells
- Intra-Geocell: Cluster the training data within each cluster using OPTICS, then pick location of most similar data point
- Final Prediction: Combine scores from both levels to select the most likely location across all top-K geocells



---

## Experiments

### Experimental Setting

- Models evaluated in five distance distance radii
- median distance error to the correct location as the primary metric
- PIGEON: trained on GeoGuessr data (Google Street View data), tested on 5000 streetview locations, GeoGuessr score as additional metric
- PIGEOTTO: trained on more general images, tested on benchmark datasets


### Evaluation

PIGEON:

- Present the scores for different metrics and categories
- Outperforms top human players in GeoGuessr
- Wins all 6 matches against one of the world’s best GeoGuessr professional

PIGEOTTO:
- Present Benchmark results


### Ablations

- Explain the impact of different ablations (most impactful ones: synthetic Captions, semantic geocells, four-image Panorama)

---

## Ethical Considerations

- Privacy risks
- Military applications


---

## Conclusion

...

---

## References

- to be included
