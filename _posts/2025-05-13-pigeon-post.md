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
2. [PIGEON/PIGEOTTO](#pigeon)
   - [Related Work](#related-work)
   - [Geocell Division](#geocell-division)  
   - [Synthetic Image Captions](#synthetic-image-captions)    
   - [Distance-Based Label Smoothing](#distance-based-label-smoothing)
   - [Hierarchical Retrieval-Based Refinement](#hierarchical-retrieval-based-refinement)  
3. [Experiments](#experiments)  
   - [Experimental Setting](#experimental-setting)  
   - [Qualitative Analysis](#qualitative-analysis)
   - [Quantative Analysis](#quantitative-analysis)  
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

## PIGEON/PIGEOTTO

### Related Work

### Geocell Division

- Naive vs. semantic Geocell creation
- OPTICS clustering and Voronoi tessellation to even out number of samples in each cell


### Synthetic Image Captions

- Added to each image, containing information about the location (e.g., climate, region, season, etc.)
- CLIP uses these to better generalize the given images

### Distance-Based Label Smoothing

One of PIGEON’s main contributions is joining the discrete nature of geocell classification with the continuous structure of the Earth’s surface. Traditional approaches treat each geocell as an independent class, penalizing all mistakes equally even when two cells lie side by side and the true location lies in between the two cells. In reality, misclassifying neighboring regions is far less severe than misclassifying two distant continents. To address this problem, PIGEON introduces a novel Haversine-smoothed loss that explicitly models spatial relations between geocells.

At its core, the technique computes the great-circle (Haversine) distance between every geocell’s centroid and the true image location. The great-circle distance between two points on the surface of a sphere is the shortest distance along the surface, accounting for the sphere’s curvature.
Instead of a one-hot target, each training example is assigned a soft label vector $$y_{n,i}$$ over all cells $$i$$, where

$$
y_{n,i} = \exp\!\left(-\frac{\mathrm{Hav}(g_i, x_n) - \mathrm{Hav}(g_{\mathrm{true}}, x_n)}{\tau}\right)
$$

with $$ \mathrm{Hav}(\cdot,\cdot) $$ measuring kilometers along Earth’s surface and $$\tau$$ a temperature hyperparameter controlling smoothing sharpness. Geocells whose centers lie closer to the ground truth receive higher weights, naturally biasing the model toward geographically plausible neighbors.

Training then minimizes a cross-entropy–style loss against these smoothed targets:

$$
L_n = - \sum_i y_{n,i}\,\log p_{n,i},
$$

where $$p_{n,i}$$ is the model’s predicted probability for cell $i$. This distance-aware objective not only penalizes large mistakes more heavily but also enables PIGEON to learn an implicit multi-scale hierarchy, first pinpointing coarse regions before zooming in on finer distinctions. Picking up the example used at the start of this section, PIGEON now assigns similar target variables to both cells since the distance from the true location to the center of the true geocells is the same as the distance between the true location and the center of the neighboring cell. This ultimately introduces spatial relations between the cells.

| ![Haversine smoothing curves](/images/haversineLoss.png) |
|:---:|           
|Training without and with haversine smoothing. |



### Hierarchical Retrieval-Based Refinement

Building on its powerful geocell classification, PIGEON further refines its location estimates through a three-level, coarse-to-fine retrieval mechanism.  

**Top layer:** PIGEON first selects its top K = 5 candidate geocells based on the pretrained CLIP-ViT probabilities.  

**Middle layer:** Within each predicted cell, training points are clustered using the OPTICS density-based algorithm, and each cluster is represented by the centroid of its CLIP image embeddings. During inference, the query embedding is assigned to the nearest cluster in Euclidean space.  

**Bottom layer:** Finally, PIGEON selects the single closest location within that cluster—adding two extra levels of granularity beyond the initial cell prediction.

![Three-level hierarchical retrieval diagram](/images/threeLayers.png)

To integrate these multi-scale cues, PIGEON multiplies each geocell’s original probability by a refinement score derived from a temperature-scaled softmax over the cluster distances. This joint scoring across all top K cells yields a final geolocation estimate that balances global confidence with local embedding proximity.




---

## Experiments

### Experimental Setting


Now that we know how the methods behind the model, we still need to explain what data the models were trained on since here lies the main difference between PIGEON and PIGEOTTO besides the setting of some hyperparameters. Then for each model the authors use a evaluation method specifically tailored for the use case of the model using both synthetic setups and real-world conditions. 

**For PIGEON**, which is optimized for Street View images like in GeoGuessr, the authors collected a dataset of 100,000 locations from the game. At each location, they captured a 360-degree view using four images spaced evenly around the compass. This panorama-style input helps the model detect geographical cues like vegetation. In total, the training data amounted to 400,000 images.

To evaluate PIGEON’s accuracy, the researchers used a separate holdout set of 5,000 unseen GeoGuessr locations. Importantly, they didn’t just rely on offline metrics. They also deployed PIGEON live into GeoGuessr using a custom Chrome extension bot. This allowed for direct comparisons against players across all skill levels including one of the world’s top-ranked professionals. 

| ![Geoguessr Panorama](/images/panoramaGeoguessr.png) |
|:---:|           
|Four images comprising a 360-degree panorama from a location in Pegswood, England|

**For PIGEOTTO**, the broader model designed to handle general-purpose geolocation from a single image, the researchers gathered over 4.5 million images from Flickr and Wikipedia (including landmarks from the Google Landmarks v2 dataset and the Flickr images from the MediaEval 2016 dataset). Instead of Street View, these images came from diverse, user-generated content around the world.


PIGEOTTO was tested on several standard geolocation benchmarks used in the literature including IM2GPS, IM2GPS3k, YFCC4k,  YFCC26k and GWS15k.
These benchmarks evaluate how many guesses fall within different distance thresholds (like 25 km or 200 km from the correct location) and measure median error in kilometers.

| ![PIGEOTTO images](/images/pigeottoRandom.png) |
|:---:|           
|Four samples from the MediaEval 2016 dataset|

To summarize, while PIGEON was built to master the game of GeoGuessr using panoramic Street View inputs, PIGEOTTO was trained for general image geolocation on a planetary scale. The careful separation of data sources and testing conditions ensured that each model was evaluated fairly in its respective domain.
 



### Qualitative Analysis

Now that we've covered the the methods and the training data the models were trained on, we want to showcased the capabilities of these models by showing some qualitative results. The qualitative results not only illustrate the models’ reasoning but also highlight the kinds of cues they’ve learned to pick up, often mirroring expert GeoGuessr strategies.

**Interpreting Model Attention for PIGEON**
The authors generated attention-attribution maps over the CLIP-based backbone to see which parts of a Street View image drive PIGEON’s geolocation predictions.

| ![Attention Attribution Map](/images/attentionAttributionMap.png) |
|:---:|           
|Attention attribution map for an image in New Zealand|

 For example, when given a view of a New Zealand countryside, PIGEON’s attention is mainly focused on the following cues:
- Road markings and signs that are specific to certain coutries like center lines and stop signs.
- Utility poles that also vary a lot even inside a country giving a big hint on where the image was taken.

This kind of behaviour is very interesting since it mimics that of a Geoguessr professionals who for example also tend to look for utilit poles and street signs since these are things that typically are a really strong indicator on where you are on the globe. And the model does so without explicitly being told to do so underlining the generaliziability and the performance of the model.

**Uncertainty Across Scenes for PIGEON**

Next we want to go about some examples where PIGEON is most uncertain about which geocell to guess.

| ![Dark Panorama ](/images/darkPanorama.png) |
|:---:|           
|Example of an image for which PIGEON was most uncertain about.|

In the above image it is to be expected that PIGEON cannot make a reasonable guess since nothing can be really recognized on these images. High Uncertainties in these kind of scenarios with out of distribution sample are bound to happen and acceptable. The next image though is more interesting.

| ![Uncertain Forest](/images/uncertainForestPanorama.png) |
|:---:|           
|Example of an image for which PIGEON was most uncertain about.|

Here we can see a road going through a forest with a lot of vegetation. But there aren't any road markings or street signs to be seen. Still there is a lot a vegetation to be seen which can give siginificant cues in which country or region the image was taken. So interestingly enough PIGEON has a hard time guessing the correct geocell in this instance underlining that the model still can improve in these kind of scenarioes. 

PIGEON:

- Present the scores for different metrics and categories
- Outperforms top human players in GeoGuessr
- Wins all 6 matches against one of the world’s best GeoGuessr professional

PIGEOTTO:
- Present Benchmark results


### Quantitative Analysis
- safsaf

[▶️ Watch PIGEON beat a GeoGuessr professional on YouTube](https://www.youtube.com/watch?v=ts5lPDV--cU)

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
