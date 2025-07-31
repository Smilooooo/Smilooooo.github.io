---
title: "A Comprehensive Introduction to PIGEON: Predicting Image Geolocations"
date: 2025-05-13
categories: [blog]
layout: single
classes: wide
---

![Placeholder for featured image](/images/PIGEONCover.jpg)


## Motivation

GeoGuessr, a popular online game with a dedicated community, challenges players to identify locations from images provided by Google Street View, demonstrating the difficulty of accurately guessing image locations. Advances in artificial intelligence (AI), particularly in multimodal and visual AI technologies, have created opportunities to automate and enhance this task significantly. Inspired by this concept, the introduction of the PIGEON model "Predicting Image Geolocations" leverages advanced visual embeddings from CLIP-ViT to significantly improve accuracy in automated image geolocation. <br>

This blog post aims to provide a comprehensive overview of the key ideas behind PIGEON, place it within the broader context of multimodal AI research, and explore practical applications and ethical considerations of this technology. As we will later showcase, this approach can even outperform GeoGuessr professionals in pinpointing geolocations on the map based on Street View images.
Join us as we explore the capabilities of PIGEON!


|![](/images/geoguessr-clip.gif)|
|:---:|           
|One round of the online game GeoGuessr in singleplayer-mode. |

## Table of Contents  
1. [PIGEON/PIGEOTTO](#pigeon)
   - [Related Work](#related-work)
   - [Geocell Division](#geocell-division)  
   - [Synthetic Image Captions](#synthetic-image-captions)    
   - [Distance-Based Label Smoothing](#distance-based-label-smoothing)
   - [Hierarchical Retrieval-Based Refinement](#hierarchical-retrieval-based-refinement)  
2. [Experiments](#experiments)  
   - [Experimental Setting](#experimental-setting)  
   - [Qualitative Analysis](#qualitative-analysis)
   - [Quantative Analysis](#quantitative-analysis)  
   - [Ablations](#ablations)  
3. [Ethical Considerations](#ethical-considerations)  
4. [Conclusion](#conclusion)


---

## PIGEON/PIGEOTTO

### Related Work 

The problem of geolocalization—that is, mapping an image to coordinates identifying where it was taken—has long been a challenging area in computer vision. Several factors contribute to this complexity, including variations in daytime, weather conditions, viewing angles, illumination, and more.

An early modern approach was IM2GPS , which employed a retrieval-based method using handcrafted visual features. However, this technique required an extensive database of reference images, making it impractical for planet-scale geolocalization tasks. As a result, later researchers narrowed their geographic scope, focusing on specific cities such as Orlando and Pittsburgh, entire countries like the U.S. , and even specific geographical features like mountain ranges, deserts, and beaches.

The rise of deep learning significantly shifted image geolocalization methods from handcrafted features to end-to-end learning. Google’s 2016 released "Planet" paper marked the first attempt to apply convolutional neural networks (CNNs) to geolocalization, framing it as a classification problem across geocells. Subsequently, researchers leveraged deep learning improvements to train CNNs on large datasets of mobile images , even deploying these models in competitive settings against human players in the game GeoGuessr.

Recently, transformer architectures—originally successful in natural language processing—have found their place in computer vision. Pretrained vision transformers (ViT) and multimodal models like OpenAI’s CLIP and GPT-4V have also been effectively applied to the task of image geolocalization.

When treating geolocation as a classification problem, a crucial question arises: How should the world be partitioned into geographical classes? Previous approaches have used simple rectangular geocells, rectangular cells adjusted for Earth's curvature and balanced size, or arbitrarily shaped geocells resulting from combinatorial partitioning. However, a significant disadvantage of these methods is their failure to capture meaningful geographic characteristics due to arbitrary boundaries. The PIGEON model addresses this limitation by integrating geographic features directly into the construction of geocells, as discussed in the following section.

### Geocell Division

- Naive vs. semantic Geocell creation
- OPTICS clustering and Voronoi tessellation to even out number of samples in each cell

To overcome the limitations of the naive geocell approach, PIGEON introduces Semantic Geocells. Instead of relying on arbitrary rectangular subdivisions, semantic geocells leverage existing administrative and political boundaries. These boundaries naturally align better with geographic divisions that people intuitively recognize.

Creating semantic geocells involves merging neighboring areas within these administrative boundaries until each geocell contains roughly the same number of images. This balancing ensures effective model training. Crucially, semantic geocells always respect country borders, ensuring that predictions remain geographically coherent.

This approach offers significant advantages. Semantic geocells can capture distinctive region-specific details—like unique street signs, road markings, or architectural styles—which are key visual clues for accurate geolocation. They also naturally incorporate meaningful geographic boundaries such as rivers or mountain ranges, creating intuitive geographic classes.

By embedding richer geographic context into the data, semantic geocells ultimately enable PIGEON to make predictions that are both more accurate and realistic.

Despite the advantages provided by semantic geocells, a fundamental issue remains: some regions naturally attract more images, particularly popular landmarks or tourist hotspots. This uneven distribution can create class imbalance, posing challenges for effective model training.

To address this imbalance, the authors propose a targeted clustering strategy using the OPTICS algorithm. OPTICS identifies densely photographed regions, grouping images into meaningful clusters. Each image is then assigned to a specific cluster, which helps balance the number of images per geocell.

Finally, a Voronoi tessellation ensures these adjusted geocells remain spatially continuous and coherent. By creating contiguous geographic areas around each cluster, this method enables PIGEON to better manage densely photographed locations, ultimately improving both the accuracy and reliability of its predictions.



### Synthetic Image Captions

- Added to each image, containing information about the location (e.g., climate, region, season, etc.)
- CLIP uses these to better generalize the given images

In PIGEON, synthetic image captions play a central role in the multi‑task contrastive pretraining of its vision-language model (CLIP). These captions aren’t human-written but are automatically generated using auxiliary geographic metadata—such as Köppen–Geiger climate zone, elevation, season, population density, and even the predominant driving side in a country. These labels are sourced from open geospatial datasets to describe each training image’s location context.

Each synthetic caption typically includes information about:
- Climate zone (e.g. “temperate maritime,” “tropical monsoon”)
- Region type (e.g. urban, mountainous, coastal)
- Season, when available
- Population density or elevation cues, and even roadside driving orientation
These enriched captions are used during CLIP-based pretraining in a multi-task contrastive framework: CLIP learns to align images not just with geocell labels, but also with these auxiliary textual descriptions.

This means PIGEON’s CLIP model develops a more robust multimodal embedding space, where images are tightly associated with not only visual features but also their geographic context. The result: stronger generalization to unseen regions, especially when facing environments with limited training data.

In short, synthetic captions serve two essential purposes in PIGEON:

They enrich training data with structured geographic context, beyond purely visual patterns.
They enable CLIP to generalize better across diverse environments, improving prediction accuracy even in sparsely represented locations.


### Distance-Based Label Smoothing

One of PIGEON’s main contributions is joining the discrete nature of geocell classification with the continuous structure of the Earth’s surface. Traditional approaches treat each geocell as an independent class, penalizing all mistakes equally even when two cells lie side by side and the true location lies in between the two cells. In reality, misclassifying neighboring regions is far less severe than misclassifying two distant continents. To address this problem, PIGEON introduces a novel Haversine-smoothed loss that explicitly models spatial relations between geocells.

At its core, the technique computes the great-circle (Haversine) distance between every geocell’s centroid and the true image location. The great-circle distance between two points on the surface of a sphere is the shortest distance along the surface, accounting for the sphere’s curvature.
Instead of a one-hot target, each training example is assigned a soft label vector $$y_{n,i}$$ over all cells $$i$$, where

$$
y_{n,i} = \exp\!\left(-\frac{\mathrm{Hav}(g_i, x_n) - \mathrm{Hav}(g_{\mathrm{true}}, x_n)}{\tau}\right)
$$

with $$ \mathrm{Hav}(\cdot,\cdot) $$ measuring kilometers along Earth’s surface and $$\tau$$ a temperature hyperparameter controlling smoothing sharpness. Geocells whose centers lie closer to the ground truth receive higher weights, naturally biasing the model toward geographically plausible neighbors.

Training then minimizes a cross-entropy-style loss against these smoothed targets:

$$
L_n = - \sum_i y_{n,i}\,\log p_{n,i},
$$

where $$p_{n,i}$$ is the model’s predicted probability for cell i. This distance-aware objective not only penalizes large mistakes more heavily but also enables PIGEON to learn an implicit multi-scale hierarchy, first pinpointing coarse regions before zooming in on finer distinctions. Picking up the example used at the start of this section, PIGEON now assigns similar target variables to both cells since the distance from the true location to the center of the true geocells is the same as the distance between the true location and the center of the neighboring cell. This ultimately introduces spatial relations between the cells.

| ![Haversine smoothing curves](/images/haversineLoss.png) |
|:---:|           
|Figure 1: Training without and with haversine smoothing. |



### Hierarchical Retrieval-Based Refinement

Building on its powerful geocell classification, PIGEON further refines its location estimates through a three-level, coarse-to-fine retrieval mechanism.  

**Top layer:** PIGEON first selects its top K = 5 candidate geocells based on the pretrained CLIP-ViT probabilities.  

**Middle layer:** Within each predicted cell, training points are clustered using the OPTICS density-based algorithm, and each cluster is represented by the centroid of its CLIP image embeddings. During inference, the query embedding is assigned to the nearest cluster in Euclidean space.  

**Bottom layer:** Finally, PIGEON selects the single closest location within that cluster—adding two extra levels of granularity beyond the initial cell prediction.

| ![Three-level hierarchical retrieval diagram](/images/threeLayers.png) |
|:---:|           
|Figure 2: The different layers used for retrieval. |

To integrate these multi-scale cues, PIGEON multiplies each geocell’s original probability by a refinement score derived from a temperature-scaled softmax over the cluster distances. This joint scoring across all top K cells yields a final geolocation estimate that balances global confidence with local embedding proximity.




---

## Experiments

### Experimental Setting


Now that we know how the methods behind the model, we still need to explain what data the models were trained on since here lies the main difference between PIGEON and PIGEOTTO besides the setting of some hyperparameters. Then for each model the authors use a evaluation method specifically tailored for the use case of the model using both synthetic setups and real-world conditions. 

**For PIGEON**, which is optimized for Street View images like in GeoGuessr, the authors collected a dataset of 100,000 locations from the game. At each location, they captured a 360-degree view using four images spaced evenly around the compass. This panorama-style input helps the model detect geographical cues like vegetation. In total, the training data amounted to 400,000 images.

To evaluate PIGEON’s accuracy, the researchers used a separate holdout set of 5,000 unseen GeoGuessr locations. Importantly, they didn’t just rely on offline metrics. They also deployed PIGEON live into GeoGuessr using a custom Chrome extension bot. This allowed for direct comparisons against players across all skill levels including one of the world’s top-ranked professionals. 

| ![Geoguessr Panorama](/images/panoramaGeoguessr.png) |
|:---:|           
|Figure 3: Four images comprising a 360-degree panorama from a location in Pegswood, England|

**For PIGEOTTO**, the broader model designed to handle general-purpose geolocation from a single image, the researchers gathered over 4.5 million images from Flickr and Wikipedia (including landmarks from the Google Landmarks v2 dataset and the Flickr images from the MediaEval 2016 dataset). Instead of Street View, these images came from diverse, user-generated content around the world.


PIGEOTTO was tested on several standard geolocation benchmarks used in the literature including IM2GPS, IM2GPS3k, YFCC4k,  YFCC26k and GWS15k.
These benchmarks evaluate how many guesses fall within different distance thresholds (like 25 km or 200 km from the correct location) and measure median error in kilometers.

| ![PIGEOTTO images](/images/pigeottoRandom.png) |
|:---:|           
|Figure 4: Four samples from the MediaEval 2016 dataset|

The primary and composite metric used to evaluate the models is the median distance error to the correct location. Just like prior literature on image geolocalization the author evaluate "% @ km" statistic in their analysis for a more fine-grained metric. The "% @ km" statistic determines the percentage of guesses that fall within a given kilometer-based distance from the ground-truth location. This leads to five distance radii:

- 1 km (roughly street-level accuracy)
- 25 km (city-level)
- 200 km (region-level)
- 750 km (country-level)
- 2500 km (continent-level)

To summarize, while PIGEON was built to master the game of GeoGuessr using panoramic Street View inputs, PIGEOTTO was trained for general image geolocation on a planetary scale. The careful separation of data sources and testing conditions ensured that each model was evaluated fairly in its respective domain.
 



### Qualitative Analysis

Now that we've covered the architecture and the training data of the models, we showcase the capabilities of these models by presenting some qualitative results. The qualitative results not only illustrate the models’ reasoning but also highlight the kinds of cues they’ve learned to pick up, often mirroring expert GeoGuessr strategies.

**Interpreting Model Attention for PIGEON**  
The authors generated attention-attribution maps over the CLIP-based backbone to see which parts of a Street View image drive PIGEON’s geolocation predictions.

| ![Attention Attribution Map](/images/attentionAttributionMap.png) |
|:---:|           
|Figure 5: Attention attribution map for an image in New Zealand|

When given a view of a New Zealand countryside, PIGEON’s attention is mainly focused on the following cues:
- Road markings and signs specific to certain countries, like center lines and stop signs.  
- Utility poles, which vary significantly even within a country, giving a strong hint about where the image was taken.

This kind of behaviour is very interesting since it mimics that of GeoGuessr professionals, who also tend to look for utility poles and street signs which are features that are typically strong indicators of location. The model does so without being explicitly told to do so, underlining its generalizability and performance.

**Uncertainty Across Scenes for PIGEON**

Next, we present some examples where PIGEON is most uncertain about which geocell to guess.

| ![Dark Panorama](/images/darkPanorama.png) |
|:---:|           
|Figure 6: Example of an image for which PIGEON was most uncertain.|

In the above image, it is expected that PIGEON cannot make a reasonable guess since virtually nothing is recognizable. High uncertainties in these kinds of scenarios with out-of-distribution samples are bound to happen and are acceptable. The next image, however, is more interesting.

| ![Uncertain Forest](/images/uncertainForestPanorama.png) |
|:---:|           
|Figure 7: Example of an image for which PIGEON was most uncertain.|

Here we see a road going through a forest with dense vegetation, but there aren't any road markings or street signs visible. Still, the vegetation itself provides significant cues about the country or region where the image was taken. Interestingly, PIGEON has a hard time guessing the correct geocell in this instance, underlining that the model can still improve in these kinds of scenarios.

**Diverse Predictions by PIGEOTTO**

Having tested PIGEON on some Street View inputs, we now examine PIGEOTTO’s ability to geolocate arbitrary user images. Below are two representative cases:

| ![Capilano Bridge](/images/bridgePIGEOTTO.png) |
|:---:|           
|Figure 8: Sample and guess of the Capilano Suspension Bridge, Canada.|

In this landmark photograph, PIGEOTTO which is trained on landmarks can easily guess where the image was taken. For this guess it might leverage the fact that it might has seen another photograph of this particular landmark it training allowing for an easy guess. As a result, PIGEOTTO makes a guess that is only 3 km off from the true position highlighting the ability of the model to correctly pinpoint landmarks.

| ![Body of Water with Buoy](/images/bodyOfWaterWithBuoy.png) |
|:---:|           
|Figure 9: Sample of a floating buoy on the coast of Denmark.|

By contrast, this is a generic image of a buoy floating on a body of water. PIGEOTTO still identifies that the image is taken from or near the sea, but its highest-probability guess lands off the coast of the northeastern United States—over 5500 km from Denmark. This large error underscores how maritime scenes without distinct, place-specific markers remain a significant challenge for geolocation models.

These examples demonstrate PIGEOTTO’s proficiency in leveraging learned visual cues for precise landmark localization, while also exposing its limitations when distinctive, place‐specific features are absent.



### Quantitative Analysis

Now that we've covered the qualitatitve analysis we jump into some numbers. 

**PIGEON**
 To assess PIGEON's performance the authors evaluated the model on a hold-out data set of 5,000 GeoGuessr locations and conducted blind “duel” experiments against actual human players. On the hold out dataset PIGEON shows a country-level accuracy of 92%, while just having a median distance error of 44.4 km. In addition to this evaluation on the hold-out data set the Authors also put PIGEON to the test by letting it play against actual humans player in 458 multi-round duels. PIGEON played against player in the gold, master and champion divison. Here players in the master-division are more skilled at the game than players in the gold-division. The same goes for the champion-division and master-division players. In the bar chart below it can be seen that PIGEON comfortably beats even players from the champion-divison that consist of the 0.01 % of the top Geoguessr players. It does this by quite a lot since it has half the error of those 0.01 % players. 

 | ![Geoguessr Divisions](/images/geoGuessrDivisionsEval.png) |
|:---:|           
|Figure 10: Geolocalization error of PIGEON against human players of various in-game skill levels across 458 multi-round matches. The Champion Division consists of the top 0.01% of players. The median error is higher since GeoGuessr round difficulties are adjusted dynamically, increasing with every round.| 

In addition to that we can really recommend watching the following video where the authors put their model to the test against an actual Geoguessr professionals in a live match:

[▶️ Watch PIGEON beat a GeoGuessr professional on YouTube](https://www.youtube.com/watch?v=ts5lPDV--cU)

**PIGEOTTO**

After looking at the results from PIGEON, we now turn to PIGEOTTO’s performance on the various benchmark datasets on which the model was evaluated.

| ![PIGEOTTO Benchmark Results](/images/pigeottoBenchmarkResults.png) |
|:---:|           
|Figure 11: Comparison of PIGEOTTO’s results against other models on benchmark datasets.|

The first mentioned benchmark is IM2GPS which consists of only 237 primarily landmark images. While PIGEOTTO is worse than the best prior model on a smaller granularities like street-, city- and region-level which might be the case because of the rather small test data set. Still it can improve performance on a country- and continent-level by 2 percent.

On the larger and more diverse IM2GPS3k dataset, PIGEOTTO performs exceptionally well except at the street level, where it underperforms by about 1.5 %. At all other levels, the model significantly outperforms its predecessors, delivering an performance boost of 11.4 % on a country level.

On YFCC4k and YFCC26k, PIGEOTTO achieves state-of-the-art results, with a 12.2 % increase in country-level accuracy on YFCC4k and a +13.6 % increase on YFCC26k. This demonstrates that PIGEOTTO generalizes effectively to unseen locations, excelling not only on smaller benchmarks but also on larger, more diverse datasets.

Finally, on GWS15k considered the toughest benchmark with 15,000 entirely unseen Street View panoramas by the authors. PIGEOTTO cuts the median error from ∼2,500 km down to 415.4 km and maintains street-level accuracy at 0.7 %. More impressively, it boosts city-level by +7.7 %, region-level by +22.5 %, country-level by +38.8 %, and continent-level by +34.6 %, underscoring its true planet-scale generalizability to brand-new places.

### Ablations

To quantify the contribution of each major component in PIGEON, the authors performed an extensive ablation study, removing one feature at a time and measuring its impact on country-level accuracy, localization errors, and GeoGuessr score. 

| ![Ablations](/images/ablations.png) |
|:---:|           
|Figure 12: Cumulative ablation study of the authors' image geolocalization system on a holdout dataset of 5,000 Street View
locations.|

Removing the four-image panorama input has the most dramatic effect on PIGEON’s performance. Without it, country­-level accuracy plunges by nearly 13 % and the mean localization error roughly triples. This makes sense since three additional images of the surroundings introduce new information to make a better guess.
Also omitting the haversine-smoothed loss in training leads to a drop of over 2 % on the country-level accuracy. Additionally, the mean and median errors worsen significantly without this novel loss function, likely due to the edge cases discussed in the “Distance-Based Label Smoothing” section.
Ablating semantic geocells leads to a drop of over 2 % on the country-level and increasing the median error from 55.5 to 60.6 kilometers. This showcases the importance of choosing semantically meaningful cells capturing specific geographical cues like street signs and road markings.
Adding contrastive CLIP pretraining yields a 1.7 % boost in country-level accuracy, demonstrating the benefit of synthetic captions during training.
In contrast, disabling hierarchical refinement yields only modest changes (under 0.8 % in accuracy), suggesting that while they contribute useful refinements, they are less central than geocell semantics, panoramic inputs, and distance-aware loss to PIGEON’s state-of-the-art geolocation capabilities.

No ablation studies were conducted for PIGEOTTO.



---

## Ethical Considerations

### Ethical Considerations

### Ethical Considerations

Before outlining specific concerns, it’s important to acknowledge that advances in image geolocalization bring both powerful benefits and potential harms. The following points highlight key ethical issues to address as this technology matures:

- **Dual-Use Risks:** Systems like PIGEON and PIGEOTTO can greatly enhance scientific research, environmental monitoring, and navigation in remote areas by accurately determining locations from images. However, these same capabilities can be repurposed for covert surveillance, unauthorized tracking of individuals or assets, and even military targeting. Without clear oversight and accountability mechanisms, dual-use applications pose a significant risk of abuse.

- **Privacy Threats:** High-precision geolocation models are capable of inferring sensitive personal information, such as daily movement patterns, frequented locations, and demographic attributes like socioeconomic status or language use from seemingly normal photographs. In the absence of strong anonymization safeguards and explicit user consent, individuals and communities may be exposed to unwanted profiling, discrimination, or privacy violations.

- **Controlled Release:** To balance the need for transparency with the imperative to prevent misuse, the authors release the full algorithmic code and comprehensive documentation (including model cards and data statements) while withholding pretrained model weights. This approach supports reproducibility and peer review by the academic community, yet limits the ability to deploy powerful geolocalization models at scale without additional oversight.

These considerations illustrate that as image geolocalization technology advances, robust mechanisms must be in place to guarantee its ethical use.



---

## Conclusion

PIGEON and its extension PIGEOTTO represent a major advance in automated image geolocalization, combining powerful visual embeddings with spatially aware training objectives to achieve state-of-the-art accuracy on both Street View panoramas and arbitrary photographs. Through extensive qualitative and quantitative evaluations, including direct GeoGuessr duels and large-scale benchmark tests, their hybrid classification-and-retrieval framework has proven both robust and generalizable.

That said, PIGEON still encounters scenarios where uncertainty remains high (showcased in the qualitative results section). Future work could address these gaps by refining the model architecture, incorporating more training samples for hard cases. At the same time, it’s essential to uphold transparency, accountability, and ethical governance so that this powerful technology continues to serve the public good.





---

## References
1. Ankerst, M., Breunig, M. M., Kriegel, H.-P., and Sander, J. OPTICS: Ordering Points to Identify the Clustering Structure. In Proceedings of the 1999 ACM SIGMOD International Conference on Management of Data, SIGMOD ’99, pp. 49–60, New York, NY, USA, 1999. Association for Computing Machinery. ISBN 1581130848. doi:10.1145/304182.304187. URL https://doi.org/10.1145/304182.304187.  
2. Beck, H. E., Zimmermann, N. E., McVicar, T. R., Vergopolan, N., Berg, A., and Wood, E. F. Present and future Köppen-Geiger climate classification maps at 1-km resolution. Scientific Data, 5(1):180214, Oct 2018. ISSN 2052-4463. doi:10.1038/sdata.2018.214. URL https://doi.org/10.1038/sdata.2018.214.  
3. Cao, L., Smith, J. R., Wen, Z., Yin, Z., Jin, X., and Han, J. BlueFinder: Estimate Where a Beach Photo Was Taken. In Proceedings of the 21st International Conference on World Wide Web, WWW ’12 Companion, pp. 469–470, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450312301. doi:10.1145/2187980.2188081. URL https://doi.org/10.1145/2187980.2188081.  
4. Haas, L., Skreta, M., Alberti, S., and Finn, C., 2024, PIGEON: Predicting Image Geolocations. Accepted at CVPR 2024. arXiv:2307.05845 [cs.CV]. URL https://doi.org/10.48550/arXiv.2307.05845.  
5. Hays, J. and Efros, A. A. IM2GPS: estimating geographic information from a single image. In Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2008.  
6. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., and Adam, H. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017. URL https://arxiv.org/abs/1704.04861.  
7. Kolesnikov, A., Dosovitskiy, A., Weissenborn, D., Heigold, G., Uszkoreit, J., Beyer, L., Minderer, M., Dehghani, M., Houlsby, N., Gelly, S., Unterthiner, T., and Zhai, X. An image is worth 16×16 words: Transformers for image recognition at scale. 2021.  
8. Luo, G., Biamby, G., Darrell, T., Fried, D., and Rohrbach, A. Gˆ3: Geolocation via Guidebook Grounding. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 5841–5853, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL https://aclanthology.org/2022.findings-emnlp.430.  
9. Masone, C. and Caputo, B. A Survey on Deep Visual Place Recognition. IEEE Access, 9:19516–19547, 2021. doi:10.1109/ACCESS.2021.3054937.  
10. Müller-Budack, E., Pustu-Iren, K., and Ewerth, R. Geolocation Estimation of Photos Using a Hierarchical Model and Scene Classification. In Ferrari, V., Hebert, M., Sminchisescu, C., and Weiss, Y. (eds.), Computer Vision – ECCV 2018, pp. 575–592, Cham, 2018. Springer International Publishing. ISBN 978-3-030-01258-8.  
11. OpenAI. GPT-4V(ision) System Card, September 2023.  
12. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning Transferable Visual Models From Natural Language Supervision, 2021.  
13. Saurer, O., Baatz, G., Köser, K., Ladický, L., and Pollefeys, M. Image Based Geo-localization in the Alps. International Journal of Computer Vision, 116(3):213–225, Feb 2016. ISSN 1573-1405. doi:10.1007/s11263-015-0830-0. URL https://doi.org/10.1007/s11263-015-0830-0.  
14. Seo, P. H., Weyand, T., Sim, J., and Han, B. CPlaNet: Enhancing Image Geolocalization by Combinatorial Partitioning of Maps, 2018.  
15. Suresh, S., Chodosh, N., and Abello, M. DeepGeo: Photo Localization with Deep Neural Network, 2018. URL https://arxiv.org/abs/1810.03077.  
16. Suresh, S., Chodosh, N., and Abello, M. DeepGeo: Photo Localization with Deep Neural Network, 2018. URL https://arxiv.org/abs/1810.03077.  
17. Tomešek, J., Cadík, M., and Brejcha, J. CrossLocate: Cross-Modal Large-Scale Visual Geo-Localization in Natural Environments Using Rendered Modalities. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 3174–3183, January 2022.  
18. Tzeng, E., Zhai, A., Clements, M., Townshend, R., and Zakhor, A. User-Driven Geolocation of Untagged Desert Imagery Using Digital Elevation Models. In 2013 IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 237–244, 2013. doi:10.1109/CVPRW.2013.42.  
19. Weyand, T., Kostrikov, I., and Philbin, J. PlaNet - Photo Geolocation with Convolutional Neural Networks. In European Conference on Computer Vision (ECCV), 2016.  
20. Zamir, A. R. and Shah, M. Image Geo-Localization Based on Multiple Nearest Neighbor Feature Matching Using Generalized Graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(8):1546–1558, 2014. doi:10.1109/TPAMI.2014.2299799.  

