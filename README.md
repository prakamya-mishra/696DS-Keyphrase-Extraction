# 696DS-Keyphrase-Extraction

Keyphrases capture the most salient words in and across sentences. Identifying them in an automated way from a text document can be useful for several downstream tasks - classification (Hulth and Megyesi, 2006), semantic and faceted search (Sanyal et al., 2019; Gutwin et al., 1999) and query expansion (Song et al., 2006). Keyphrases could either be extractive (part of the document) or abstractive (not part of the document). Recent work on pre-training transformers by leveraging keyphrases (Kulkarni et al., 2022) shows significant performance improvements both in the classification (Keyphrase Extraction) and generation (Keyphrase Generation) settings over previous baselines.


The authors of the paper pre-train the transformers on scientific research articles and limited experiments show that such transformers perform well on out-domain tasks like NER on CoNLL-2003 (News) and zero-shot results seem promising. However, a deep analysis on out-of-domain performance as well as quantifying the zero-shot and few-shot capabilities on Keyphrase Extraction and Keyphrase Generation tasks are yet to be explored.


The focus of this project would be to explore the KBIR and KeyBART models from Kulkarni et al., 2022 in cross and multidomain settings and also explore the effectiveness of these models in few and zero-shot settings:
Out of domain datasets would potentially include StackEx (Yuan et al., 2020), OpenKP (Xiong et al., 2019), KPCrowd (Marujo et al., 2012) and KPTimes (Gallina et al., 2019).
We would work towards building cross domain models by leaving a domain out during training and evaluating it as zero shot.
Similar to NER experiments in Wang et al., 2020, we would work on building a multidomain model that doesn't deteriorate performance on the evaluated datasets in the paper but also performs well on the new out of domain data mentioned above.
Few and zero shot experiments, would explore the effect of freezing and gradual freezing the encoders and decoders and also the potential need for Task Adaptive Pretraining (Gururangan et al., 2020).
