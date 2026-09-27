This CVPR Findings paper is the Open Access version, provided by the Computer Vision Foundation. Except for this watermark, it is identical to the accepted version; the final published version of the proceedings is available on IEEE Xplore. 

# **HeartcareGPT: A Unified Multimodal ECG Suite for Dual Signal–Image Modeling and Understanding** 

Yihan Xie<sup>1*</sup> Sijing Li<sup>1*</sup> Zhuonan Wang<sup>1*</sup> Tianwei Lin<sup>1*</sup> Chenglin Yang<sup>1</sup> Yu Zhong<sup>1</sup> Wenjie Yan<sup>1</sup> Wenqiao Zhang<sup>1†</sup> Xiaogang Guo<sup>1</sup> Jun Xiao<sup>1</sup> Yueting Zhuang<sup>1</sup> Beng Chin Ooi<sup>1</sup> 1Zhejiang University 

{yihanxie, wenqiaozhang}@zju.edu.cn 

## **Abstract** 

_Although electrocardiograms (ECG) play a dominant role in cardiovascular diagnosis and treatment, their intrinsic data forms and representational patterns pose significant challenges for medical multimodal large language models (Med-MLLMs) in achieving cross-modal semantic alignment. To address this gap, we propose_ **_Heartcare Suite_** _, a unified ECG suite designed for dual signal–image modeling and understanding:_ **_(i) Heartcare-400K._** _A finegrained ECG instruction dataset on top of our data pipeline engine—_ **_HeartAgent_** _—by integrating high quality clinical ECG reports from top hospitals with open-source data._ **_(ii) Heartcare-Bench._** _A systematic benchmark assessing performance of models in multi-perspective ECG understanding and cross-modal generalization, providing guidance for optimizing ECG comprehension models._ **_(iii) HeartcareGPT._** _Built upon a structure-aware discrete tokenizer_ **_Beat_** _, we propose_ **_Dual Stream Projection Alignment (DSPA)_** _paradigm— a dual encoder projection alignment mechanism enabling joint optimizing and modeling native ECG signal–image within a shared feature space. HeartcareGPT achieves consistent improvements across diverse ECG understanding tasks, validating both the effectiveness of the unified modeling paradigm and the necessity of a high-quality data pipeline, and establishing a methodological foundation for extending Med-MLLMs towards physiological signal domains. Our project is available at https://github. com/ZJU4HealthCare/HeartcareGPT._ 

## **1. Introduction** 

Multimodal large language models (MLLMs) [2, 4, 6, 11, 21, 27, 33, 36, 43, 50, 51] demonstrate strong performance in general-purpose scenarios by jointly modeling multiple 

> *Equal contributions. 

> †Corresponding author. 

modalities such as text, image and video. Motivated by their success in open-domain scenarios, researchers have extended the MLLM paradigm to the medical domain, giving rise to medical multimodal large language models (Med-MLLMs), including LLaVA-Med [20], HuatuoGPTVision [10], HealthGPT [25], and Lingshu [48]. These models show potential for medical diagnosis and reasoning, driving progress in intelligent healthcare applications. 

In fact, existing Med-MLLMs are constrained by visioncentric paradigms that focus on imaging data [30, 56], yet medical applications frequently require integration of nonimaging modalities and domain-specific knowledge [42]. Electrocardiograms (ECG) illustrates this limitation, as it encodes critical diagnostic information in a dual signal–image form, tightly combining temporal and morphological features [5, 47]. Consequently, ECG not only delineates the failure boundary of existing approaches but also highlights the fundamental rationale for developing dedicated MedMLLMs: _the structural characteristics of medical modalities must serve as the foundation of model design, rather than an afterthought for adaptation._ 

Intelligent ECG interpretation relies on symbolic semantics to integrate patient history, ECG data, and clinical reports into interpretable reasoning chains. This paradigm, naturally suited to MLLMs, still faces fundamental challenges: **Data Misalignment.** Mainstream ECG datasets such as PTB-XL [46] are designed for rhythm and disease classification, with annotations at the global level and lacking step-wise supervision. As a result, these datasets mainly enable direct mapping from ECG signals to labels, making it difficult to support interpretable reasoning or cross-modal information integration, and thus limiting the shift from simple classification to deeper understanding and reasoning. **Evaluation Deficiency.** Existing medical benchmarks primarily emphasize classification accuracy and tend to overlook the assessment of comprehensive reasoning processes [29, 44]. For instance, they rarely evaluate models’ capabilities in clinical reasoning or pathology question– answering (QA) for ECG tasks. Therefore, it is essential to develop multidimensional assessment frameworks that include reliability and scenario-based metrics for evaluating the clinical utility of ECG-specific Med-MLLMs. 


![The proposed Heartcare-400K dataset. Heartcare-400K aggregates real-world ECG data, supporting Closed-QA , Open-QA , Comparison-QA , Report Generation and Signal Prediction.](fig_1.png)


Figure 1. The proposed **Heartcare-400K** dataset. Heartcare-400K aggregates real-world ECG data, supporting _Closed-QA_ , _Open-QA_ , _Comparison-QA_ , _Report Generation_ and _Signal Prediction_ . 



**Dual Signal–Image Fusion Incapacity.** The dual nature of ECG data—comprising both raw temporal signals and waveform images—poses unique modeling challenges [37, 41]. While signals capture dynamic cardiac activity and images reflect spatial morphology, both are essential for diagnosis. Most vision-focused MLLMs, however, process only ECG images [3, 31], neglecting key temporal features in the signal, such as intervals, calibration, inter-lead phase, and beat-to-beat changes. Moreover, multi-lead synchronous acquisition results in a tightly coupled, heterogeneous structure across temporal and spatial domains [23]. Addressing this dual complexity requires Med-MLLMs capable of consistent multimodal modeling and alignment. 

We propose **Heartcare Suite** —a systematic and innovative framework for ECG, dedicated to establishing a unified and extensible ecosystem for ECG-specific Med-MLLMs: 

**(i) Dataset** . We construct **Heartcare-400K** , a large-scale, fine-grained, multi-task multimodal ECG instruction dataset. It combines two sources: the public PTB-XL dataset [46] with 21,799 12-lead ECG signals annotated with 179 SCPECG classes, and 12,170 ECG images with structured reports from top hospitals, including scanned traces, clinical conclusions, and de-identified metadata—substantially enriching modality and label diversity. To transform heterogeneous ECG data into structured annotations, we develop **HeartAgent** , a multimodal engine with a bottom-up pipeline that 

ensures annotation consistency and generates high-quality instruction-style QA pairs. 

**(ii) Benchmark** . We propose **Heartcare-Bench** , the first fine-grained, multidimensional evaluation framework for ECG diagnostic intelligence, designed to assess a spectrum of model capabilities ranging from feature recognition to reasoning. Built upon Heartcare-400K, Heartcare-Bench systematically covers five major task types— _Closed-QA_ , _Open-QA_ , _Comparison-QA_ , _Report Generation_ , and _Signal Prediction_ —spanning key diagnostic dimensions such as rhythm, waveform, and morphology. It comprises three complementary modality subsets: _Signal_ (S), _Image_ (I), and _Cross-Modal_ (C), enabling unified evaluation from singlemodality reasoning to multi-ECG semantic alignment. With a hierarchical, multi-metric scoring system, Heartcare-Bench integrates knowledge reasoning and cross-modal understanding within a unified evaluation coordinate. 

**(iii) Model.** The dual-form characteristics of ECG introduce unique structural complexity in modeling. We propose **HeartcareGPT** , aiming to build ECG-specific Med-MLLMs. We design **Bidirectional ECG Abstract Tokenization (Beat)** , a structure-aware discrete encoding mechanism centered on vector quantization [12, 45, 54], which maps highfrequency continuous signals into token sequences. The design comprises three components: (i) _Dual-level Vector Quantization (DVQ)_ , which refines rhythm and inter-lead phase dependencies captured by the codebook to achieve high-fidelity compression; (ii) _Query-guided Bidirectional Diffusion (QBD)_ , which jointly models past and future contexts within the latent token space to support both signal reconstruction and prediction; and (iii) _Joint Supervision Strategy_ , which jointly optimizes reconstruction and prediction to maximize clinical semantic fidelity during encoding. Furthermore, we propose **Dual Stream Projection Alignment (DSPA)** , which employs dual experts to separately process ECG inputs. Through distinct preprocessing strategies and modality-specific encoders, ECG representations are transformed into embeddings compatible with Med-MLLMs. All modality embeddings are projected into a shared language space and concatenated into a unified sequence, enabling cross-modal joint reasoning for ECG under a unified autoregressive paradigm. 

Experimental results demonstrate the powerful paradigm of Heartcare Suite. The main contributions of this work are as follows: 

- **High-quality ECG Instruction Dataset.** Heartcare-400K serves as the first comprehensive ECG instruction dataset, which significantly enhances Med-MLLM performance across ECG-related tasks. 

- **Multidimensional ECG Benchmark.** We propose Heartcare-Bench, an evaluation framework that assesses clinical performance of ECG tasks for Med-MLLMs. 

- • **Fine-grained ECG Understanding Paradigm.** We develop HeartcareGPT, the first model supporting pathologylevel ECG understanding with state-of-the-art (SOTA) results. 

## **2. Related Work** 

**Multimodal Representation Learning for ECG.** Recent advances in multimodal ECG representation learning follow three main directions. First, signal-semantic alignment. ECG-SL [52] and MERL [26] align heartbeats and clinical reports via self-supervision and knowledge prompting, while HeartLang [19] decomposes waveforms into semantic tokens for fine grained cardiac analysis. Second, cross-lead fusion. ECG-DAN [9] adopts a dual attention network to balance global cross lead interactions with local temporal dynamics, and ESI [53] adds a signal text contrastive learning to strengthen robustness under limited labels. Third, large language model (LLM)-driven pretraining. ECG-LM [49] maps ECG embeddings into a pretrained language space, and SuPreME [7] structures domain knowledge from clinical reports for pretraining. However, existing approaches typically address isolated aspects such as reconstruction quality, lead interaction, or semantic alignment, without forming a unified end-to-end multimodal ECG modeling framework. **Medical Multimodal Large Language Models.** MedMLLMs have shown strong capabilities in medical understanding and diagnostic support [10, 18, 35, 48]. MedFlamingo [29] and LLaVA-Med [20] are representative early medical multimodal models, focusing on image–text alignment and visual question answering. MedVLM [35] 

employs multi-stage pretraining and attains SOTA performance in radiology report generation and organ localization. HealthGPT [25] unifies image understanding and generation within one framework. Domain-specific variants—LLaVARad [8], EyecareGPT [22], and SkinGPT-4 [57]—enable structured reporting and multimodal reasoning across radiology, ophthalmology, and dermatology. However, the dualform characteristics of ECG presents significant challenges to vision-centric models, and an effective ECG-specific MedMLLM framework remains absent to date. 

## **3. Heartcare Suite: Heartcare-400K** 

### **3.1. Data Collection and Organization** 

Existing ECG datasets suffer from limited modalities, coarse annotations, insufficient scale, and substantial heterogeneity in signal sampling, lead configuration, and preprocessing, all of which hinder the development of Med-MLLMs for intelligent ECG diagnosis. To address these gaps, we introduce **Heartcare-400K** , a large-scale multimodal ECG QA dataset with two complementary modalities: **(i) structured digital signals** and **(ii) unstructured ECG report images** . We normalize them by rendering all signals into a unified ECG visual format consistent with clinical layout. These visualized signals, together with native report images, enrich modality diversity and support comprehensive ECG understanding. 

To construct Heartcare-400K, we collaborated with several major public hospitals to collect 12,170 standardized PDF-format clinical ECG reports. These reports include patient demographics, physiological parameters, physician diagnoses, and about 5 seconds of 12-lead waveform images. In addition, we systematically integrated multiple publicly available digital signal ECG datasets, with PTB-XL [46] as the primary source. PTB-XL is one of the largest open ECG repositories, containing 21,799 12-lead records sampled at 500 Hz over 10 seconds, with standardized diagnostic labels and detailed patient metadata (such as gender, age and weight). 

Raw ECG data typically contain only brief diagnostic text and lack the instruction-style supervision and semantic diversity needed for MLLM fine-tuning. To address this, we developed a multimodal data engine for automated extraction, cleaning, standardization, and expert review, and further augmented the dataset with QA pairs from the publicly available ECG-QA [32], which supports both single-modality queries and cross-modality reasoning tasks. 

Ultimately, Heartcare-400K is organized into four types of QA tasks: (i) _Closed-QA_ (single-ECG multiple-choice question), (ii) _Open-QA_ (single-ECG short-form question), (iii) _Comparison-QA_ (multi-ECG multiple-choice question), (iv) _Report Generation_ (long-form answers), and (v) _Signal Prediction_ (ECG generation). As Figure 1 shows, these tasks equip models with fine-grained ECG comprehension and clinical reasoning capabilities, making Heartcare-400K a foundational resource for developing practical and generalizable intelligent ECG diagnosis systems. 



![Framework of HeartAgent for QA generation](fig_2.png)

Figure 2. Framework of HeartAgent for QA generation. 

### **3.2. Multimodal Data Engine** 

To efficiently construct Heartcare-400K, we design **HeartAgent** (as shown in Figure 2), an automated multimodal data engine with two key stages: _raw data processing_ and _data building_ . Details of each module are provided in the Appendix A. 

**Raw Data Processing.** This stage integrates three core components to transform heterogeneous ECG sources into standardized inputs: **(i) Multimodal Feature Converter.** Extracts patient information and clinical descriptions from hospital PDFs, maps diagnostic text to structured labels, and prepares both text and image data for subsequent processing. **(ii) Noise Filtering and Quality Optimizer.** Unifies signal formats and applies denoising, resampling, and quality assessment to produce high-quality, standardized ECG signals. **(iii) Diversified Image Generator.** Renders clean waveform images from processed digital signals and generates de-identified tracings from hospital reports, ensuring multi-modal data alignment. 

**Data Building.** This stage is handled by the **Multi-task QA Builder** , which constructs instruction-style QA samples using GPT-4 [33]. Each sample contains structured context, clear task instructions, auxiliary labels, and standardized answers, supporting four diverse ECG understanding tasks and a _Signal Prediction_ task critical for comprehensive model training. 

These components collectively ensure Heartcare-400K achieves high-quality, diverse, and task-relevant multimodal annotations. 

## **4. Heartcare Suite: Heartcare-Bench** 

To systematically evaluate the capabilities of Med-MLLMs across unified ECG understanding and prediction tasks, we 

propose Heartcare-Bench—a multidimensional benchmark encompassing five major task types: _Closed-QA_ , _Open-QA_ , _Comparison-QA_ , _Report Generation_ , and _Signal Prediction_ . Heartcare-Bench is structured into three complementary subsets based on input modality: **(i) Heartcare-Bench**<sup>**S**</sup> (signal-based), **(ii) Heartcare-Bench**<sup>**I**</sup> (image-based), and **(iii) Heartcare-Bench**<sup>**C**</sup> (cross-modal comparison). In addition, Heartcare-Bench employs strict patient-level data partitioning with thorough cross-split duplicate inspection; full details are provided in Appendix B.4. 

**Heartcare-Bench**<sup>**S/I**</sup> **: Single-Modality Subsets.** The signal and image subsets share a unified framework to evaluate four core clinical dimensions: _Diagnosis_ , _Waveform analysis_ , _Rhythm interpretation_ and _Miscellaneous Features_ . To broaden task diversity and clinical relevance, we further integrate QA pairs from ECG-QA [32], primarily those derived from PTB-XL, enriching _Closed-QA_ and _Open-QA_ settings and enabling categorization by these four dimensions. The _Miscellaneous (Misc.)_ category further assesses ECG attributes such as noise, infarction stage, ectopic beats, and cardiac axis. In addition to QA tasks, the single-modality subsets also include _Report Generation_ and _Signal Prediction_ tasks. Detailed formats and examples are provided in Appendix B.4. 

**Heartcare-Bench**<sup>**C**</sup> **: Cross-Modality Comparison Subset.** Beyond single-modality tasks, this subset evaluates a model’s ability to reason over multiple ECGs through comparison QA tasks adapted from ECG-QA [32]. It is centered around three modality comparison dimensions: (i) _signal– signal (S–S)_ , (ii) _image–image (I–I)_ , and (iii) _signal–image (S–I)_ . Each configuration is further categorized into two clinically relevant subtypes: consecutive (ECGs from the same patient, assessing temporal consistency) and irrelevant (ECGs from different patients, evaluating differential diagnostic reasoning). 

Overall, Heartcare-Bench adopts a multidimensional evaluation protocol covering semantic consistency, linguistic fluency, clinical accuracy, and waveform prediction quality. Full scoring rules and evaluation details are provided in Appendix B.4. 

## **5. Methodology** 

### **5.1. Bidirectional ECG Abstract Tokenization** 

We present a structure-aware discrete encoding mechanism centered on vector quantization— **B** idirectional **E** CG **A** bstract **T** okenization **(Beat)** . The architecture of Beat is depicted in Figure 3 (a). 

**Forward Diffusion Process.** Given an acquired ECG signal, we denoise and resample it to obtain a representative segment **S** _∈_ R<sup>_T ×C_</sup> (sequence length _T_ , _C_ leads). We then partition **S** into contiguous, non-overlapping temporal patches and map them with a projection layer Ψ to yield the corresponding patch embeddings:

![Architecture of HeartcareGPT. (a) The dual-form ECG inputs are routed and encoded with modality-specific expert projections aligned to the LLM backbone. (b) The unified autoregressive architecture efficiently supports interleaved and joint modeling of ECG multimodal inputs.](fig_3.png)

Figure 3. Architecture of HeartcareGPT. (a) The dual-form ECG inputs are routed and encoded with modality-specific expert projections aligned to the LLM backbone. (b) The unified autoregressive architecture efficiently supports interleaved and joint modeling of ECG multimodal inputs. 


where _f_ is the patch frame size, _t_ the number of patches, and _θ_ Ψ the projection parameters. To inject high-level semantic control, we append _m_ learnable queries **q** _∈_ R<sup>_m×c_</sup> to the patch embeddings **e** , forming the model input _H_ in = [ **e** _∥_ **q** ]. A Transformer encoder then performs forward diffusion, yielding compressed contextual representations: 



**Dual-level Vector Quantization.** To compress ECG signals while preserving rhythmic structure and salient pathological cues, we adopt dual-level VQ with a core codebook _C_ 1 and a residual codebook _C_ 2. For each query vector **h**<sup>_i_</sup> _q_<sup>=(</sup><sup>_H_</sup> latent<sup>_q_)[</sup><sup>_i,_:]</sup><sup>_∈_R</sup><sup>_c_,wefirstapplythecorequantiza-</sup> tion: 



Finally, we quantize the forward-diffused representation to **h**<sup>ˆ</sup><sup>_i_</sup> latent<sup>=</sup><sup>**h**ˆ</sup><sup>_i_</sup> _⟨q,_ 1 _⟩_<sup>+</sup><sup>**h**ˆ</sup><sup>_i_</sup> _⟨q,_ 2 _⟩_<sup>,yieldingahigh-fidelitymap-</sup> ping from continuous latents to a discrete code space and significantly improving reconstruction quality. **Query-guided Bidirectional Diffusion.** Forward diffusion (Eq. 2) compresses the ECG into a dense discrete space 

but offers no guarantees of completeness or invertibility. We therefore adopt an autoencoding view and introduce a reverse diffusion anchored at the quantized query _H_ latent<sup>_q_,</sup> casting compression and reconstruction as a joint constraint: the forward path enforces compact coding, the reverse path restores details, yielding a closed-loop, bidirectional token refinement between latent and signal spaces. 

During reverse diffusion, the original input **e** is mapped to its forward-diffusion slot _H_ origin<sup>_q_andmaskedby</sup><sup>_M_to</sup> prevent leakage, so only the query vectors _H_ latent<sup>_q_retain ECG</sup> signal information for reconstruction: 



**Joint Supervision Strategy.** To fully exploit the bidirectional diffusion capacity, Beat employs a multi-objective loss that jointly optimizes reconstruction and compression, integrating three objectives. The reconstruction and prediction losses are defined as: 



where **e** pred denotes the ECG segment used for prediction. The vector quantization loss is defined as: 



where sg[ _·_ ] denotes the stop-gradient operation, and **h**<sup>_i_</sup> _j_ refers to the feature vector before quantization at the _j_ -th level. The overall training objective is given by _L_ total = _λ_ 1 _L_ recon + _λ_ 2 _L_ pred + _λ_ 3 _L_ VQ. 

Table 1. Performance comparison between HeartcareGPT and other baselines on _Closed-QA_ tasks from Heartcare-Bench<sup>S</sup> and HeartcareBench<sup>I</sup> , evaluated by accuracy. We use **bold** text to indicate the best results and underline to indicate the second-best results. 

|**Model**||**Heartcare-B**|**ench**<sup>**S**</sup>|||**Heartcare-B**|**ench**<sup>**I**</sup>||**Avg.**|
|---|---|---|---|---|---|---|---|---|---|
||**Diagnosis**|**Waveform**|**Rhythm**|**Misc.**|**Diagnosis**|**Waveform**|**Rhythm**|**Misc.**||
||||**_Generalist_**|**_Models_**||||||
|LLaVA-1.5-7B [27]|19.26|27.03|22.19|28.96|15.15|12.40|18.20|25.47|21.08|
|Qwen2.5-VL-7B [6]|28.29|35.36|31.70|35.31|35.84|31.06|22.35|41.41|32.67|
|InternVL-2.5-8B [11]|29.87|34.18|31.25|40.75|25.86|17.89|38.23|40.05|32.26|
|Yi-VL-6B [51]|49.19|38.90|31.06|39.65|59.55|44.46|17.06|39.85|39.97|
|MiniCPM-V2.6-8B [14]|32.37|31.34|23.40|32.23|23.71|33.59|10.05|30.63|27.17|
|Gemma-3-4B [43]|23.84|24.36|18.30|21.47|38.73|38.09|20.63|19.32|25.59|
|Claude-3.5 [2]|37.37|20.73|15.96|35.16|30.58|37.76|37.83|34.45|31.23|
|GPT5 [34]|29.87|15.23|19.57|25.23|41.42|34.14|33.73|26.23|28.30|
||||**_Medical_**|**_Models_**||||||
|Lingshu-7B [48]|29.56|33.44|40.00|20.75|47.96|37.98|19.31|30.28|32.41|
|LLaVA-Med-7B [20]|27.79|25.86|24.04|33.69|25.11|25.96|26.17|33.36|27.75|
|MedVLM-R1-2B [35]|16.27|15.42|15.32|36.32|14.08|19.43|8.33|35.56|20.09|
|HealthGPT-M3-3.8B [25]|21.63|10.87|26.58|20.23|15.35|12.41|16.06|23.97|18.39|
|**HeartcareGPT-3.8B**|81.95|95.94|82.79|**79.84**|87.85|**92.21**|79.25|67.80|83.33|
|**HeartcareGPT-7B**|**86.16**|**98.62**|**93.20**|75.15|**62.71**|87.82|**85.13**|**78.59**|**83.42**|



**Tokenization.** During inference, we discard the decoder and prediction heads, retaining only the encoder and quantizer. Given an ECG segment **S** of length _T_ , Beat applies temporal normalization: if _T > t_ , split into _⌈T/t⌉_ slices from right to left; if _T < t_ , left-pad to _t_ . For recordings with fewer than 12 leads, pad missing channels with zeros. Each segment is then encoded into discrete tokens: 



These discrete tokens can be directly used as multimodal extensions of LLM vocabulary, enabling unified semantic modeling and cross-modal reasoning between ECG signals and texts. 

For autoregressive forecasting, Beat predicts the next _t_<sup>_′_</sup> segment conditioned on the previous context ( _T_ + _t_<sup>_′_</sup> ), iteratively generating arbitrary-length ECG sequences. 

### **5.2. HeartcareGPT** 

As Figure 3 (b) shows, we present **HeartcareGPT** , which maps ECG signals, ECG images, and text into a unified discrete space, enabling ECG reasoning with a single autoregressive architecture. 

**Unified ECG Tokens.** Specifically, for a multi-lead ECG _S ∈_ R<sup>_T ×C_</sup> , Beat (Eq. 8) encodes it into a token sequence _S_ capturing elementary ECG morphology. For ECG images, we partition and rearrange them into 12 lead-wise maps to decouple lead semantics, then apply SigLip [55] to obtain visual tokens _V_ that embed per-lead waveform and spatial layout. Meanwhile, clinical notes, basic physiological data, and diagnostic instructions are tokenized by the native tokenizer of the LLM into textual features _FT_ , providing a semantic anchor for non-text modalities and enabling finegrained alignment and joint modeling in the shared token space. 

**Dual Stream Projection Alignment.** To avoid parameter interference among modalities in shallow layers, we adopt a decoupled design, mapping sequential feature _FS_ and visual feature _FV_ into the model’s embedding space through dedicated expert projections Ψsig and Ψimg: 



so that they lie in the same representation space. Either projection layer is implemented as a lightweight MLP block. Subsequently, at the input side we directly concatenate _FS_ , _FV_ and _FT_ into a single long sequence: 



where _⟨_ sig _⟩_ and _⟨_ img _⟩_ are specially introduced conditional tokens that can be expanded as needed at the implementation level to accommodate multiple signal segments and multiple images. With this design, the routing and composition of multimodal inputs no longer rely on additional architectural components, but are uniformly reduced to simple template filling of instruction prompts. 

**Training Strategy.** To enable stable coexistence of tokens from heterogeneous modalities and distributions within the base model, HeartcareGPT adopts an overall training pipeline with clearly separated roles: **(i) Multimodal warmup** , where we only train the signal and image projection layers using a captioning task to preliminarily align multimodal features with textual features; **(ii) Joint fine-tuning** , where, once all modalities have been embedded into a compatible space, we unfreeze the weights of the LLM and optimize it together with the modality projection layers in an end-to-end manner on instruction tasks. 

**Autoregressive Multimodal Generation.** The unified training objective of the model is to generate the corresponding textual output _R_ = [ _r_ 1 _, r_ 2 _, . . . , rNr_ ] conditioned on the 


Table 2. Performance comparison between HeartcareGPT and baselines on _Open-QA_ tasks from Heartcare-Bench<sup>S</sup> and Heartcare-Bench<sup>I</sup> . 

|||||**Heartcar**|**e-Bench**<sup>**S**</sup>|||||||**Heartcar**|**e-Bench**<sup>**I**</sup>||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|**Model**|**Dia**|**gnosis**|**Wav**|**eform**|**Rh**|**ythm**|**M**|**isc.**|**Dia**|**gnosis**|**Wav**|**eform**|**Rh**|**ythm**|**M**|**isc.**|
||**F1-Bio**|**Rouge-L**|**F1-Bio**|**Rouge-L**|**F1-Bio**|**Rouge-L**|**F1-Bio**|**Rouge-L**|**F1-Bio**|**Rouge-L**|**F1-Bio**|**Rouge-L**|**F1-Bio**|**Rouge-L**|**F1-Bio**|**Rouge-L**|
||||||||**_Generalist_**|**_Models_**|||||||||
|LLaVA-1.5-7B [27]|18.52|11.57|25.34|13.93|20.84|12.45|27.53|14.82|14.82|10.25|11.24|9.67|17.53|11.38|24.21|13.76|
|Qwen2.5-VL-7B [6]|27.63|12.84|34.15|15.72|30.82|13.96|34.27|16.38|35.24|14.93|30.35|13.27|21.84|12.18|40.13|17.24|
|InternVL-2.5-8B [11]|29.27|13.29|33.52|14.86|30.64|13.72|39.82|16.95|25.37|11.82|17.26|10.45|37.53|15.63|39.27|16.84|
|Yi-VL-6B [51]|48.53|15.38|38.27|14.72|30.42|13.85|38.96|15.63|58.92|17.82|43.85|16.27|16.43|10.84|38.74|15.49|
|MiniCPM-V2.6-8B [14]|31.84|12.93|30.73|12.84|22.86|11.75|31.52|13.92|23.17|11.28|32.94|13.46|9.53|8.67|29.83|13.27|
|Gemma-3-4B [43]|12.90|5.49|11.57|6.95|13.85|5.87|23.01|6.08|14.13|6.65|14.74|9.90|15.12|5.95|21.02|8.98|
|Claude-3.5 [2]|21.43|5.04|14.81|4.36|23.01|6.75|17.23|7.91|21.43|5.39|27.69|5.02|17.78|4.13|19.14|10.24|
|GPT5 [34]|23.00|7.25|21.42|13.93|37.39|14.71|24.83|9.57|40.82|15.93|33.52|12.84|33.17|12.53|25.64|9.82|
||||||||**_Medical M_**|**_odels_**|||||||||
|Lingshu-7B [48]|28.94|12.37|32.84|14.26|39.27|15.84|20.17|10.53|47.36|16.28|37.35|15.12|18.74|10.87|29.63|13.15|
|LLaVA-Med-7B [20]|27.15|12.84|25.27|11.93|23.48|11.27|33.02|14.38|24.53|11.29|25.37|12.15|25.53|11.84|32.74|14.26|
|MedVLM-R1-2B [35]|15.64|8.73|14.83|8.25|14.73|8.16|35.64|13.52|13.52|7.84|18.84|9.27|16.83|6.45|34.92|13.17|
|HealthGPT-M3-3.8B [25]|21.07|10.82|10.27|8.93|25.94|11.76|19.63|10.24|14.83|9.67|11.84|8.75|15.43|9.82|23.37|11.39|
|**HeartcareGPT-3.8B**|68.53|32.27|72.74|35.38|**78.63**|**36.42**|65.84|30.97|63.17|29.83|70.92|**33.57**|**75.36**|**34.69**|61.53|28.24|
|**HeartcareGPT-7B**|**72.94**|**36.85**|**78.26**|**39.64**|73.62|35.27|**68.37**|**34.18**|**67.43**|**33.52**|**76.73**|**37.84**|69.84|32.47|**64.17**|**31.73**|



above multimodal input _U_ = _{FS , FV , FT }_ , by maximizing the following probability distribution: 



Here, _θ_ denotes the parameters of _M_ llm. The above optimization equips the model with strong capabilities in ECG diagnosis and cross-modal knowledge complementation, yielding a general paradigm for ECG specific Med-MLLMs. 

## **6. Experiments** 

### **6.1. Data and Experimental Setup** 

**Data Details.** We systematically evaluate our model with baseline models on the proposed Heartcare-Bench<sup>S</sup> , Heartcare-Bench<sup>I</sup> and Heartcare-Bench<sup>C</sup> , providing a comprehensive assessment of generalization and diagnostic accuracy. For baseline models that cannot accept digital signal input directly, we convert signals into images. More details are provided in Appendix B.3. 

**Model Details.** HeartcareGPT employs SigLip [55] and the proposed Beat as dual-form feature encoders and adopts a three-stage training paradigm: (i) training Beat to extract high-fidelity ECG embeddings, (ii) warming up the visual and signal projectors to stabilize feature alignment, and (iii) performing joint instruction fine-tuning on Heartcare-400K for end-to-end modeling. Detailed hyperparameter settings and model configurations are provided in Appendix B.2. **Baseline.** We conduct a zero-shot evaluation on 12 representative LLMs, including eight open-world LLMs (e.g., LLaVA-v1.5 [20], Qwen2.5-VL [6], InternVL2.5 [11], Yi-VL [51], MiniCPM-V2.6 [14], gemma3 [43], Claude-3.5 [2], GPT5 [34] and four Med-MLLMs (e.g., Lingshu-7B [48], LLaVA-Med [20], MedVLMR1 [35], HealthGPT [25]). _Signal prediction_ tasks are not included in the evaluation when baseline models fail to respond to prediction instructions correctly. More details refer 

to Appendix B.1. 

### **6.2. Main Results** 

**Closed-QA.** As shown in Table 1, HeartcareGPT-7B achieves SOTA performance on _Closed-QA_ with an average accuracy of 83.42%, while HeartcareGPT-3.8B achieves 83.33%, surpassing other models across all subtasks by a large margin. We attribute this to the ECG-aware tokenization and instruction tuning framework, which enables precise alignment between temporal signal patterns and clinically grounded language reasoning. 

**Open-QA.** Table 2 reports the results on _Open-QA_ tasks, evaluated using _BioBERTScore-F1 (F1-Bio_ ) [38] and _ROUGE-L_ [24]. Our HeartcareGPT series achieve the highest overall performance across four subtasks, demonstrating strong capability in generating clinically relevant, semantically consistent answers grounded in ECG signals. 

**Comparison-QA.** The _Comparison-QA_ task assesses crossmodal reasoning by requiring analysis and contrast of two ECG inputs—either signals, images, or signal–image pairs— in Heartcare-Bench<sup>C</sup> . As shown in Table 3, HeartcareGPT excels in this challenging setting, with both 3.8B and 7B models achieving SOTA performance. This success underscores the effectiveness of our multimodal fusion design, which enables a unified and nuanced understanding across heterogeneous ECG representations. 

**Report Generation.** Table 4 presents HeartcareGPT’s _Report Generation_ results on Heartcare-Bench<sup>S</sup> and HeartcareBench<sup>I</sup> , evaluated by _Score_<sup>_GPT_</sup> , _F1-Rad_ [17], and _ROUGEL_ [24]. HeartcareGPT achieves top scores on most metrics, outperforming all baseline models. While Qwen2.5-VL-7B scores higher on some metrics related to report structure and expression, HeartcareGPT excels in clinical content accuracy and reliability. Since GPT-based evaluation considers formatting and completeness, score differences often reflect output style rather than clinical accuracy. Full criteria and further examine ECG image segmentation by contrasting full-image and 12-lead sub-image strategies. As summarized in Figure 5 (a), multimodal fusion and sub-image segmentation yield clear performance gains, highlighting the value of integrating diverse ECG modalities for improved diagnostic understanding. 


Table 3. Performance comparison between HeartcareGPT and other baselines on _Comparison-QA_ tasks from Heartcare-Bench<sup>C</sup> . _Cons._ = Consecutive ECGs from the same patient; _Irr._ = Irrelevant ECGs from different patients. Yi-VL-6B and HealthGPT-M3-3.8B do not support multi-image input. 

|**Model**|**S–**|**S**|**I–**|**I**|**S–**|**I**|**Avg.**|
|---|---|---|---|---|---|---|---|
||**Cons.**|**Irr.**|**Cons.**|**Irr.**|**Cons.**|**Irr.**||
||**_G_**<br>|**_eneralist_**<br>|**_Models_**<br>|||||
|LLaVA-1.5-7B [27]|62.70|56.24|66.17|63.51|47.94|45.65|57.04|
|Qwen2.5-VL-7B [6]|50.64|47.26|49.36|50.68|50.21|51.28|49.91|
|InternVL-2.5-8B [11]|47.83|45.13|46.18|43.33|49.40|43.24|45.85|
|MiniCPM-V2.6-8B [14]|53.22|48.13|51.87|45.50|49.85|45.05|48.97|
|Gemma-3-4B [43]|49.18|38.53|49.63|35.44|49.55|35.59|42.99|
|Claude-3.5 [2]|46.17|43.96|39.15|56.78|24.47|54.82|44.23|
|GPT5 [34]|50.97|49.18|50.82|49.40|50.00|47.90|49.71|
|||**_Medical_**|**_Models_**|||||
|Lingshu-7B [48]|58.79|45.01|62.13|59.46|45.11|43.12|52.27|
|LLaVA-Med-7B [20]|35.46|34.82|32.84|32.65|43.26|34.14|35.53|
|MedVLM-R1-2B [35]|58.44|39.33|53.69|40.34|49.57|37.95|46.55|
|**HeartcareGPT-3.8B**|66.40|67.47|69.88|75.19|78.71|78.83|72.74|
|**HeartcareGPT-7B**|**74.01**|**75.87**|**74.80**|**79.09**|**80.05**|**79.55**|**77.23**|


![Results of ablation studies on training pipeline.](fig_4.png)

Figure 4. Results of ablation studies on training pipeline. 


|del comparison<br>e 4. Performa|s are d<br>nce co|etaile<br>mpariso|d in Ta<br>n betw|ble7.<br>een H|eartcar|eGPT and|Acc<br>CldA<br>F1-Bio<br>(Open-QA)<br>Rouge-L<br>(Open-QA)<br><br>**4%**<br>**5%**<br>**21%**<br>**12%**<br>**7%**<br>**7%**<br>**（b）**<br>**（a）**|
|---|---|---|---|---|---|---|---|
|r baseline metho|ds on_R_|_eport G_|_enerat_|_ion_task|s from|Heartcare-|(ose-Q)<br>**40%**<br>**9%**<br>**9%**|
|ch<sup>S </sup>and Heartca|re-Ben|ch<sup>I</sup>.|||||Score<br>(Report)<br>Acc<br>(Comparison-QA)<br>**9%**<br><br>**10%**<br>**7%**<br>**14%**<br>|
||||||||**7%**<br>**15%**|
|**Model**|**Hea**|**rtcare-Ben**|**ch**<sup>**S**</sup>|**Hea**|**rtcare-Be**|**nch**<sup>**I**</sup>|**10%**<br>|
||**Score**<sup>**GPT**</sup><br>|**F1-Rad**<br>**_Generalist_**|**Rouge-L**<br>**_Models_**|**Score**<sup>**GPT**</sup>|**F1-Rad**|**Rouge-L**|Rouge-L<br>(Report)<br>F1-Rad<br>(Report)<br>**7%**<br>**7%**|
|LLaVA-1.5-7B [27]|55.69|5.22|18.30|37.40|12.90|20.75|HeartcareGPT-3.8B<br>InternVL-2.5<br>MiniCPM-V2.6|
|Qwen2.5-VL-7B [6]|64.60|5.64|9.00|67.40|7.67|13.56|Qwen2.5-VL-7B<br>Lingshu-7B<br><br><br>Yi-VL|
|InternVL-2.5-8B [11]|42.07|5.50|9.13|33.69|7.73|11.02|LLaVA-Med<br>MedVLM-R1<br>HealthGPT-M3|
|Yi-VL-6B [51]|26.58|3.75|13.05|21.74|6.73|15.18||
|MiniCPM-V2.6-8B [14]|34.54|5.74|10.66|49.60|7.48|12.63|Fi 5Rl f bli di  lidl i|
|Gemma-3-4B [43]|64.57|4.38|7.57|57.10|7.00|9.07|gure . (a) esuts o aaton stues on mutmoa ntegra|
|Claude-3.5 [2]|63.11|5.03|12.33|63.29|7.63|14.02|(b) Expert preference distribution across models. Inner-ring re|
|GPT5 [34]|62.73|4.85<br>**_Medical M_**|8.86<br>**_odels_**|78.80|6.29|9.55|are based on _Open-QA_ evaluations; outer-ring results are base<br>|
|Lingshu-7B [48]|58.89|7.13|13.94|51.94|9.52|16.31|_Report Generation_ evaluations.|
|LLaVA-Med-7B [20]<br>MedVLM-R1-2B [35]|50.02<br>32.26|5.21<br>2.10|14.95<br>15.51|27.71<br>56.58|6.56<br>9.05|16.45<br>18.27|**Expert Evaluation.** We conduct expert evaluation to|
|HealthGPT-M3-3.8B [25]|25.39|1.00|7.53|37.62|1.22|9.13|sess clinical reference on _Oen-QA_ and _Reort Genera_|
|**HeartcareGPT-3.8B**<br>**HeartcareGPT-7B**|61.29<br>**76.55**|**26.84**<br>21.70|**34.39**<br>32.55|**78.50**<br>65.03|23.10<br>**27.17**|38.68<br>**44.86**|p_p_ _p_<br>tasksTen board-certified cardiologists reviewed 400 s|



model comparisons are detailed in Table 7. 

Table 4. Performance comparison between HeartcareGPT and other baseline methods on _Report Generation_ tasks from HeartcareBench<sup>S</sup> and Heartcare-Bench<sup>I</sup> . 

![(a) Results of ablation studies on multimodal integration. (b) Expert preference distribution across models. Inner-ring results are based on Open-QA evaluations; outer-ring results are based on Report Generation evaluations.](fig_5.png)

Figure 5. (a) Results of ablation studies on multimodal integration. (b) Expert preference distribution across models. Inner-ring results are based on _Open-QA_ evaluations; outer-ring results are based on _Report Generation_ evaluations. 

**Expert Evaluation.** We conduct expert evaluation to assess clinical preference on _Open-QA_ and _Report Generation_ tasks. Ten board-certified cardiologists reviewed 400 sampled cases with shuffled outputs from HeartcareGPT-3.8B and eight representative baselines, selecting responses best aligned with clinical reasoning and diagnostic conventions. As shown in Figure 5 (b), HeartcareGPT achieves the highest first-choice selections—40% in _Open-QA_ and 21% in _Report Generation_ —substantially surpassing all baselines (7–15%). These results demonstrate clear clinical preference for HeartcareGPT, with expert scoring consistent with automated metrics, further supporting its clinical reliability. 

### **6.3. Ablation Study and Expert Evaluation** 

We conduct extensive ablation studies to analyze the contribution of each design component in HeartcareGPT. The complete results are presented in Appendix C.1. **The Three-Stage Training Pipeline.** We further verify the necessity of each training stage in our three-phase optimization scheme. We experiment with models that omit (i) Step 1: beat-level training, (ii) Step 2: projector warming-up, and (iii) both steps simultaneously. Figure 4 show our results of ablation studies on the training pipeline. The performance significantly drops in all ablated variants, indicating that each stage plays an indispensable role in stabilizing multimodal alignment and improving diagnostic accuracy. **Multimodal Integration.** To assess multimodal fusion, we compare our tri-modal (signal–image–text) model against single-modality baselines (image-only and signal-only). We 

## **7. Conclusion** 

Heartcare Suite establishes a comprehensive multimodal foundation framework for fine-grained ECG understanding, integrating high-quality dataset, clinically aligned benchmarks, and scalable modeling strategies. We hope this work serves as a stepping stone for future research on MedMLLMs in clinically grounded signal–language reasoning. 


## **Acknowledgements** 

This work has been supported in part by the NSFC (No. 62436007), the ZJNSF (No. LZ25F020004), the Key Research and Development Projects in Zhejiang Province (No. 2025C01128, 2025C01030, 2025C02156), Ningbo Yongjiang Talent Introduction Programme (2023A400-G). 

## **References** 

- [1] Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. _arXiv preprint arXiv:2404.14219_ , 2024. 12 

- [2] Anthropic. Claude 3.5. https://www.anthropic. com, 2024. Large Language Model by Anthropic. 1, 6, 7, 8 

- [3] Raymond Ao and George He. Image based deep learning in 12-lead ecg diagnosis. _Frontiers in Artificial Intelligence_ , 5: 1087370, 2023. 2 

- [4] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. _arXiv preprint arXiv:2308.01390_ , 2023. 

- [5] Ahtisham Ayyub, Christos Politis, and Muhammad Arslan Usman. A comprehensive review of ai-based detection of arrhythmia using electrocardiogram (ecg). _Computers in Biology and Medicine_ , 196:110594, 2025. 1 

- [6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. _arXiv preprint arXiv:2502.13923_ , 2025. 1, 6, 7, 8, 12 

- [7] Mingsheng Cai, Jiuming Jiang, Wenhao Huang, Che Liu, and Rossella Arcucci. Supreme: A supervised pre-training framework for multimodal ecg representation learning. _arXiv preprint arXiv:2502.19668_ , 3, 2025. 3 

- [8] Juan Manuel Zambrano Chaves, Shih-Cheng Huang, Yanbo Xu, Hanwen Xu, Naoto Usuyama, Sheng Zhang, Fei Wang, Yujia Xie, Mahmoud Khademi, Ziyi Yang, et al. Towards a clinically accessible radiology foundation model: openaccess and lightweight, with automated evaluation. _arXiv preprint arXiv:2403.08002_ , 2024. 3 

- [9] Chen Chen, Lei Li, Marcel Beetz, Abhirup Banerjee, Ramneek Gupta, and Vicente Grau. Large language modelinformed ecg dual attention network for heart failure risk prediction. _IEEE transactions on big data_ , 11(3):948–960, 2025. 3 

- [10] Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Zhenyang Cai, Ke Ji, Xiang Wan, et al. Towards injecting medical visual knowledge into multimodal llms at scale. In _Proceedings of the 2024 conference on empirical methods in natural language processing_ , pages 7346–7370, 2024. 1, 3 

- [11] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. _arXiv preprint arXiv:2412.05271_ , 2024. 1, 6, 7, 8 

- [12] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 12873–12883, 2021. 2 

- [13] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. _Iclr_ , 1 (2):3, 2022. 12 

- [14] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. _arXiv preprint arXiv:2404.06395_ , 2024. 6, 7, 8 

- [15] John D Hunter. Matplotlib: A 2d graphics environment. _Computing in science & engineering_ , 9(3):90–95, 2007. 12 

- [16] Artifex Software Inc. Pymupdf: Python bindings for mupdf – a lightweight pdf and xps viewer. https://github. com/pymupdf/PyMuPDF, 2024. Accessed: 2025-05-16. 


- [17] Saahil Jain, Ashwin Agrawal, Adriel Saporta, Steven QH Truong, Du Nguyen Duong, Tan Bui, Pierre Chambon, Yuhao Zhang, Matthew P Lungren, Andrew Y Ng, et al. Radgraph: Extracting clinical entities and relations from radiology reports. _arXiv preprint arXiv:2106.14463_ , 2021. 7, 14 

- [18] Songtao Jiang, Tuo Zheng, Yan Zhang, Yeying Jin, Li Yuan, and Zuozhu Liu. Med-moe: Mixture of domain-specific experts for lightweight medical vision-language models. In _Findings of the association for computational linguistics: EMNLP 2024_ , pages 3843–3860, 2024. 3 

- [19] Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, and Shenda Hong. Reading your heart: Learning ecg words and sentences via pre-training ecg language model. _arXiv preprint arXiv:2502.10707_ , 2025. 3 

- [20] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llava-med: Training a large languageand-vision assistant for biomedicine in one day. _Advances in Neural Information Processing Systems_ , 36:28541–28564, 2023. 1, 3, 6, 7, 8 

- [21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In _International conference on machine learning_ , pages 19730–19742. PMLR, 2023. 1 

- [22] Sijing Li, Tianwei Lin, Lingshuai Lin, Wenqiao Zhang, Jiang Liu, Xiaoda Yang, Juncheng Li, Yucheng He, Xiaohui Song, Jun Xiao, et al. Eyecaregpt: Boosting comprehensive ophthalmology understanding with tailored dataset, benchmark and model. In _Proceedings of the 33rd ACM International Conference on Multimedia_ , pages 3893–3902, 2025. 3 

- [23] Xihong Lian, Limin Jiao, Zejin Liu, Qiqi Jia, Jing Zhong, Miao Fang, and Weilin Wang. Multi-spatiotemporal heterogeneous legacy effects of climate on terrestrial vegetation dynamics in china. _GIScience & Remote Sensing_ , 59(1): 164–183, 2022. 2 

- [24] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In _Text summarization branches out_ , pages 74–81, 2004. 7, 14 

- [25] Tianwei Lin, Wenqiao Zhang, Sijing Li, Yuqian Yuan, Binhe Yu, Haoyuan Li, Wanggui He, Hao Jiang, Mengze Li, Song Xiaohui, et al. Healthgpt: A medical large vision-language model for unifying comprehension and generation via heterogeneous knowledge adaptation. In _International Conference on Machine Learning_ , pages 37975–37995. PMLR, 2025. 1, 3, 6, 7, 8 

- [26] Che Liu, Zhongwei Wan, Cheng Ouyang, Anand Shah, Wenjia Bai, and Rossella Arcucci. Zero-shot ecg classification with multimodal learning and test-time clinical knowledge enhancement. _arXiv preprint arXiv:2403.06659_ , 2024. 3 

- [27] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. _Advances in neural information processing systems_ , 36:34892–34916, 2023. 1, 6, 7, 8 

- [28] Dominique Makowski and contributors. Neurokit2: Python toolbox for neurophysiological signal processing. https:// github.com/neuropsychology/NeuroKit, 2024. Accessed: 2025-05-16. 12 

- [29] Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. Med-flamingo: a multimodal medical few-shot learner. In _Machine learning for health (ML4H)_ , pages 353–367. PMLR, 2023. 1, 3 

- [30] Yoojin Nam, Dong Yeong Kim, Sunggu Kyung, Jinyoung Seo, Jeong Min Song, Jimin Kwon, Jihyun Kim, Wooyoung Jo, Hyungbin Park, Jimin Sung, et al. Multimodal large language models in medical imaging: current state and future directions. _Korean Journal of Radiology_ , 26(10):900, 2025. 1 

- [31] Cuong V Nguyen, Hieu X Nguyen, Dung D Pham Minh, and Cuong D Do. Comparing deep neural network for multi-label ecg diagnosis from scanned ecg. _arXiv preprint arXiv:2502.14909_ , 2025. 2 

- [32] Jungwoo Oh, Gyubok Lee, Seongsu Bae, Joon-myoung Kwon, and Edward Choi. Ecg-qa: A comprehensive question answering dataset combined with electrocardiogram. _Advances in Neural Information Processing Systems_ , 36:66277– 66288, 2023. 3, 4 

- [33] OpenAI. Gpt-4 system card. https://cdn.openai. com/papers/gpt-4-system-card.pdf, 2023. 1, 4, 12, 14 

- [34] OpenAI. Gpt-5 system card. https://cdn.openai. com/gpt-5-system-card.pdf, 2023. 6, 7, 8 

- [35] Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. Medvlm-r1: Incentivizing medical reasoning capability of vision-language models (vlms) via reinforcement learning. In _International Conference on Medical Image Computing and Computer-Assisted Intervention_ , pages 337–347. Springer, 2025. 3, 6, 7, 8 

- [36] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. _arXiv preprint arXiv:2306.14824_ , 2023. 1 

- [37] Eedara Prabhakararao and Samarendra Dandapt. Multi-label ecg classification using temporal convolutional neural network. _arXiv preprint arXiv:2306.03844_ , 2023. 2 

- [38] Lance Ramshaw and Mitch Marcus. Text chunking using transformation-based learning. In _Third workshop on very large corpora_ , 1995. 7, 14 

- [39] Paul Rubel, Danilo Pani, Alois Schloegl, Jocelyne Fayn, Fabio Badilini, Peter W Macfarlane, and Alpo Varri. Scpecg v3. 0: An enhanced standard communication protocol for computer-assisted electrocardiography. In _2016 Computing in Cardiology Conference (CinC)_ , pages 309–312. IEEE, 2016. 12 

- [40] Ikaros Silva. Wfdb app toolbox for matlab/octave. https://github.com/ikarosilva/wfdb-apptoolbox, 2023. Accessed: 2025-05-16. 12 

- [41] Nils Strodthoff, Patrick Wagner, Tobias Schaeffter, and Wojciech Samek. Deep learning for ecg analysis: Benchmarks and insights from ptb-xl. _IEEE journal of biomedical and health informatics_ , 25(5):1519–1528, 2020. 2 

- [42] Kai Sun, Siyan Xue, Fuchun Sun, Haoran Sun, Yu Luo, Ling Wang, Siyuan Wang, Na Guo, Lei Liu, Tian Zhao, et al. Medical multimodal foundation models in clinical diagnosis and treatment: Applications, challenges, and future directions. _Artificial Intelligence in Medicine_ , page 103265, 2025. 1 

- [43] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. _arXiv preprint arXiv:2503.19786_ , 2025. 1, 6, 7, 8 

- [44] Eric J Topol. High-performance medicine: the convergence of human and artificial intelligence. _Nature medicine_ , 25(1): 44–56, 2019. 1 

- [45] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. _Advances in neural information processing systems_ , 30, 2017. 2 

- [46] Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot, Dieter Kreiseler, Fatima I Lunze, Wojciech Samek, and Tobias Schaeffter. Ptb-xl, a large publicly available electrocardiography dataset. _Scientific data_ , 7(1):154, 2020. 1, 2, 3, 13 

- [47] Liping Xie, Zilong Li, Yihan Zhou, Yiliu He, and Jiaxin Zhu. Computational diagnostic techniques for electrocardiogram signal analysis. _Sensors_ , 20(21):6318, 2020. 1 

- [48] Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, et al. Lingshu: A generalist foundation model for unified multimodal medical understanding and reasoning. _arXiv preprint arXiv:2506.07044_ , 2025. 1, 3, 6, 7, 8 

- [49] Kai Yang, Massimo Hong, Jiahuan Zhang, Yizhen Luo, Suyuan Zhao, Ou Zhang, Xiaomao Yu, Jiawen Zhou, Liuqing Yang, Ping Zhang, et al. Ecg-lm: understanding electrocardiogram with a large language model. _Health Data Science_ , 5:0221, 2025. 3 

- [50] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. _arXiv preprint arXiv:2408.04840_ , 2024. 1 

- [51] Alex Young, Bei Chen, Chao Li, Chengen Huang, Ge Zhang, Guanwei Zhang, Guoyin Wang, Heng Li, Jiangcheng Zhu, Jianqun Chen, et al. Yi: Open foundation models by 01. ai. _arXiv preprint arXiv:2403.04652_ , 2024. 1, 6, 7, 8 

- [52] Han Yu, Huiyuan Yang, and Akane Sano. Ecg-sl: electrocardiogram (ecg) segment learning, a deep learning method for ecg signal. _arXiv preprint arXiv:2310.00818_ , 2023. 3 

- [53] Han Yu, Peikun Guo, and Akane Sano. Ecg semantic integrator (esi): A foundation ecg model pretrained with llm-enhanced cardiological text. _arXiv preprint arXiv:2405.19366_ , 2024. 3 

- [54] Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-toend neural audio codec. _IEEE/ACM Transactions on Audio, Speech, and Language Processing_ , 30:495–507, 2021. 2 

- [55] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In _Proceedings of the IEEE/CVF international conference on computer vision_ , pages 11975–11986, 2023. 6, 7, 12 

- [56] Yutong Zhang, Yi Pan, Tianyang Zhong, Peixin Dong, Kangni Xie, Yuxiao Liu, Hanqi Jiang, Zihao Wu, Zhengliang Liu, Wei Zhao, et al. Potential of multimodal large language models for data mining of medical images and free-text reports. _MetaRadiology_ , 2(4):100103, 2024. 1 

- [57] Juexiao Zhou, Xiaonan He, Liyuan Sun, Jiannan Xu, Xiuying Chen, Yuetan Chu, Longxi Zhou, Xingyu Liao, Bin Zhang, and Xin Gao. Skingpt-4: an interactive dermatology diagnostic system with visual large language model. _arXiv preprint arXiv:2304.10691_ , 2023. 3 

