---
license: mit
task_categories:
- question-answering
language:
- en
tags:
- Mental Health
size_categories:
- 10K<n<100K
---

# 🗣️ Synthetic Counseling Conversations Dataset

## 📝 Description

### Synthetic Data 10K

This dataset consists of 9,775 synthetic conversations between a counselor and a client, covering 33 mental health topics such as 💑 Relationships, 😟 Anxiety, 😔 Depression, 🤗 Intimacy, and 👨‍👩‍👧‍👦 Family Conflict. The conversations were generated using the OpenAI GPT-3.5 Turbo model and a customized adaptation of the Airoboros self-generation framework.

The Airoboros framework was used to create a new prompt that provided clear instructions for generating patient queries. These queries were then fed back into the GPT-3.5 Turbo model to generate corresponding responses. The proportion of each topic was specified in the prompt to ensure the synthetic conversations authentically mimic the complexity and diversity of human therapist-client interactions.

This dataset aims to equip language models with exposure to a wide spectrum of psychological conditions and therapeutic strategies, enabling them to engage in more realistic and effective counseling conversations. 🧠

### Interview Data 6K

This dataset consists of 6338 question-answer pairs from 378 interview transcripts. The transcripts are collected from an ongoing clinical trial transcribed by human experts based on audio recordings of behavioral intervention sessions between behavior health coaches and caregivers of individuals in palliative or hospice care

We employed the local Mistral-7B-Instruct-v0.2 model, which is a state-of-the-art lightweight LLM to paraphrase and summarize interview transcripts documents. We fed each page of transcripts into the model and provided instructions (see Table 1) to summarize the page into a single round of conversation between the caregiver and the behavioral health coach. Subsequently, we filtered out any conversations with less than 40 words in the question and answer.

## 📊 Dataset Characteristics

- **Number of Conversations**: 9,775 🗣️
- **Topics Covered**: 💑 Relationships, 😟 Anxiety, 😔 Depression, 🤗 Intimacy, 👨‍👩‍👧‍👦 Family Conflict, and 28 other mental health topics
- **Language**: English 🇺🇸
- **Generation Method**: OpenAI GPT-3.5 Turbo model with a customized Airoboros self-generation framework

## 🤖 Dataset Usage

This dataset can be used to train and evaluate language models for counseling and mental health applications, such as chatbots, virtual assistants, and dialogue systems. It provides a diverse and realistic set of conversational scenarios that can help improve the models' understanding of psychological conditions and therapeutic strategies.

## 🌍 Dataset Limitations

The dataset is entirely synthetic and may not fully capture the nuances and complexities of real-world counseling conversations. Additionally, the dataset is limited to English language conversations and may not be representative of diverse cultural and linguistic contexts.

## 📚 Citation
If you use MentalChat16K in your research, please cite the dataset as follows:
```
@dataset{MentalChat16K,
  author    = {Jia Xu, Tianyi Wei, Bojian Hou, Patryk Orzechowski, Shu Yang, Ruochen Jin, Rachael Paulbeck, Joost Wagenaar, George Demiris, Li Shen},
  title     = {MentalChat16K: A Benchmark Dataset for Conversational Mental Health Assistance},
  year      = {2024},
  url       = {https://huggingface.co/datasets/ShenLab/MentalChat16K},
}
```

```