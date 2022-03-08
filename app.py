# -*- coding: utf-8 -*-
"""
Created on Friday 17th Dec- 2021
@author: Indrajit Singh
"""

import pandas as pd
import numpy as np 
import torch 
import streamlit as st


from transformers import pipeline,QuestionAnsweringPipeline, DistilBertForQuestionAnswering,AutoTokenizer

model_checkpoint = "distilbert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# The model_path here would be the directory in which you saved the model using the HuggingFace model.save_pretrained() function
model_path = "self-realization-model"
myQAModel = DistilBertForQuestionAnswering.from_pretrained(model_path)

QAPipeline = QuestionAnsweringPipeline(model = myQAModel,tokenizer = tokenizer)

# This is a markdown message at the beginning of my application in which I'm introducing myself and explaining the question. You should add whatever message you want to. 
st.markdown("Project Self-Realization")

st.markdown("In America Vivekananda's mission was the interpretation of India's spiritual culture,especially in its Vedantic setting.")

select_context = """Ramakrishna, the God-man of modern times, was born on February 18, 1836, in the
little village of Kamarpukur, in the district of Hooghly in Bengal. How different were
his upbringing and the environment of his boyhood from those of Narendranath, who
was to become, later, the bearer and interpreter of his message! Ramakrishna's parents,
belonging to the brahmin caste, were poor, pious, and devoted to the traditions of their
ancient religion. Full of fun and innocent joys, the fair child, with flowing hair and a
sweet, musical voice, grew up in a simple countryside of rice-fields, cows, and banyan
and mango trees. He was apathetic about his studies and remained practically illiterate
all his life, but his innate spiritual tendencies found expression through devotional
songs and the company of wandering monks, who fired his boyish imagination by the
stories of their spiritual adventures. At the age of six he experienced a spiritual ecstasy
while watching a flight of snow-white cranes against a black sky overcast with rainclouds. He began to go into trances as he meditated on gods and goddesses. His father's
death, which left the family in straitened circumstances, deepened his spiritual mood.
And so, though at the age of sixteen he joined his brother in Calcutta, he refused to go
on there with his studies; for, as he remarked, he was simply not interested in an.

The floodgate of Ramakrishna's emotion burst all bounds when he took up the duties of
a priest in the Kali temple of Dakshineswar, where the Deity was worshipped as the
Divine Mother. Ignorant of the scriptures and of the intricacies of ritual, Ramakrishna
poured his whole soul into prayer, which often took the form of devotional songs.
Food, sleep, and other physical needs were completely forgotten in an all-consuming
passion for the vision of God. His nights were spent in contemplation in the
neighbouring woods. Doubt sometimes alternated with hope; but an inner certainty and
the testimony of the illumined saints sustained him in his darkest hours of despair.
Formal worship or the mere sight of the image did not satisfy his inquiring mind; for
he felt that a figure of stone could not be the bestower of peace and immortality.
Behind the image there must be the real Spirit, which he was determined to behold.
This was not an easy task. For a long time the Spirit played with him a teasing game of
hide-and-seek, but at last it yielded to the demand of love on the part of the young
devotee. When he felt the direct presence of the Divine Mother, Ramakrishna dropped
unconscious to the floor, experiencing within himself a constant flow of bliss. 
When the singing was over, Sri Ramakrishna suddenly grasped Narendra's hand and
took him into the northern porch. To Narendra's utter amazement, the Master said with
tears streaming down his cheeks: 'Ah! you have come so late. How unkind of you to
keep me waiting so long!
My ears are almost seared listening to the cheap talk of worldly people. Oh, how I have
been yearning to unburden my mind to one who will understand my thought!' Then
with folded hands he said: 'Lord! I know you are the ancient sage Nara — the
Incarnation of Narayana — born on earth to remove the miseries of mankind.' The
rationalist Naren regarded these words as the meaningless jargon of an insane person.
He was further dismayed when Sri Ramakrishna presently brought from his room some
sweets and fed him with his own hands. But the Master nevertheless extracted from
him a promise to visit Dakshineswar again.
They returned to the room and Naren asked the Master, 'Sir, have you seen God?'
Without a moment's hesitation the reply was given: 'Yes, I have seen God. I see Him as
I see you here, only more clearly. God can be seen. One can talk to him. But who cares
for God? People shed torrents of tears for their wives, children, wealth, and property,
but who weeps for the vision of God? If one who cries sincerely for God, one can surely see
God."""

context = st.text_area("select a context",select_context)
options = st.multiselect(
     'Select question from Below List',
     ['Who am I?', 'Where am I going to?', "What's the purpose of Life ?", 'What do others mean to me?'])

st.markdown("OR")
question = st.text_input("Write a question of your choice here", options)

if context:
    # Execute question against paragraph
    if question:
        outputs = QAPipeline(question = question,context = context,topk = 3, max_seq_len = 512)
        answer = outputs[0]["answer"]
        output_answer = st.text_area("Answer",answer)
