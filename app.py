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



select_context = """
Swami Vivekananda's inspiring personality was well known both in India and in 
America during the last decade of the nineteenth century and the first decade of the 
twentieth. The unknown monk of India suddenly leapt into fame at the Parliament of 
Religions held in Chicago in 1893, at which he represented Hinduism. His vast 
knowledge of Eastern and Western culture as well as his deep spiritual insight, fervid 
eloquence, brilliant conversation, broad human sympathy, colourful personality, and 
handsome figure made an irresistible appeal to the many types of Americans who came 
in contact with him. People who saw or heard Vivekananda even once still cherish his 
memory after a lapse of more than half a century. 
In America Vivekananda's mission was the interpretation of India's spiritual culture, 
especially in its Vedantic setting. He also tried to enrich the religious consciousness of 
the Americans through the rational and humanistic teachings of the Vedanta 
philosophy. In America he became India's spiritual ambassador and pleaded eloquently 
for better understanding between India and the New World in order to create a healthy 
synthesis of East and West, of religion and science. 
In his own motherland Vivekananda is regarded as the patriot saint of modern India 
and an inspirer of her dormant national consciousness. To the Hindus he preached the 
ideal of a strength-giving and man-making religion. Service to man as the visible 
manifestation of the Godhead was the special form of worship he advocated for the 
Indians, devoted as they were to the rituals and myths of their ancient faith. Many 
political leaders of India have publicly acknowledged their indebtedness to Swami 
Vivekananda. 
The Swami's mission was both national and international. A lover of mankind, he 
strove to promote peace and human brotherhood on the spiritual foundation of the 
Vedantic Oneness of existence. A mystic of the highest order, Vivekananda had a 
direct and intuitive experience of Reality. He derived his ideas from that unfailing 
source of wisdom and often presented them in the soul-stirring language of poetry. 
The natural tendency of Vivekananda's mind, like that of his Master, Ramakrishna, was 
to soar above the world and forget itself in contemplation of the Absolute. But another 
part of his personality bled at the sight of human suffering in East and West alike. It 
might appear that his mind seldom found a point of rest in its oscillation between 
contemplation of God and service to man. Be that as it may, he chose, in obedience to 
a higher call, service to man as his mission on earth; and this choice has endeared him 
to people in the West, Americans in particular. 
In the course of a short life of thirty-nine years (1863-1902), of which only ten were 
devoted to public activities — and those, too, in the midst of acute physical suffering 
— he left for posterity his four classics: Jnana-Yoga, Bhakti-Yoga, Karma-Yoga, and 
Raja-Yoga, all of which are outstanding treatises on Hindu philosophy. In addition, he 
delivered innumerable lectures, wrote inspired letters in his own hand to his many 
friends and disciples, composed numerous poems, and acted as spiritual guide to the 
many seekers who came to him for instruction. He also organized the Ramakrishna 
Order of monks, which is the most outstanding religious organization of modern India. 
It is devoted to the propagation of the Hindu spiritual culture not only in the Swami's 
native land, but also in America and in other parts of the world. 
Swami Vivekananda once spoke of himself as a 'condensed India.' His life and 
teachings are of inestimable value to the West for an understanding of the mind of 
Asia. William James, the Harvard philosopher, called the Swami the 'paragon of 
Vedantists.' Max Müller and Paul Deussen, the famous Orientalists of the nineteenth 
century, held him in genuine respect and affection. 'His words,' writes Romain Rolland, 
'are great music, phrases in the style of Beethoven, stirring rhythms like the march of 
Handel choruses. I cannot touch these sayings of his, scattered as they are through the 
pages of books, at thirty years' distance, without receiving a thrill through my body like 
an electric shock. And what shocks, what transports, must have been produced when in 
burning words they issued from the lips of the hero!'
EARLY YEARS
Swami Vivekananda, the great soul loved and revered in East and West alike as the 
rejuvenator of Hinduism in India and the preacher of its eternal truths abroad, was born 
at 6:33, a few minutes before sunrise, on Monday, January 12, 1863. It was the day of 
the great Hindu festival Makarasamkranti, when special worship is offered to the 
Ganga by millions of devotees. Thus the future Vivekananda first drew breath when 
the air above the sacred river not far from the house was reverberating with the 
prayers, worship, and religious music of thousands of Hindu men and women. 
Before Vivekananda was born, his mother, like many other pious Hindu mothers, had 
observed religious vows, fasted, and prayed so that she might be blessed with a son 
who would do honour to the family. She requested a relative who was living in 
Varanasi to offer special worship to the Vireswara Siva of that holy place and seek His 
blessings; for Siva, the great god of renunciation, dominated her thought. One night 
she dreamt that this supreme Deity aroused Himself from His meditation and agreed to 
be born as her son. When she woke she was filled with joy. 
The mother, Bhuvaneswari Devi, accepted the child as a boon from Vireswara Siva 
and named him Vireswara. The family, however, gave him the name of Narendranath 
Datta, calling him, for short, Narendra, or more endearingly, Naren. 
The Datta family of Calcutta, into which Narendranath had been born, was well known 
for its affluence, philanthropy, scholarship, and independent spirit. The grand father, 
Durgacharan, after the birth of his first son, had renounced the world in search of God. 
The father, Viswanath, an attorney-at-law of the High Court of Calcutta, was versed in 
English and Persian literature and often entertained himself and his friends by reciting 
from the Bible and the poetry of Hafiz, both of which, he believed, contained truths 
unmatched by human thinking elsewhere. He was particularly attracted to the Islamic 
culture, with which he was familiar because of his close contact with the educated 
Moslems of North-western India. Moreover, he derived a large income from his law 
practice and, unlike his father, thoroughly enjoyed the worldly life. An expert in 
cookery, he prepared rare dishes and liked to share them with his friends. Travel was 
another of his hobbies. Though agnostic in religion and a mocker of social 
conventions, he possessed a large heart and often went out of his way to support idle 
relatives, some of whom were given to drunkenness. Once, when Narendra protested 
against his lack of judgement, his father said: 'How can you understand the great 
misery of human life? When you realize the depths of men's suffering, you will 
sympathize with these unfortunate creatures who try to forget their sorrows, even 
though only for a short while, in the oblivion created by intoxicants.' Naren's father, 
however, kept a sharp eye on his children and would not tolerate the slightest deviation 
from good manners. 
Bhuvaneswari Devi, the mother, was cast in a different mould. Regal in appearance 
and gracious in conduct, she belonged to the old tradition of Hindu womanhood. As 
mistress of a large household, she devoted her spare time to sewing and singing, being 
particularly fond of the great Indian epics, the Ramayana and the Mahabharata, large 
portions of which she had memorized. She became the special refuge of the poor, and 
commanded universal respect because of her calm resignation to God, her inner 
tranquillity, and her dignified detachment in the midst of her many arduous duties. 
Two sons were born to her besides Narendranath, and four daughters, two of whom 
died at an early age. 
Narendra grew up to be a sweet, sunny-tempered, but very restless boy. Two nurses 
were necessary to keep his exuberant energy under control, and he was a great tease to 
his sisters. In order to quiet him, the mother often put his head under the cold-water 
tap, repeating Siva's name, which always produced the desired effect. Naren felt a 
child's love for birds and animals, and this characteristic reappeared during the last 
days of his life. Among his boyhood pets were a family cow, a monkey, a goat, a 
peacock, and several pigeons and guinea-pigs. The coachman of the family, with his 
turban, whip, and bright-coloured livery, was his boyhood ideal of a magnificent 
person, and he often expressed the ambition to be like him when he grew up. 
Narendra bore a striking resemblance to the grand-father who had renounced the world 
to lead a monastic life, and many thought that the latter had been reborn in him. The 
youngster developed a special fancy for wandering monks, whose very sight would 
greatly excite him. One day when such a monk appeared at the door and asked for 
alms, Narendra gave him his only possession, the tiny piece of new cloth that was 
wrapped round his waist. Thereafter, whenever a monk was seen in the neighbourhood, 
Narendra would be locked in a room. But even then he would throw out of the window 
whatever he found near at hand as an offering to the holy man. In the meantime, he 
was receiving his early education from his mother, who taught him the Bengali 
alphabet and his first English words, as well as stories from the Ramayana and the 
Mahabharata. 
During his childhood Narendra, like many other Hindu children of his age, developed a 
love for the Hindu deities, of whom he had learnt from his mother. Particularly 
attracted by the heroic story of Rama and his faithful consort Sita, he procured their 
images, bedecked them with flowers, and worshipped them in his boyish fashion. But 
disillusionment came when he heard someone denounce marriage vehemently as a 
terrible bondage. When he had thought this over he discarded Rama and Sita as 
unworthy of worship. In their place he installed the image of Siva, the god of 
renunciation, who was the ideal of the yogis. Nevertheless he retained a fondness for 
the Ramayana. 
At this time he daily experienced a strange vision when he was about to fall asleep. 
Closing his eyes, he would see between his eyebrows a ball of light of changing 
colours, which would slowly expand and at last burst, bathing his whole body in a 
white radiance. Watching this light he would gradually fall asleep. Since it was a daily 
occurrence, he regarded the phenomenon as common to all people, and was surprised 
when a friend denied ever having seen such a thing. Years later, however, Narendra's 
spiritual teacher, Sri Ramakrishna, said to him, 'Naren, my boy, do you see a light 
when you go to sleep?' Ramakrishna knew that such a vision indicated a great spiritual 
past and an inborn habit of meditation. The vision of light remained with Narendra 
until the end of his life, though later it lost its regularity and intensity. 
While still a child Narendra practised meditation with a friend before the image of 
Siva. He had heard that the holy men of ancient India would become so absorbed in 
contemplation of God that their hair would grow and gradually enter into the earth, like 
the roots of the banyan tree. While meditating, therefore, he would open his eyes, now 
and then, to see if his own hair had entered into the earth. Even so, during meditation, 
he often became unconscious of the world. On one occasion he saw in a vision a 
luminous person of serene countenance who was carrying the staff and water-bowl of a 
monk. The apparition was about to say something when Naren became frightened and 
left the room. He thought later that perhaps this had been a vision of Buddha. 
At the age of six he was sent to a primary school. One day, however, he repeated at 
home some of the vulgar words that he had learnt from his classmates, whereupon his 
disgusted parents took him out of the school and appointed a private tutor, who 
conducted classes for him and some other children of the neighbourhood in the 
worship hall of the house. Naren soon showed a precocious mind and developed a keen 
memory. Very easily he learnt by heart the whole of a Sanskrit grammar and long 
passages from the Ramayana and the Mahabharata. Some of the friendships he made 
at this age lasted his whole lifetime. At school he was the undisputed leader. When 
playing his favourite game of 'King and the Court,' he would assume the role of the 
monarch and assign to his friends the parts of the ministers, commander-in-chief, and 
other state officials. He was marked from birth to be a leader of men, as his name 
Narendra (lord of men) signified. 
Even at that early age he questioned why one human being should be considered 
superior to another. In his father's office separate tobacco pipes were provided for 
clients belonging to the different castes, as orthodox Hindu custom required, and the 
pipe from which the Moslems smoked was set quite apart. Narendra once smoked 
tobacco from all the pipes, including the one marked for the Moslems, and when 
reprimanded, remarked, 'I cannot see what difference it makes.' 
During these early years, Narendra's future personality was influenced by his gifted 
father and his saintly mother, both of whom kept a chastening eye upon him. The 
father had his own manner of discipline. For example, when, in the course of an 
argument with his mother, the impetuous boy once uttered a few rude words and the 
report came to the father, Viswanath did not directly scold his son, but wrote with 
charcoal on the door of his room: 'Narendra today said to his mother — ' and added the 
words that had been used. He wanted Narendra's friends to know how rudely he had 
treated his mother. 
Another time Narendra bluntly asked his father, 'What have you done for me?' 
Instead of being annoyed, Viswanath said, 'Go and look at yourself in the mirror, and 
then you will know.' 
Still another day, Narendra said to his father, 'How shall I conduct myself in the world?' 
'Never show surprise at anything,' his father replied. 
This priceless advice enabled Narendranath, in his future chequered life, to preserve 
his serenity of mind whether dwelling with princes in their palaces or sharing the straw 
huts of beggars. 
The mother, Bhuvaneswari, played her part in bringing out Narendranath's innate 
virtues. When he told her, one day, of having been unjustly treated in school, she said 
to him, in consolation: 'My child, what does it matter, if you are in the right? Always 
follow the truth without caring about the result. Very often you may have to suffer 
injustice or unpleasant consequences for holding to the truth; but you must not, under 
any circumstances, abandon it.' Many years later Narendranath proudly said to an 
audience, 'I am indebted to my mother for whatever knowledge I have acquired.' 
One day, when he was fighting with his play-fellows, Narendra accidentally fell from 
the porch and struck his forehead against a stone. The wound bled profusely and left a 
permanent scar over his right eye. Years later, when Ramakrishna heard of this 
accident, he remarked: 'In a way it was a good thing. If he had not thus lost some of his 
blood, he would have created havoc in the world with his excessive energy.' 
In 1871, at the age of eight, Narendra entered high school. His exceptional intelligence 
was soon recognized by his teachers and classmates. Though at first reluctant to study 
English because of its foreign origin, he soon took it up with avidity. But the 
curriculum consumed very little of his time. He used most of his inexhaustible energy 
in outside activities. Games of various kinds, many of which he invented or improvised 
kept him occupied. He made an imitation gas-works and a factory for aerating water, 
these two novelties having just been introduced in Calcutta. He organized an amateur 
theatrical company and a gymnasium, and took lessons in fencing, wrestling, rowing, 
and other manly sports. He also tried his hand at the art of cooking. Intensely restless, 
he would soon tire of one pastime and seek a new one. With his friends he visited the 
museum and the zoological garden. He arbitrated the disputes of his play-fellows and 
was a favourite with the people of the neighbourhood. Everybody admired his courage, 
straight-forwardness, and simplicity. 
From an early age this remarkable youth had no patience with fear or superstition. One 
of his boyish pranks had been to climb a flowering tree belonging to a neighbour, 
pluck the flowers, and do other mischief. The owner of the tree, finding his 
remonstrances unheeded, once solemnly told Naren's friends that the tree was guarded 
by a white-robed ghost who would certainly wring their necks if they disturbed his 
peace. The boys were frightened and kept away. But Narendra persuaded them to 
follow him back, and he climbed the tree, enjoying his usual measure of fun, and broke 
some branches by way of further mischief. Turning to his friends, he then said: 'What 
asses you all are! See, my neck is still there. The old man's story is simply not true. 
Don't believe what others say unless you your-selves know it to be true.' 
These simple but bold words were an indication of his future message to the world. 
Addressing large audiences in the later years, he would often say: 'Do not believe in a 
thing because you have read about it in a book. Do not believe in a thing because 
another man has said it was true. Do not believe in words because they are hallowed by 
tradition. Find out the truth for yourself. Reason it out. That is realization.' 
The following incident illustrates his courage and presence of mind. He one day 
wished to set up a heavy trapeze in the gymnasium, and so asked the help of some 
people who were there. Among them was an English sailor. The trapeze fell and 
knocked the sailor unconscious, and the crowd, thinking him dead, ran away for fear of 
the police. But Naren tore a piece from his cloth, bandaged the sailor's wound, washed 
his face with water, and gradually revived him. Then he moved the wounded man to a 
neighbouring schoolhouse where he nursed him for a week. When the sailor had 
recovered, Naren sent him away with a little purse collected from his friends. 
All through this period of boyish play Narendra retained his admiration for the life of 
the wandering monk. Pointing to a certain line on the palm of his hand, he would say to 
his friends: 'I shall certainly become a sannyasin. A palmist has predicted it.' 
As Narendra grew into adolescence, his temperament showed a marked change. He 
became keen about intellectual matters, read serious books on history and literature, 
devoured newspapers, and attended public meetings. Music was his favourite pastime. 
He insisted that it should express a lofty idea and arouse the feelings of the musician. 
At the age of fifteen he experienced his first spiritual ecstasy. The family was 
journeying to Raipur in the Central Provinces, and part of the trip had to be made in a 
bullock cart. On that particular day the air was crisp and clear; the trees and creepers 
were covered with green leaves and many-coloured blossoms; birds of brilliant 
plumage warbled in the woods. The cart was moving along a narrow pass where the 
lofty peaks rising on the two sides almost touched each other. Narendra's eyes spied a 
large bee-hive in the cleft of a giant cliff, and suddenly his mind was filled with awe 
and reverence for the Divine Providence. He lost outer consciousness and lay thus in 
the cart for a long time. Even after returning to the sense-perceived world he radiated 
joy. 
Another interesting mental phenomenon may be mentioned here; for it was one often 
experienced by Narendranath. From boyhood, on first beholding certain people or 
places, he would feel that he had known them before; but how long before he could 
never remember. One day he and some of his companions were in a room in a friend's 
house, where they were discussing various topics. Something was mentioned, and 
Narendra felt at once that he had on a previous occasion talked about the same subject 
with the selfsame friends in that very house. He even correctly described every nook 
and corner of the building, which he had not seen before. He tried at first to explain 
this singular phenomenon by the doctrine of reincarnation, thinking that perhaps he had 
lived in that house in a previous life. But he dismissed the idea as improbable. Later he 
concluded that before his birth he must have had previsions of the people, places, and 
events that he was to experience in his present incarnation; that was why, he thought, 
he could recognize them as soon as they presented themselves to him. 
At Raipur Narendra was encouraged by his father to meet notable scholars and discuss 
with them various intellectual topics usually considered too abstruse for boys of his 
age. On such occasions he exhibited great mental power. From his father, Narendra 
had learnt the art of grasping the essentials of things, seeing truth from the widest and 
most comprehensive standpoints, and holding to the real issue under discussion. 
In 1879 the family returned to Calcutta, and Narendra within a short time graduated 
from high school in the first division. In the meantime he had read a great many 
standard books of English and Bengali literature. History was his favourite subject. He 
also acquired at this time an unusual method of reading a book and acquiring the 
knowledge of its subject-matter. To quote his own words: 'I could understand an author 
without reading every line of his book. I would read the first and last lines of a 
paragraph and grasp its meaning. Later I found that I could understand the subjectmatter by reading only the first and last lines of a page. Afterwards I could follow the 
whole trend of a writer's argument by merely reading a few lines, though the author 
himself tried to explain the subject in five or more pages.' 
Soon the excitement of his boyhood days was over, and in 1879 Narendranath entered 
the Presidency College of Calcutta for higher studies. After a year he joined the 
General Assembly's Institution, founded by the Scottish General Missionary Board and 
later known as the Scottish Church College. It was from Hastie, the principal of the 
college and the professor of English literature, that he first heard the name Sri 
Ramakrishna. 
In college Narendra, now a handsome youth, muscular and agile, though slightly 
inclined to stoutness, enjoyed serious studies. During the first two years he studied 
Western logic. Thereafter he specialized in Western philosophy and the ancient and 
modern history of the different European nations. His memory was prodigious. It took 
him only three days to assimilate Green's History of the English People. Often, on the 
eve of an examination, he would read the whole night, keeping awake by drinking 
strong tea or coffee. 
About this time he came in contact with Sri Ramakrishna; this event, as we shall 
presently see, was to become the major turning-point of his life. As a result of his 
association with Sri Ramakrishna, his innate spiritual yearning was stirred up, and he 
began to feel the transitoriness of the world and the futility of academic education. The 
day before his B.A. examination, he suddenly felt an all-consuming love for God and, 
standing before the room of a college-mate, was heard to sing with great feeling:
Sing ye, O mountains, O clouds, O great winds! 
Sing ye, sing ye, sing His glory! 
Sing with joy, all ye suns and moons and stars! 
Sing ye, sing ye, His glory!
The friends, surprised, reminded him of the next day's examination, but Narendra was 
unconcerned; the shadow of the approaching monastic life was fast falling on him. He 
appeared for the examination, however, and easily passed. 
About Narendra's scholarship, Professor Hastie once remarked: 'Narendra is a real 
genius. I have travelled far and wide, but have not yet come across a lad of his talents 
and possibilities even among the philosophical students in the German universities. He 
is bound to make his mark in life.' 
Narendra's many-sided genius found its expression in music, as well. He studied both 
instrumental and vocal music under expert teachers. He could play on many 
instruments, but excelled in singing. From a Moslem teacher he learnt Hindi, Urdu, and 
Persian songs, most of them of devotional nature. 
He also became associated with the Brahmo Samaj, an important religious movement 
of the time, which influenced him during this formative period of his life. 
The introduction of English education in India following the British conquest of the 
country brought Hindu society in contact with the intellectual and aggressive European 
culture. The Hindu youths who came under the spell of the new, dynamic way of life 
realized the many shortcomings of their own society. Under the Moslem rule, even 
before the coming of the British, the dynamic aspect of the Hindu culture had been 
suppressed and the caste-system stratified. The priests controlled the religious life of 
the people for their own selfish interest. Meaningless dogmas and lifeless ceremonies 
supplanted the invigorating philosophical teachings of the Upanishads and the 
Bhagavad Gita. The masses were exploited, moreover, by the landlords, and the lot of 
women was especially pitiable. Following the break-down of the Moslem rule, chaos 
reigned in every field of Indian life, social, political, religious, and economic. The 
newly introduced English education brought into sharp focus the many drawbacks of 
society, and various reform movements, both liberal and orthodox, were initiated to 
make the national life flow once more through healthy channels. 
The Brahmo Samaj, one of these liberal movements, captured the imagination of the 
educated youths of Bengal. Raja Rammohan Roy (1774-1833), the founder of this 
religious organization, broke away from the rituals, image worship, and priestcraft of 
orthodox Hinduism and exhorted his followers to dedicate themselves to the 'worship 
and adoration of the Eternal, the Unsearchable, the Immutable Being, who is the 
Author and the Preserver of the universe.' The Raja, endowed with a gigantic intellect, 
studied the Hindu, Moslem, Christian, and Buddhist scriptures and was the first Indian 
to realize the importance of the Western rational method for solving the diverse 
problems of Hindu society. He took a prominent part in the introduction of English 
education in India, which, though it at first produced a deleterious effect on the newly 
awakened Hindu consciousness, ultimately revealed to a few Indians the glorious 
heritage of their own indigenous civilization. 
Among the prominent leaders of the Brahmo Samaj who succeeded Rammohan Roy 
were Devendranath Tagore (1817-1905), a great devotee of the Upanishads, and 
Keshab Chandra Sen (1838-1884), who was inclined to the rituals and doctrines of 
Christianity. The Brahmo Samaj, under their leadership, discarded many of the 
conventions of Hinduism such as rituals and the worship of God through images. 
Primarily a reformist movement, it directed its main energy to the emancipation of 
women, the remarriage of Hindu widows, the abolition of early marriage, and the 
spread of mass education. Influenced by Western culture, the Brahmo Samaj upheld 
the supremacy of reason, preached against the uncritical acceptance of scriptural 
authority, and strongly supported the slogans of the French Revolution. The whole 
movement was intellectual and eclectic in character, born of the necessity of the times; 
unlike traditional Hinduism, it had no root in the spiritual experiences of saints and 
seers. Narendra, like many other contemporary young men, felt the appeal of its 
progressive ideas and became one of its members. But, as will be presently seen, the 
Brahmo Samaj could not satisfy the deep spiritual yearning of his soul. 
About this time Narendra was urged by his father to marry, and an opportunity soon 
presented itself. A wealthy man, whose daughter Narendra was asked to accept as his 
bride, offered to defray his expenses for higher studies in England so that he might 
qualify himself for the much coveted Indian Civil Service. Narendra refused. Other 
proposals of similar nature produced no different result. Apparently it was not his 
destiny to lead a householder's life. 
From boyhood Narendra had shown a passion for purity. Whenever his warm and 
youthful nature tempted him to walk into a questionable adventure, he was held back 
by an unseen hand. His mother had taught him the value of chastity and had made him 
observe it as a matter of honour, in loyalty to herself and the family tradition. But 
purity to Narendra was not a negative virtue, a mere abstention from carnal pleasures. 
To be pure, he felt, was to conserve an intense spiritual force that would later manifest 
itself in all the noble aspirations of life. He regarded himself as a brahmacharin, a 
celibate student of the Hindu tradition, who worked hard, prized ascetic disciplines, 
held holy things in reverence, and enjoyed clean words, thoughts, and acts. For 
according to the Hindu scriptures, a man, by means of purity, which is the greatest of 
all virtues, can experience the subtlest spiritual perceptions. In Naren it accounts for 
the great power of concentration, memory, and insight, and for his indomitable mental 
energy and physical stamina. 
In his youth Narendra used to see every night two visions, utterly dissimilar in nature, 
before falling asleep. One was that of a worldly man with an accomplished wife and 
children, enjoying wealth, luxuries, fame, and social position; the other, that of a 
sannyasin, a wandering monk, bereft of earthly security and devoted to the 
contemplation of God. Narendra felt that he had the power to realize either of these 
ideals; but when his mind reflected on their respective virtues, he was inevitably drawn 
to the life of renunciation. The glamour of the world would fade and disappear. His 
deeper self instinctively chose the austere path. 
For a time the congregational prayers and the devotional songs of the Brahmo Samaj 
exhilarated Narendra's mind, but soon he found that they did not give him any real 
spiritual experience. He wanted to realize God, the goal of religion, and so felt the 
imperative need of being instructed by a man who had seen God. 
In his eagerness he went to Devendranath, the venerable leader of the Brahmo Samaj, 
and asked him, even before the latter had uttered a word, 'Sir, have you seen God?' 
Devendranath was embarrassed and replied: 'My boy, you have the eyes of a yogi. You 
should practise meditation.' 
The youth was disappointed and felt that this teacher was not the man to help him in 
his spiritual struggle. But he received no better answer from the leaders of other 
religious sects. Then he remembered having heard the name of Ramakrishna 
Paramahamsa from Professor Hastie, who while lecturing his class on Wordsworth's 
poem The Excursion, had spoken of trances, remarking that such religious ecstasies 
were the result of purity and concentration. He had said, further, that an exalted 
experience of this kind was a rare phenomenon, especially in modern times. 'I have 
known,' he had said, 'only one person who has realized that blessed state, and he is 
Ramakrishna of Dakshineswar. You will understand trances if you visit the saint.' 
Narendra had also heard about Sri Ramakrishna from a relative, Ramchandra Datta, 
who was one of the foremost householder disciples of the Master. Learning of 
Narendra's unwillingness to marry and ascribing it to his desire to lead a spiritual life, 
Ramchandra had said to him, 'If you really want to cultivate spirituality, then visit 
Ramakrishna at Dakshineswar.' 
Narendra met Ramakrishna for the first time in November 1881 at the house of the 
Master's devotee Surendranath Mitra, the young man having been invited there to 
entertain the visitors with his melodious music. The Paramahamsa was much 
impressed by his sincerity and devotion, and after a few inquiries asked him to visit 
him at Dakshineswar. Narendra accepted. He wished to learn if Ramakrishna was the 
man to help him in his spiritual quest.
AT THE FEET OF RAMAKRISHNA
Ramakrishna, the God-man of modern times, was born on February 18, 1836, in the 
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
while watching a flight of snow-white cranes against a black sky overcast with rainclouds. He began to go into trances as he meditated on gods and goddesses. His father's 
death, which left the family in straitened circumstances, deepened his spiritual mood. 
And so, though at the age of sixteen he joined his brother in Calcutta, he refused to go 
on there with his studies; for, as he remarked, he was simply not interested in an 
education whose sole purpose was to earn mere bread and butter. He felt a deep 
longing for the realization of God. 
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
This foretaste of what was to follow made him God-intoxicated, and whetted his 
appetite for further experience. He wished to see God uninterruptedly, with eyes open 
as well as closed. He therefore abandoned himself recklessly to the practice of various 
extreme spiritual disciplines. To remove from his mind the least trace of the arrogance 
of his high brahmin caste, he used to clean stealthily the latrine at a pariah's house. 
Through a stern process of discrimination he effaced all sense of distinction between 
gold and clay. Purity became the very breath of his nostrils, and he could not regard a 
woman, even in a dream, in any other way except as his own mother or the Mother of 
the universe. For years his eyelids did not touch each other in sleep. And he was finally 
thought to be insane. 
Indeed, the stress of his spiritual practice soon told upon Ramakrishna's delicate body 
and he returned to Kamarpukur to recover his health. His relatives and old friends saw 
a marked change in his nature; for the gay boy had been transformed into a 
contemplative young man whose vision was directed to something on a distant horizon. 
His mother proposed marriage, and finding in this the will of the Divine Mother, 
Ramakrishna consented. He even indicated where the girl was to be found, namely, in 
the village of Jayrambati, only three miles away. Here lived the little Saradamani, a girl 
of five, who was in many respects very different from the other girls of her age. The 
child would pray to God to make her character as fragrant as the tuberose. Later, at 
Dakshineswar, she prayed to God to make her purer than the full moon, which, pure as 
it was, showed a few dark spots. The marriage was celebrated and Ramakrishna, 
participating, regarded the whole affair as fun or a new excitement. 
In a short while he came back to Dakshineswar and plunged again into the stormy life 
of religious experimentation. His mother, his newly married wife, and his relatives 
were forgotten. Now, however, his spiritual disciplines took a new course. He wanted 
to follow the time-honoured paths of the Hindu religion under the guidance of 
competent teachers, and they came to him one by one, nobody knew from where. One 
was a woman, under whom he practised the disciplines of Tantra and of the Vaishnava 
faith and achieved the highest result in an incredibly short time. It was she who 
diagnosed his physical malady as the manifestation of deep spiritual emotions and 
described his apparent insanity as the result of an agonizing love for God; he was 
immediately relieved. It was she, moreover, who first declared him to be an 
Incarnation of God, and she proved her statement before an assembly of theologians by 
scriptural evidence. Under another teacher, the monk Jatadhari, Ramakrishna delved 
into the mysteries of Rama worship and experienced Rama's visible presence. Further, 
he communed with God through the divine relationships of Father, Mother, Friend, and 
Beloved. By an austere sannyasin named Totapuri, he was initiated into the monastic 
life, and in three days he realized his complete oneness with Brahman, the 
undifferentiated Absolute, which is the culmination of man's spiritual endeavour. 
Totapuri himself had had to struggle for forty years to realize this identity. 
Ramakrishna turned next to Christianity and Islam, to practise their respective 
disciplines, and he attained the same result that he had attained through Hinduism. He 
was thereby convinced that these, too, were ways to the realization of Godconsciousness. Finally, he worshipped his own wife — who in the meantime had 
grown into a young woman of nineteen — as the manifestation of the Divine Mother of 
the universe and surrendered at her feet the fruit of his past spiritual practices. After 
this he left behind all his disciplines and struggles. For according to Hindu tradition, 
when the normal relationship between husband and wife, which is the strongest 
foundation of the worldly life, has been transcended and a man sees in his wife the 
divine presence, he then sees God everywhere in the universe. This is the culmination 
of the spiritual life. 
Ramakrishna himself was now convinced of his divine mission on earth and came to 
know that through him the Divine Mother would found a new religious order 
comprising those who would accept the doctrine of the Universal Religion which he 
had experienced. It was further revealed to him that anyone who had prayed to God 
sincerely, even once, as well as those who were passing through their final birth on 
earth, would accept him as their spiritual ideal and mould their lives according to his 
universal teaching. 
The people around him were bewildered to see this transformation of a man whom 
they had ridiculed only a short while ago as insane. The young priest had become 
God's devotee; the devotee, an ascetic; the ascetic, a saint; the saint, a man of 
realization; and the man of realization, a new Prophet. Like the full-blown blossom 
attracting bees, Ramakrishna drew to him men and women of differing faith, 
intelligence, and social position. He gave generously to all from the inexhaustible store 
house of divine wisdom, and everyone felt uplifted in his presence. But the Master 
himself was not completely satisfied. He longed for young souls yet untouched by the 
world, who would renounce everything for the realization of God and the service of 
humanity. He was literally consumed with this longing. The talk of worldly people was 
tasteless to him. He often compared such people to mixture of milk and water with the 
latter preponderating, and said that he had become weary of trying to prepare thick 
milk from the mixture. Evenings, when his anguish reached its limit, he would climb 
the roof of a building near the temple and cry at the top of his voice: 'Come, my boys! 
Oh, where are you all? I cannot bear to live without you!' A mother could not feel more 
intensely for her beloved children, a friend for his dearest friend, or a lover for her 
sweetheart. 
Shortly thereafter the young men destined to be his monastic disciples began to arrive. 
And foremost among them was Narendranath. 
The first meeting at Dakshineswar between the Master and Narendra was momentous. 
Sri Ramakrishna recognized instantaneously his future messenger. Narendra, careless 
about his clothes and general appearance, was so unlike the other young men who had 
accompanied him to the temple. His eyes were impressive, partly indrawn, indicating a 
meditative mood. He sang a few songs, and as usual poured into them his whole soul. 
His first song was this:
Let us go back once more, 
O mind, to our proper home! 
Here in this foreign land of earth Why should we wander aimlessly in stranger's guise? 
These living beings round about, 
And the five elements, 
Are strangers to you, all of them; none are your own. 
Why do you so forget yourself, 
In love with strangers, foolish mind? 
Why do you so forget your own? 
Mount the path of truth, 
O mind! Unflaggingly climb, 
With love as the lamp to light your way. 
As your provision on the journey, take with you 
The virtues, hidden carefully; 
For, like two highwaymen, 
Greed and delusion wait to rob you of your wealth. 
And keep beside you constantly, 
As guards to shelter you from harm, 
Calmness of mind and self-control. 
Companionship with holy men will be for you 
A welcome rest-house by the road; 
There rest your weary limbs awhile, asking your way, 
If ever you should be in doubt, 
Of him who watches there. 
If anything along the path should cause you fear, 
Then loudly shout the name of God; 
For He is ruler of that road, 
And even Death must bow to Him.
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
but who weeps for the vision of God? If one cries sincerely for God, one can surely see 
Him.' 
Narendra was astounded. For the first time, he was face to face with a man who 
asserted that he had seen God. For the first time, in fact, he was hearing that God could 
be seen. He could feel that Ramakrishna's words were uttered from the depths of an 
inner experience. They could not be doubted. Still he could not reconcile these words 
with Ramakrishna's strange conduct, which he had witnessed only a few minutes 
before. What puzzled Narendra further was Ramakrishna's normal behaviour in the 
presence of others. The young man returned to Calcutta bewildered, but yet with a 
feeling of inner peace. 
During his second visit to the Master, Narendra had an even stranger experience. After 
a minute or two Sri Ramakrishna drew near him in an ecstatic mood, muttered some 
words, fixed his eyes on him, and placed his right foot on Naren's body. At this touch 
Naren saw, with eyes open, the walls, the room, the temple garden — nay, the whole 
world — vanishing, and even himself disappearing into a void. He felt sure that he was 
facing death. He cried in consternation: 'What are you doing to me? I have my parents, 
brothers, and sisters at home.' 
The Master laughed and stroked Naren's chest, restoring him to his normal mood. He 
said, 'All right, everything will happen in due time.' 
Narendra, completely puzzled, felt that Ramakrishna had cast a hypnotic spell upon 
him. But how could that have been? Did he not pride himself in the possession of an 
iron will? He felt disgusted that he should have been unable to resist the influence of a 
madman. Nonetheless he felt a great inner attraction for Sri Ramakrishna. 
On his third visit Naren fared no better, though he tried his utmost to be on guard. Sri 
Ramakrishna took him to a neighbouring garden and, in a state of trance, touched him. 
Completely overwhelmed, Naren lost consciousness. 
Sri Ramakrishna, referring later to this incident, said that after putting Naren into a 
state of unconsciousness, he had asked him many questions about his past, his mission 
in the world, and the duration of his present life. The answer had only confirmed what 
he himself had thought about these matters. Ramakrishna told his other disciples that 
Naren had attained perfection even before this birth; that he was an adept in 
meditation; and that the day Naren recognized his true self, he would give up the body 
by an act of will, through yoga. Often he was heard to say that Naren was one of the 
Saptarshis, or Seven Sages, who live in the realm of the Absolute. He narrated to them 
a vision he had had regarding the disciple's spiritual heritage. 
Absorbed, one day, in samadhi, Ramakrishna had found that his mind was soaring 
high, going beyond the physical universe of the sun, moon, and stars, and passing into 
the subtle region of ideas. As it continued to ascend, the forms of gods and goddesses 
were left behind, and it crossed the luminous barrier separating the phenomenal 
universe from the Absolute, entering finally the transcendental realm. There 
Ramakrishna saw seven venerable sages absorbed in meditation. These, he thought, 
must have surpassed even the gods and goddesses in wisdom and holiness, and as he 
was admiring their unique spirituality he saw a portion of the undifferentiated Absolute 
become congealed, as it were, and take the form of a Divine Child. Gently clasping the 
neck of one of the sages with His soft arms, the Child whispered something in his ear, 
and at this magic touch the sage awoke from meditation. He fixed his half-open eyes 
upon the wondrous Child, who said in great joy: 'I am going down to earth. Won't you 
come with me?' With a benign look the sage expressed assent and returned into deep 
spiritual ecstasy. Ramakrishna was amazed to observe that a tiny portion of the sage, 
however, descended to earth, taking the form of light, which struck the house in 
Calcutta where Narendra's family lived, and when he saw Narendra for the first time, 
he at once recognized him as the incarnation of the sage. He also admitted that the 
Divine Child who brought about the descent of the rishi was none other than himself. 
The meeting of Narendra and Sri Ramakrishna was an important event in the lives of 
both. A storm had been raging in Narendra's soul when he came to Sri Ramakrishna, 
who himself had passed through a similar struggle but was now firmly anchored in 
peace as a result of his intimate communion with the Godhead and his realization of 
Brahman as the immutable essence of all things. 
A genuine product of the Indian soil and thoroughly acquainted with the spiritual 
traditions of India, Sri Ramakrishna was ignorant of the modern way of thinking. But 
Narendra was the symbol of the modern spirit. Inquisitive, alert, and intellectually 
honest, he possessed an open mind and demanded rational proof before accepting any 
conclusion as valid. As a loyal member of the Brahmo Samaj he was critical of image 
worship and the rituals of the Hindu religion. He did not feel the need of a guru, a 
human intermediary between God and man. He was even sceptical about the existence 
of such a person, who was said to be free from human limitations and to whom an 
aspirant was expected to surrender himself completely and offer worship as to God. 
Ramakrishna's visions of gods and goddesses he openly ridiculed, and called them 
hallucinations. 
For five years Narendra closely watched the Master, never allowing himself to be 
influenced by blind faith, always testing the words and actions of Sri Ramakrishna in 
the crucible of reason. It cost him many sorrows and much anguish before he accepted 
Sri Ramakrishna as the guru and the ideal of the spiritual life. But when the acceptance 
came, it was wholehearted, final, and irrevocable. The Master, too, was overjoyed to 
find a disciple who doubted, and he knew that Naren was the one to carry his message 
to the world. 
The inner process that gradually transformed the chrysalis of Narendra into a beautiful 
butterfly will for ever remain, like all deep spiritual mysteries, unknown to the outer 
world. People, however, noticed the growth of an intimate relationship between the 
loving, patient, and forgiving teacher and his imperious and stubborn disciple. The 
Master never once asked Naren to abandon reason. He met the challenge of Naren's 
intellect with his superior understanding, acquired through firsthand knowledge of the 
essence of things. When Naren's reasoning failed to solve the ultimate mystery, the 
teacher gave him the necessary insight. Thus, with infinite patience, love, and 
vigilance, he tamed the rebellious spirit, demanding complete obedience to moral and 
spiritual disciplines, without which the religious life can not be built on a firm 
foundation. 
The very presence of Narendranath would fill the Master's mind with indescribable joy 
and create ecstatic moods. He had already known, by many indications, of the 
disciple's future greatness, the manifestation of which awaited only the fullness of 
time, What others regarded in Naren as stubbornness or haughtiness appeared to Sri 
Ramakrishna as the expression of his manliness and self-reliance, born of his selfcontrol and innate purity. He could not bear the slightest criticism of Naren and often 
said: 'Let no one judge him hastily. People will never understand him fully.' 
Ramakrishna loved Narendranath because he saw him as the embodiment of Narayana, 
the Divine Spirit, undefiled by the foul breath of the world. But he was criticized for 
his attachment. Once a trouble-maker of twisted mind named Hazra, who lived with 
the Master at Dakshineswar, said to him, 'If you long for Naren and the other 
youngsters all the time, when will you think of God?' The Master was distressed by 
this thought. But it was at once revealed to him that though God dwelt in all beings, He 
was especially manifest in a pure soul like Naren. Relieved of his worries, he then said: 
'Oh, what a fool Hazra is! How he unsettled my mind! But why blame the poor fellow? 
How could he know?' 
Sri Ramakrishna was outspoken in Narendra's praise. This often embarrassed the 
young disciple, who would criticize the Master for what he termed a sort of infatuation. 
One day Ramakrishna spoke highly of Keshab Sen and the saintly Vijay Goswami, the 
two outstanding leaders of the Brahmo Samaj. Then he added: 'If Keshab possesses 
one virtue which has made him world-famous, Naren is endowed with eighteen such 
virtues. I have seen in Keshab and Vijay the divine light burning like a candle flame, 
but in Naren it shines with the radiance of the sun.' 
Narendra, instead of feeling flattered by these compliments, became annoyed and 
sharply rebuked the Master for what he regarded as his foolhardiness. 'I cannot help it,' 
the Master protested. 'Do you think these are my words? The Divine Mother showed 
me certain things about you, which I repeated. And She reveals to me nothing but the 
truth.' 
But Naren was hardly convinced. He was sure that these so-called revelations were 
pure illusions. He carefully explained to Sri Ramakrishna that, from the viewpoint of 
Western science and philosophy, very often a man was deceived by his mind, and that 
the chances of deception were greater when a personal attachment was involved. He 
said to the Master, 'Since you love me and wish to see me great, these fancies naturally 
come to your mind.' 
The Master was perplexed. He prayed to the Divine Mother for light and was told: 
'Why do you care about what he says? In a short time he will accept your every word 
as true.' 
On another occasion, when the Master was similarly reprimanded by the disciple, he 
was reassured by the Divine Mother. Thereupon he said to Naren with a smile: 'You 
are a rogue. I won't listen to you any more. Mother says that I love you because I see 
the Lord in you. The day I shall not see Him in you, I shall not be able to bear even the 
sight of you.' 
On account of his preoccupation with his studies, or for other reasons, Narendra could 
not come to Dakshineswar as often as Sri Ramakrishna wished. But the Master could 
hardly endure his prolonged absence. If the disciple had not visited him for a number 
of days, he would send someone to Calcutta to fetch him. Sometimes he went to 
Calcutta himself. One time, for example, Narendra remained away from Dakshineswar 
for several weeks; even the Master's eager importunities failed to bring him. Sri 
Ramakrishna knew that he sang regularly at the prayer meetings of the Brahmo Samaj, 
and so one day he made his way to the Brahmo temple that the disciple attended. 
Narendra was singing in the choir as the Master entered the hall, and when he heard 
Narendra's voice, Sri Ramakrishna fell into a deep ecstasy. The eyes of the 
congregation turned to him, and soon a commotion followed. Narendra hurried to his 
side. One of the Brahmo leaders, in order to stop the excitement, put out the lights. The 
young disciple, realizing that the Master's sudden appearance was the cause of the 
disturbance, sharply took him to task. The latter answered, with tears in his eyes, that 
he had simply not been able to keep himself away from Narendra. 
On another occasion, Sri Ramakrishna, unable to bear Narendra's absence, went to 
Calcutta to visit the disciple at his own home. He was told that Naren was studying in 
an attic in the second floor that could be reached only by a steep staircase. His nephew 
Ramlal, who was a sort of caretaker of the Master, had accompanied him, and with his 
help Sri Ramakrishna climbed a few steps. Narendra appeared at the head of the stair, 
and at the very sight of him Sri Ramakrishna exclaimed, 'Naren, my beloved!' and went 
into ecstasy. With considerable difficulty Naren and Ramlal helped him to finish 
climbing the steps, and as he entered the room the Master fell into deep samadhi. A 
fellow student who was with Naren at the time and did not know anything of religious 
trances, asked Naren in bewilderment, 'Who is this man?' 
'Never mind,' replied Naren. 'You had better go home now.' 
Naren often said that the 'Old Man,' meaning Ramakrishna, bound the disciple for ever 
to him by his love. 'What do worldly men,' he remarked, 'know about love? They only 
make a show of it. The Master alone loves us genuinely.' Naren, in return, bore a deep 
love for Sri Ramakrishna, though he seldom expressed it in words. He took delight in 
criticizing the Master's spiritual experiences as evidences of a lack of self-control. He 
made fun of his worship of Kali. 
'Why do you come here,' Sri Ramakrishna once asked him, 'if you do not accept Kali, 
my Mother?' 
'Bah! Must I accept Her,' Naren retorted, 'simply because I come to see you? I come to 
you because I love you.' 
'All right,' said the Master, 'ere long you will not only accept my blessed Mother, but 
weep in Her name.' 
Turning to his other disciples, he said: 'This boy has no faith in the forms of God and 
tells me that my visions are pure imagination. But he is a fine lad of pure mind. He 
does not accept anything without direct evidence. He has studied much and cultivated 
great discrimination. He has fine judgement.'
TRAINING OF THE DISCIPLE
It is hard to say when Naren actually accepted Sri Ramakrishna as his guru. As far as 
the master was concerned, the spiritual relationship was established at the first meeting 
at Dakshineswar, when he had touched Naren, stirring him to his inner depths. From 
that moment he had implicit faith in the disciple and bore him a great love. But he 
encouraged Naren in the independence of his thinking. The love and faith of the Master 
acted as a restraint upon the impetuous youth and became his strong shield against the 
temptations of the world. By gradual steps the disciple was then led from doubt to 
certainty, and from anguish of mind to the bliss of the Spirit. This, however, was not an 
easy attainment. 
Sri Ramakrishna, perfect teacher that he was, never laid down identical disciplines for 
disciples of diverse temperaments. He did not insist that Narendra should follow strict 
rules about food, nor did he ask him to believe in the reality of the gods and goddesses 
of Hindu mythology. It was not necessary for Narendra's philosophic mind to pursue 
the disciplines of concrete worship. But a strict eye was kept on Naren's practice of 
discrimination, detachment, self-control, and regular meditation. Sri Ramakrishna 
enjoyed Naren's vehement arguments with the other devotees regarding the dogmas 
and creeds of religion and was delighted to hear him tear to shreds their unquestioning 
beliefs. But when, as often happened, Naren teased the gentle Rakhal for showing 
reverence to the Divine Mother Kali, the Master would not tolerate these attempts to 
unsettle the brother disciple's faith in the forms of God. 
As a member of the Brahmo Samaj, Narendra accepted its doctrine of monotheism and 
the Personal God. He also believed in the natural depravity of man. Such doctrines of 
non-dualistic Vedanta as the divinity of the soul and the oneness of existence he 
regarded as blasphemy; the view that man is one with God appeared to him pure 
nonsense. When the master warned him against thus limiting God's infinitude and 
asked him to pray to God to reveal to him His true nature, Narendra smiled. One day 
he was making fun of Sri Ramakrishna's non-dualism before a friend and said, 'What 
can be more absurd than to say that this jug is God, this cup is God, and that we too are 
God?' Both roared with laughter. 
Just then the Master appeared. Coming to learn the cause of their fun, he gently 
touched Naren and plunged into deep samadhi. The touch produced a magic effect, and 
Narendra entered a new realm of consciousness. He saw the whole universe permeated 
by the Divine Spirit and returned home in a daze. While eating his meal, he felt the 
presence of Brahman in everything — in the food, and in himself too. While walking 
in the street, he saw the carriages, the horses, the crowd, and himself as if made of the 
same substance. After a few days the intensity of the vision lessened to some extent, 
but still he could see the world only as a dream. While strolling in a public park of 
Calcutta, he struck his head against the iron railing, several times, to see if they were 
real or a mere illusion of the mind. Thus he got a glimpse of non-dualism, the fullest 
realization of which was to come only later, at the Cossipore garden. 
Sri Ramakrishna was always pleased when his disciples put to the test his statements or 
behaviour before accepting his teachings. He would say: 'Test me as the moneychangers test their coins. You must not believe me without testing me thoroughly.' The 
disciples often heard him say that his nervous system had undergone a complete 
change as a result of his spiritual experiences, and that he could not bear the touch of 
any metal, such as gold or silver. One day, during his absence in Calcutta, Narendra 
hid a coin under Ramakrishna's bed. After his return when the Master sat on the bed, 
he started up in pain as if stung by an insect. The mattress was examined and the 
hidden coin was found. 
Naren, on the other hand, was often tested by the Master. One day, when he entered the 
Master's room, he was completely ignored. Not a word of greeting was uttered. A week 
later he came back and met with the same indifference, and during the third and fourth 
visits saw no evidence of any thawing of the Master's frigid attitude. 
At the end of a month Sri Ramakrishna said to Naren, 'I have not exchanged a single 
word with you all this time, and still you come.' 
The disciple replied: 'I come to Dakshineswar because I love you and want to see you. 
I do not come here to hear your words.' 
The Master was overjoyed. Embracing the disciple, he said: 'I was only testing you. I 
wanted to see if you would stay away on account of my outward indifference. Only a 
man of your inner strength could put up with such indifference on my part. Anyone 
else would have left me long ago.' 
On one occasion Sri Ramakrishna proposed to transfer to Narendranath many of the 
spiritual powers that he had acquired as a result of his ascetic disciplines and visions of 
God. Naren had no doubt concerning the Master's possessing such powers. He asked if 
they would help him to realize God. Sri Ramakrishna replied in the negative but added 
that they might assist him in his future work as a spiritual teacher. 'Let me realize God 
first,' said Naren, 'and then I shall perhaps know whether or not I want supernatural 
powers. If I accept them now, I may forget God, make selfish use of them, and thus 
come to grief.' Sri Ramakrishna was highly pleased to see his chief disciple's singleminded devotion. 
Several factors were at work to mould the personality of young Narendranath. 
Foremost of these were his inborn spiritual tendencies, which were beginning to show 
themselves under the influence of Sri Ramakrishna, but against which his rational mind 
put up a strenuous fight. Second was his habit of thinking highly and acting nobly, 
disciplines acquired from a mother steeped in the spiritual heritage of India. Third were 
his broadmindedness and regard for truth wherever found, and his sceptical attitude 
towards the religious beliefs and social conventions of the Hindu society of his time. 
These he had learnt from his English-educated father, and he was strengthened in them 
through his own contact with Western culture. 
With the introduction in India of English education during the middle of the nineteenth 
century, as we have seen, Western science, history, and philosophy were studied in the 
Indian colleges and universities. The educated Hindu youths, allured by the glamour, 
began to mould their thought according to this new light, and Narendra could not 
escape the influence. He developed a great respect for the analytical scientific method 
and subjected many of the Master's spiritual visions to such scrutiny. The English poets 
stirred his feelings, especially Wordsworth and Shelley, and he took a course in 
Western medicine to understand the functioning of the nervous system, particularly the 
brain and spinal cord, in order to find out the secrets of Sri Ramakrishna's trances. But 
all this only deepened his inner turmoil. 
John Stuart Mill's Three Essays on Religion upset his boyish theism and the easy 
optimism imbibed from the Brahmo Samaj. The presence of evil in nature and man 
haunted him and he could not reconcile it at all with the goodness of an omnipotent 
Creator. Hume's scepticism and Herbert Spencer's doctrine of the Unknowable filled 
his mind with a settled philosophical agnosticism. After the wearing out of his first 
emotional freshness and naivete, he was beset with a certain dryness and incapacity for 
the old prayers and devotions. He was filled with an ennui which he concealed, 
however, under his jovial nature. Music, at this difficult stage of his life, rendered him 
great help; for it moved him as nothing else and gave him a glimpse of unseen realities 
that often brought tears to his eyes. 
Narendra did not have much patience with humdrum reading, nor did he care to absorb 
knowledge from books as much as from living communion and personal experience. 
He wanted life to be kindled by life, and thought kindled by thought. He studied 
Shelley under a college friend, Brajendranath Seal, who later became the leading 
Indian philosopher of his time, and deeply felt with the poet his pantheism, impersonal 
love, and vision of a glorified millennial humanity. The universe, no longer a mere 
lifeless, loveless mechanism, was seen to contain a spiritual principle of unity. 
Brajendranath, moreover, tried to present him with a synthesis of the Supreme 
Brahman of Vedanta, the Universal Reason of Hegel, and the gospel of Liberty, 
Equality, and Fraternity of the French Revolution. By accepting as the principle of 
morals the sovereignty of the Universal Reason and the negation of the individual, 
Narendra achieved an intellectual victory over scepticism and materialism, but no 
peace of mind. 
Narendra now had to face a new difficulty. The 'ballet of bloodless categories' of Hegel 
and his creed of Universal Reason required of Naren a suppression of the yearning and 
susceptibility of his artistic nature and joyous temperament, the destruction of the 
cravings of his keen and acute senses, and the smothering of his free and merry 
conviviality. This amounted almost to killing his own true self. Further, he could not 
find in such a philosophy any help in the struggle of a hot-blooded youth against the 
cravings of the passions, which appeared to him as impure, gross, and carnal. Some of 
his musical associates were men of loose morals for whom he felt a bitter and 
undisguised contempt. 
Narendra therefore asked his friend Brajendra if the latter knew the way of deliverance 
from the bondage of the senses, but he was told only to rely upon Pure Reason and to 
identify the self with it, and was promised that through this he would experience an 
ineffable peace. The friend was a Platonic transcendentalist and did not have faith in 
what he called the artificial prop of grace, or the mediation of a guru. But the problems 
and difficulties of Narendra were very different from those of his intellectual friend. 
He found that mere philosophy was impotent in the hour of temptation and in the 
struggle for his soul's deliverance. He felt the need of a hand to save, to uplift, to 
protect — shakti or power outside his rational mind that would transform his 
impotence into strength and glory. He wanted a flesh-and-blood reality established in 
peace and certainty, in short, a living guru, who, by embodying perfection in the flesh, 
would compose the commotion of his soul. 
The leaders of the Brahmo Samaj, as well as those of the other religious sects, had 
failed. It was only Ramakrishna who spoke to him with authority, as none had spoken 
before, and by his power brought peace into the troubled soul and healed the wounds 
of the spirit. At first Naren feared that the serenity that possessed him in the presence 
of the Master was illusory, but his misgivings were gradually vanquished by the calm 
assurance transmitted to him by Ramakrishna out of his own experience of 
Satchidananda Brahman — Existence, Knowledge, and Bliss Absolute. (This account 
of the struggle of Naren's collegiate days summarizes an article on Swami 
Vivekananda by Brajendranath Seal, published in the Life of Swami Vivekananda by 
the Advaita Ashrama, Mayavati, India.) 
Narendra could not but recognize the contrast of the Sturm und Drang of his soul with 
the serene bliss in which Sri Ramakrishna was always bathed. He begged the Master to 
teach him meditation, and Sri Ramakrishna's reply was to him a source of comfort and 
strength. The Master said: 'God listens to our sincere prayer. I can swear that you can 
see God and talk with Him as intensely as you see me and talk with me. You can hear 
His words and feel His touch.' Further the Master declared: 'You may not believe in 
divine forms, but if you believe in an Ultimate Reality who is the Regulator of the 
universe, you can pray to Him thus: "O God, I do not know Thee. Be gracious to reveal 
to me Thy real nature." He will certainly listen to you if your prayer is sincere.' 
Narendra, intensifying his meditation under the Master's guidance, began to lose 
consciousness of the body and to feel an inner peace, and this peace would linger even 
after the meditation was over. Frequently he felt the separation of the body from the 
soul. Strange perceptions came to him in dreams, producing a sense of exaltation that 
persisted after he awoke. The guru was performing his task in an inscrutable manner, 
Narendra's friends observed only his outer struggle; but the real transformation was 
known to the teacher alone — or perhaps to the disciple too. 
In 1884, when Narendranath was preparing for the B.A. examination, his family was 
struck by a calamity. His father suddenly died, and the mother and children were 
plunged into great grief. For Viswanath, a man of generous nature, had lived beyond 
his means, and his death burdened the family with a heavy debt. Creditors, like hungry 
wolves, began to prowl about the door, and to make matters worse, certain relatives 
brought a lawsuit for the partition of the ancestral home. Though they lost it, Narendra 
was faced, thereafter, with poverty. As the eldest male member of the family, he had to 
find the wherewithal for the feeding of seven or eight mouths and began to hunt a job. 
He also attended the law classes. He went about clad in coarse clothes, barefoot, and 
hungry. Often he refused invitations for dinner from friends, remembering his starving 
mother, brothers, and sisters at home. He would skip family meals on the fictitious plea 
that he had already eaten at a friend's house, so that the people at home might receive a 
larger share of the scanty food. The Datta family was proud and would not dream of 
soliciting help from outsiders. With his companions Narendra was his usual gay self. 
His rich friends no doubt noticed his pale face, but they did nothing to help. Only one 
friend sent occasional anonymous aid, and Narendra remained grateful to him for life. 
Meanwhile, all his efforts to find employment failed. Some friends who earned money 
in a dishonest way asked him to join them, and a rich woman sent him an immoral 
proposal, promising to put an end to his financial distress. But Narendra gave to these a 
blunt rebuff. Sometimes he would wonder if the world were not the handiwork of the 
Devil — for how could one account for so much suffering in God's creation? 
One day, after a futile search for a job, he sat down, weary and footsore, in the big park 
of Calcutta in the shadow of the Ochterlony monument. There some friends joined him 
and one of them sang a song, perhaps to console him, describing God's abundant grace. 
Bitterly Naren said: 'Will you please stop that song? Such fancies are, no doubt, 
pleasing to those who are born with silver spoons in their mouths. Yes, there was a 
time when I, too, thought like that. But today these ideas appear to me a mockery.' 
The friends were bewildered. 
One morning, as usual, Naren left his bed repeating God's name, and was about to go 
out in search of work after seeking divine blessings. His mother heard the prayer and 
said bitterly: 'Hush, you fool! You have been crying yourself hoarse for God since your 
childhood. Tell me what has God done for you?' Evidently the crushing poverty at 
home was too much for the pious mother. 
These words stung Naren to the quick. A doubt crept into his mind about God's 
existence and His Providence. 
It was not in Naren's nature to hide his feelings. He argued before his friends and the 
devotees of Sri Ramakrishna about God's non-existence and the futility of prayer even 
if God existed. His over-zealous friends thought he had become an atheist and ascribed 
to him many unmentionable crimes, which he had supposedly committed to forget his 
misery. Some of the devotees of the Master shared these views. Narendra was angry 
and mortified to think that they could believe him to have sunk so low. He became 
hardened and justified drinking and the other dubious pleasures resorted to by 
miserable people for a respite from their suffering. He said, further, that he himself 
would not hesitate to follow such a course if he were assured of its efficacy. Openly 
asserting that only cowards believed in God for fear of hell-fire, he argued the 
possibility of God's non-existence and quoted Western philosophers in support of his 
position. And when the devotees of the Master became convinced that he was 
hopelessly lost, he felt a sort of inner satisfaction. 
A garbled report of the matter reached Sri Ramakrishna, and Narendra thought that 
perhaps the Master, too, doubted his moral integrity. The very idea revived his anger. 
'Never mind,' he said to himself. 'If good or bad opinion of a man rests on such flimsy 
grounds, I don't care.' 
But Narendra was mistaken. For one day Bhavanath, a devotee of the master and an 
intimate friend of Narendra, cast aspersions on the latter's character, and the Master 
said angrily: 'Stop, you fool! The Mother has told me that it is simply not true. I shan't 
look at your face if you speak to me again that way.' 
The fact was that Narendra could not, in his heart of hearts, disbelieve in God. He 
remembered the spiritual visions of his own boyhood and many others that he had 
experienced in the company of the Master. Inwardly he longed to understand God and 
His ways. And one day he gained this understanding. It happened in the following way: 
He had been out since morning in a soaking rain in search of employment, having had 
neither food nor rest for the whole day. That evening he sat down on the porch of a 
house by the roadside, exhausted. He was in a daze. Thoughts began to flit before his 
mind, which he could not control. Suddenly he had a strange vision, which lasted 
almost the whole night. He felt that veil after veil was removed from before his soul, 
and he understood the reconciliation of God's justice with His mercy. He came to know 
— but he never told how — that misery could exist in the creation of a compassionate 
God without impairing His sovereign power or touching man's real self. He understood 
the meaning of it all and was at peace. Just before daybreak, refreshed both in body 
and in mind, he returned home. 
This revelation profoundly impressed Narendranath. He became indifferent to people's 
opinion and was convinced that he was not born to lead an ordinary worldly life, 
enjoying the love of a wife and children and physical luxuries. He recalled how the 
several proposals of marriage made by his relatives had come to nothing, and he 
ascribed all this to God's will. The peace and freedom of the monastic life cast a spell 
upon him. He determined to renounce the world, and set a date for this act. Then, 
coming to learn that Sri Ramakrishna would visit Calcutta that very day, he was happy 
to think that he could embrace the life of a wandering monk with his guru's blessings. 
When they met, the Master persuaded his disciple to accompany him to Dakshineswar. 
As they arrived in his room, Sri Ramakrishna went into an ecstatic mood and sang a 
song, while tears bathed his eyes. The words of the song clearly indicated that the 
Master knew of the disciple's secret wish. When other devotees asked him about the 
cause of his grief, Sri Ramakrishna said, 'Oh, never mind, it is something between me 
and Naren, and nobody else's business.' At night he called Naren to his side and said 
with great feeling: 'I know you are born for Mother's work. I also know that you will be 
a monk. But stay in the world as long as I live, for my sake at least.' He wept again. 
Soon after, Naren procured a temporary job, which was sufficient to provide a hand-tomouth living for the family. 
One day Narendra asked himself why, since Kali, the Divine Mother listened to Sri 
Ramakrishna prayers, should not the Master pray to Her to relieve his poverty. When 
he told Sri Ramakrishna about this idea, the latter inquired why he did not pray himself 
to Kali, adding that Narendranath suffered because he did not acknowledge Kali as the 
Sovereign Mistress of the universe. 
'Today,' the Master continued, 'is a Tuesday, an auspicious day for the Mother's 
worship. Go to Her shrine in the evening, prostrate yourself before the image, and pray 
to Her for any boon; it will be granted. Mother Kali is the embodiment of Love and 
Compassion. She is the Power of Brahman. She gives birth to the world by Her mere 
wish. She fulfils every sincere prayer of Her devotees.' 
At nine o'clock in the evening, Narendranath went to the Kali temple. Passing through 
the courtyard, he felt within himself a surge of emotion, and his heart leapt with joy in 
anticipation of the vision of the Divine Mother. Entering the temple, he cast his eyes 
upon the image and found the stone figure to be nothing else but the living Goddess, 
the Divine Mother Herself, ready to give him any boon he wanted — either a happy 
worldly life or the joy of spiritual freedom. He was in ecstasy. He prayed for the boon 
of wisdom, discrimination, renunciation, and Her uninterrupted vision, but forgot to 
ask the Deity for money. He felt great peace within as he returned to the Master's 
room, and when asked if he had prayed for money, was startled. He said that he had 
forgotten all about it. The Master told him to go to the temple again and pray to the 
Divine Mother to satisfy his immediate needs. Naren did as he was bidden, but again 
forgot his mission. The same thing happened a third time. Then Naren suddenly 
realized that Sri Ramakrishna himself had made him forget to ask the Divine Mother 
for worldly things; perhaps he wanted Naren to lead a life of renunciation. So he now 
asked Sri Ramakrishna to do something for the family. The master told the disciple that 
it was not Naren's destiny to enjoy a worldly life, but assured him that the family 
would be able to eke out a simple existence. 
The above incident left a deep impression upon Naren's mind; it enriched his spiritual 
life, for he gained a new understanding of the Godhead and Its ways in the phenomenal 
universe. Naren's idea of God had hitherto been confined either to that of a vague 
Impersonal Reality or to that of an extracosmic Creator removed from the world. He 
now realized that the Godhead is immanent in the creation, that after projecting the 
universe from within Itself, It has entered into all created entities as life and 
consciousness, whether manifest or latent. This same immanent Spirit, or the World 
Soul, when regarded as a person creating, preserving, and destroying the universe, is 
called the Personal God, and is worshipped by different religions through such a 
relationship as that of father, mother, king, or beloved. These relationships, he came to 
understand, have their appropriate symbols, and Kali is one of them. 
Embodying in Herself creation and destruction, love and terror, life and death, Kali is 
the symbol of the total universe. The eternal cycle of the manifestation and nonmanifestation of the universe is the breathing-out and breathing-in of this Divine 
Mother. In one aspect She is death, without which there cannot be life. She is smeared 
with blood, since without blood the picture of the phenomenal universe is not 
complete. To the wicked who have transgressed Her laws, She is the embodiment of 
terror, and to the virtuous, the benign Mother. Before creation She contains within Her 
womb the seed of the universe, which is left from the previous cycle. After the 
manifestation of the universe She becomes its preserver and nourisher, and at the end 
of the cycle She draws it back within Herself and remains as the undifferentiated Sakti, 
the creative power of Brahman. She is non-different from Brahman. When free from 
the acts of creation, preservation, and destruction, the Spirit, in Its acosmic aspect, is 
called Brahman; otherwise It is known as the World Soul or the Divine Mother of the 
universe. She is therefore the doorway to the realization of the Absolute; She is the 
Absolute. To the daring devotee who wants to see the transcendental Absolute, She 
reveals that form by withdrawing Her phenomenal aspect. Brahman is Her 
transcendental aspect. She is the Great Fact of the universe, the totality of created 
beings. She is the Ruler and the Controller. 
All this had previously been beyond Narendra's comprehension. He had accepted the 
reality of the phenomenal world and yet denied the reality of Kali. He had been 
conscious of hunger and thirst, pain and pleasure, and the other characteristics of the 
world, and yet he had not accepted Kali, who controlled them all. That was why he had 
suffered. But on that auspicious Tuesday evening the scales dropped from his eyes. He 
accepted Kali as the Divine Mother of the universe. He became Her devotee. 
Many years later he wrote to an American lady: 'Kali worship is my special fad.' But 
he did not preach Her in public, because he thought that all that modern man required 
was to be found in the Upanishads. Further, he realized that the Kali symbol would not 
be understood by universal humanity. 
Narendra enjoyed the company of the Master for six years, during which time his 
spiritual life was moulded. Sri Ramakrishna was a wonderful teacher in every sense of 
the word. Without imposing his ideas upon anyone, he taught more by the silent 
influence of his inner life than by words or even by personal example. To live near him 
demanded of the disciple purity of thought and concentration of mind. He often 
appeared to his future monastic followers as their friend and playmate. Through fun 
and merriment he always kept before them the shining ideal of God-realization. He 
would not allow any deviation from bodily and mental chastity, nor any compromise 
with truth and renunciation. Everything else he left to the will of the Divine Mother. 
Narendra was his 'marked' disciple, chosen by the Lord for a special mission. Sri 
Ramakrishna kept a sharp eye on him, though he appeared to give the disciple every 
opportunity to release his pent-up physical and mental energy. Before him, Naren often 
romped about like a young lion cub in the presence of a firm but indulgent parent. His 
spiritual radiance often startled the Master, who saw that maya, the Great Enchantress, 
could not approach within 'ten feet' of that blazing fire. 
Narendra always came to the Master in the hours of his spiritual difficulties. One time 
he complained that he could not meditate in the morning on account of the shrill note 
of a whistle from a neighbouring mill, and was advised by the Master to concentrate on 
the very sound of the whistle. In a short time he overcame the distraction. Another time 
he found it difficult to forget the body at the time of meditation. Sri Ramakrishna 
sharply pressed the space between Naren's eyebrows and asked him to concentrate on 
that sensation. The disciple found this method effective. 
Witnessing the religious ecstasy of several devotees, Narendra one day said to the 
Master that he too wanted to experience it. 'My child,' he was told, 'when a huge 
elephant enters a small pond, a great commotion is set up, but when it plunges into the 
Ganga, the river shows very little agitation. These devotees are like small ponds; a 
little experience makes their feelings flow over the brim. But you are a huge river.' 
Another day the thought of excessive spiritual fervour frightened Naren. The Master 
reassured him by saying: 'God is like an ocean of sweetness; wouldn't you dive into it? 
Suppose there is a bowl filled with syrup, and you are a fly, hungry for the sweet 
liquid. How would you like to drink it?' Narendra said that he would sit on the edge of 
the bowl, otherwise he might be drowned in the syrup and lose his life. 'But,' the 
Master said, 'you must not forget that I am talking of the Ocean of Satchidananda, the 
Ocean of Immortality. Here one need not be afraid of death. Only fools say that one 
should not have too much of divine ecstasy. Can anybody carry to excess the love of 
God? You must dive deep in the Ocean of God.' 
On one occasion Narendra and some of his brother disciples were vehemently arguing 
about God's nature — whether He was personal or impersonal, whether Divine 
Incarnation was fact or myth, and so forth and so on. Narendra silenced his opponents 
by his sharp power of reasoning and felt jubilant at his triumph. Sri Ramakrishna 
enjoyed the discussion and after it was over sang in an ecstatic mood:
How are you trying, O my mind,
to know the nature of God?
You are groping like a madman
locked in a dark room.
He is grasped through ecstatic love;
how can you fathom Him without it?
Only through affirmation, never negation,
can you know Him;
Neither through Veda nor through Tantra
nor the six darsanas.
All fell silent, and Narendra realized the inability of the intellect to fathom God's 
mystery. 
In his heart of hearts Naren was a lover of God. Pointing to his eyes, Ramakrishna said 
that only a bhakta possessed such a tender look; the eyes of the jnani were generally 
dry. Many a time, in his later years, Narendra said, comparing his own spiritual attitude 
with that of the Master: 'He was a jnani within, but a bhakta without; but I am a bhakta 
within, and a jnani without.' He meant that Ramakrishna's gigantic intellect was hidden 
under a thin layer of devotion, and Narendra's devotional nature was covered by a 
cloak of knowledge. 
We have already referred to the great depth of Sri Ramakrishna's love for his beloved 
disciple. He was worried about the distress of Naren's family and one day asked a 
wealthy devotee if he could not help Naren financially. Naren's pride was wounded and 
he mildly scolded the Master. The latter said with tears in his eyes: 'O my Naren! I can 
do anything for you, even beg from door to door.' Narendra was deeply moved but said 
nothing. Many days after, he remarked, 'The Master made me his slave by his love for 
me.' 
This great love of Sri Ramakrishna enabled Naren to face calmly the hardships of life. 
Instead of hardening into a cynic, he developed a mellowness of heart. But, as will be 
seen later, Naren to the end of his life was often misunderstood by his friends. A bold 
thinker, he was far ahead of his time. Once he said: 'Why should I expect to be 
understood? It is enough that they love me. After all, who am I? The Mother knows 
best. She can do Her own work. Why should I think myself to be indispensable?' 
The poverty at home was not an altogether unmitigated evil. It drew out another side of 
Naren's character. He began to feel intensely for the needy and afflicted. Had he been 
nurtured in luxury, the Master used to say, he would perhaps have become a different 
person — a statesman, a lawyer, an orator, or a social reformer. But instead, he 
dedicated his life to the service of humanity. 
Sri Ramakrishna had had the prevision of Naren's future life of renunciation. Therefore 
he was quite alarmed when he came to know of the various plans made by Naren's 
relatives for his marriage. Prostrating himself in the shrine of Kali, he prayed 
repeatedly: 'O Mother! Do break up these plans. Do not let him sink in the quagmire of 
the world.' He closely watched Naren and warned him whenever he discovered the 
trace of an impure thought in his mind. 
Naren's keen mind understood the subtle implications of Sri Ramakrishna's teachings. 
One day the Master said that the three salient disciplines of Vaishnavism were love of 
God's name, service to the devotees, and compassion for all living beings. But he did 
not like the word compassion and said to the devotees: 'How foolish to speak of 
compassion! Man is an insignificant worm crawling on the earth — and he to show 
compassion to others! This is absurd. It must not be compassion, but service to all. 
Recognize them as God's manifestations and serve them.' 
The other devotees heard the words of the Master but could hardly understand their 
significance. Naren, however fathomed the meaning. Taking his young friends aside, 
he said that Sri Ramakrishna's remarks had thrown wonderful light on the philosophy 
of non-dualism with its discipline of non-attachment, and on that of dualism with its 
discipline of love. The two were not really in conflict. A non-dualist did not have to 
make his heart dry as sand, nor did he have to run away from the world. As Brahman 
alone existed in all men, a non-dualist must love all and serve all. Love, in the true 
sense of the word, is not possible unless one sees God in others. Naren said that the 
Master's words also reconciled the paths of knowledge and action. An illumined person 
did not have to remain inactive; he could commune with Brahman through service to 
other embodied beings, who also are embodiments of Brahman. 
'If it be the will of God,' Naren concluded, 'I shall one day proclaim this noble truth 
before the world at large. I shall make it the common property of all — the wise and 
the fool, the rich and the poor, the brahmin and the pariah.' 
Years later he expressed these sentiments in a noble poem which concluded with the 
following words:
Thy God is here before thee now, 
Revealed in all these myriad forms: 
Rejecting them, where seekest thou 
His presence? He who freely shares 
His love with every living thing 
Proffers true service unto God.
It was Sri Ramakrishna who re-educated Narendranath in the essentials of Hinduism. 
He, the fulfilment of the spiritual aspirations of the three hundred millions of Hindus 
for the past three thousand years, was the embodiment of the Hindu faith. The beliefs 
Narendra had learnt on his mother's lap had been shattered by a collegiate education, 
but the young man now came to know that Hinduism does not consist of dogmas or 
creeds; it is an inner experience, deep and inclusive, which respects all faiths, all 
thoughts, all efforts and all realizations. Unity in diversity is its ideal. 
Narendra further learnt that religion is a vision which, at the end, transcends all barriers 
of caste and race and breaks down the limitations of time and space. He learnt from the 
Master that the Personal God and worship through symbols ultimately lead the devotee 
to the realization of complete oneness with the Deity. The Master taught him the 
divinity of the soul, the non-duality of the Godhead, the unity of existence, and the 
harmony of religions. He showed Naren by his own example how a man in this very 
life could reach perfection, and the disciple found that the Master had realized the same 
God-consciousness by following the diverse disciplines of Hinduism, Christianity, and 
Islam. 
One day the Master, in an ecstatic mood, said to the devotees: 'There are many 
opinions and many ways. I have seen them all and do not like them any more. The 
devotees of different faiths quarrel among themselves. Let me tell you something. You 
are my own people. There are no strangers around. I clearly see that God is the whole 
and I am a part of Him. He is the Lord and I am His servant. And sometimes I think He 
is I and I am He.' 
Narendra regarded Sri Ramakrishna as the embodiment of the spirit of religion and did 
not bother to know whether he was or not an Incarnation of God. He was reluctant to 
cast the Master in any theological mould. It was enough for Naren if he could see 
through the vista of Ramakrishna's spiritual experiences all the aspects of the Godhead. 
How did Narendra impress the other devotees of the Master, especially the youngsters? 
He was their idol. They were awed by his intellect and fascinated by his personality. In 
appearance he was a dynamic youth, overflowing with vigour and vitality, having a 
physical frame slightly over middle height and somewhat thickset in the shoulders. He 
was graceful without being feminine. He had a strong jaw, suggesting his staunch will 
and fixed determination. The chest was expansive, and the breadth of the head towards 
the front signified high mental power and development. 
But the most remarkable thing about him was his eyes, which Sri Ramakrishna 
compared to lotus petals. They were prominent but not protruding, and part of the time 
their gaze was indrawn, suggesting the habit of deep meditation; their colour varied 
according to the feeling of the moment. Sometimes they would be luminous in 
profundity, and sometimes they sparkled in merriment. Endowed with the native grace 
of an animal, he was free in his movements. He walked sometimes with a slow gait and 
sometimes with rapidity, always a part of his mind absorbed in deep thought. And it 
was a delight to hear his resonant voice, either in conversation or in music. 
But when Naren was serious his face often frightened his friends. In a heated 
discussion his eyes glowed. If immersed in his own thoughts, he created such an air of 
aloofness that no one dared to approach him. Subject to various moods, sometimes he 
showed utter impatience with his environment, and sometimes a tenderness that melted 
everybody's heart. His smile was bright and infectious. To some he was a happy 
dreamer, to some he lived in a real world rich with love and beauty, but to all he 
unfailingly appeared a scion of an aristocratic home. 
And how did the Master regard his beloved disciple? To quote his own words: 
'Narendra belongs to a very high plane — the realm of the Absolute. He has a manly 
nature. So many devotees come here, but there is no one like him. 
'Every now and then I take stock of the devotees. I find that some are like lotuses with 
ten petals, some like lotuses with a hundred petals. But among lotuses Narendra is a 
thousand-petalled one. 
'Other devotees may be like pots or pitchers; but Narendra is a huge water-barrel. 
'Others may be like pools or tanks; but Narendra is a huge reservoir like the 
Haldarpukur. 
'Among fish, Narendra is a huge red-eyed carp; others are like minnows or smelts or 
sardines. 
'Narendra is a "very big receptacle", one that can hold many things. He is like a 
bamboo with a big hollow space inside. 
'Narendra is not under the control of anything. He is not under the control of 
attachment or sense pleasures. He is like a male pigeon. If you hold a male pigeon by 
its beak, it breaks away from you; but the female pigeon keeps still. I feel great 
strength when Narendra is with me in a gathering.' 
Sometime about the middle of 1885 Sri Ramakrishna showed the first symptoms of a 
throat ailment that later was diagnosed as cancer. Against the advice of the physicians, 
he continued to give instruction to spiritual seekers, and to fall into frequent trances. 
Both of these practices aggravated the illness. For the convenience of the physicians 
and the devotees, he was at first removed to a house in the northern section of Calcutta 
and then to a garden house at Cossipore, a suburb of the city. Narendra and the other 
young disciples took charge of nursing him. Disregarding the wishes of their 
guardians, the boys gave up their studies or neglected their duties at home, at least 
temporarily, in order to devote themselves heart and soul to the service of the Master. 
His wife, known among the devotees as the Holy Mother, looked after the cooking; the 
older devotees met the expenses. All regarded this service to the guru as a blessing and 
privilege. 
Narendra time and again showed his keen insight and mature judgement during Sri 
Ramakrishna's illness. Many of the devotees, who looked upon the Master as God's 
Incarnation and therefore refused to see in him any human frailty, began to give a 
supernatural interpretation of his illness. They believed that it had been brought about 
by the will of the Divine Mother or the Master himself to fulfil an inscrutable purpose, 
and that it would be cured without any human effort after the purpose was fulfilled. 
Narendra said, however, that since Sri Ramakrishna was a combination of God and 
man the physical element in him was subject to such laws of nature as birth, growth, 
decay, and destruction. He refused to give the Master's disease, a natural phenomenon, 
any supernatural explanation. Nonetheless, he was willing to shed his last drop of 
blood in the service of Sri Ramakrishna. 
Emotion plays an important part in the development of the spiritual life. While intellect 
removes the obstacles, it is emotion that gives the urge to the seeker to move forward. 
But mere emotionalism without the disciplines of discrimination and renunciation 
often leads him astray. He often uses it as a short cut to trance or ecstasy. Sri 
Ramakrishna, no doubt, danced and wept while singing God's name and experienced 
frequent trances; but behind his emotion there was the long practice of austerities and 
renunciation. His devotees had not witnessed the practice of his spiritual disciplines. 
Some of them, especially the elderly householders, began to display ecstasies 
accompanied by tears and physical contortions, which in many cases, as later appeared, 
were the result of careful rehearsal at home or mere imitation of Sri Ramakrishna's 
genuine trances. Some of the devotees, who looked upon the Master as a Divine 
Incarnation, thought that he had assumed their responsibilities, and therefore they 
relaxed their own efforts. Others began to speculate about the part each of them was 
destined to play in the new dispensation of Sri Ramakrishna. In short, those who 
showed the highest emotionalism posed as the most spiritually advanced. 
Narendra's alert mind soon saw this dangerous trend in their lives. He began to make 
fun of the elders and warned his young brother disciples about the harmful effect of 
indulging in such outbursts. Real spirituality, he told them over and over again, was the 
eradication of worldly tendencies and the development of man's higher nature. He 
derided their tears and trances as symptoms of nervous disorder, which should be 
corrected by the power of the will, and, if necessary, by nourishing food and proper 
medical treatment. Very often, he said, unwary devotees of God fall victims to mental 
and physical breakdown. 'Of one hundred persons who take up the spiritual life,' he 
grimly warned, 'eighty turn out to be charlatans, fifteen insane, and only five, maybe, 
get a glimpse of the real truth. Therefore, beware.' He appealed to their inner strength 
and admonished them to keep away from all sentimental nonsense. He described to the 
young disciples Sri Ramakrishna's uncompromising self-control, passionate yearning 
for God, and utter renunciation of attachment to the world, and he insisted that those 
who loved the Master should apply his teachings in their lives. 
Sri Ramakrishna, too, coming to realize the approaching end of his mortal existence, 
impressed it upon the devotees that the realization of God depended upon the giving up 
of lust and greed. The young disciples became grateful to Narendranath for thus 
guiding them during the formative period of their spiritual career. They spent their 
leisure hours together in meditation, study, devotional music, and healthy spiritual 
discussions. 
The illness of Sri Ramakrishna showed no sign of abatement; the boys redoubled their 
efforts to nurse him, and Narendra was constantly by their side, cheering them 
whenever they felt depressed. One day he found them hesitant about approaching the 
Master. They had been told that the illness was infectious. Narendra dragged them to 
the Master's room. Lying in a corner was a cup containing part of the gruel which Sri 
Ramakrishna could not swallow. It was mixed with his saliva. Narendra seized the cup 
and swallowed its contents. This set at rest the boys' misgivings. 
Narendra, understanding the fatal nature of Sri Ramakrishna's illness and realizing that 
the beloved teacher would not live long, intensified his own spiritual practices. His 
longing for the vision of God knew no limit. One day he asked the Master for the boon 
of remaining merged in samadhi three or four days at a stretch, interrupting his 
meditation now and then for a bite of food. 'You are a fool,' said the Master. 'There is a 
state higher than that. It is you who sing: "O Lord! Thou art all that exists."' Sri 
Ramakrishna wanted the disciple to see God in all beings and to serve them in a spirit 
of worship. He often said that to see the world alone, without God, is ignorance, 
ajnana; to see God alone, without the world, is a kind of philosophical knowledge, 
jnana; but to see all beings permeated by the spirit of God is supreme wisdom, vijnana. 
Only a few blessed souls could see God dwelling in all. He wanted Naren to attain this 
supreme wisdom. So the master said to him, 'Settle your family affairs first, then you 
shall know a state even higher than samadhi.' 
On another occasion, in response to a similar request, Sri Ramakrishna said to Naren: 
'Shame on you! You are asking for such an insignificant thing. I thought that you 
would be like a big banyan tree, and that thousands of people would rest in your shade. 
But now I see that you are seeking your own liberation.' Thus scolded, Narendra shed 
profuse tears. He realized the greatness of Sri Ramakrishna's heart. 
An intense fire was raging within Narendra's soul. He could hardly touch his college 
books; he felt it was a dreadful thing to waste time in that way. One morning he went 
home but suddenly experienced an inner fear. He wept for not having made much 
spiritual progress, and hurried to Cossipore almost unconscious of the outside world. 
His shoes slipped off somewhere, and as he ran past a rick of straw some of it stuck to 
his clothes. Only after entering the Master's room did he feel some inner peace. 
Sri Ramakrishna said to the other disciples present: 'Look at Naren's state of mind. 
Previously he did not believe in the Personal God or divine forms. Now he is dying for 
God's vision.' The Master then gave Naren certain spiritual instructions about 
meditation. 
Naren was being literally consumed by a passion for God. The world appeared to him 
to be utterly distasteful. When the Master reminded him of his college studies, the 
disciple said, 'I would feel relieved if I could swallow a drug and forget all I have 
learnt' He spent night after night in meditation under the tress in the Panchavati at 
Dakshineswar, where Sri Ramakrishna, during the days of his spiritual discipline, had 
contemplated God. He felt the awakening of the Kundalini (The spiritual energy, 
usually dormant in man, but aroused by the practice of spiritual disciplines. See 
glossary.) and had other spiritual visions. 
One day at Cossipore Narendra was meditating under a tree with Girish, another 
disciple. The place was infested with mosquitoes. Girish tried in vain to concentrate his 
mind. Casting his eyes on Naren, he saw him absorbed in meditation, though his body 
appeared to be covered by a blanket of the insects. 
A few days later Narendra's longing seemed to have reached the breaking-point. He 
spent an entire night walking around the garden house at Cossipore and repeating 
Rama's name in a heart-rending manner. In the early hours of the morning Sri 
Ramakrishna heard his voice, called him to his side, and said affectionately: 'Listen, 
my child, why are you acting that way? What will you achieve by such impatience?' 
He stopped for a minute and then continued: 'See, Naren. What you have been doing 
now, I did for twelve long years. A storm raged in my head during that period. What 
will you realize in one night?' 
But the master was pleased with Naren's spiritual struggle and made no secret of his 
wish to make him his spiritual heir. He wanted Naren to look after the young disciples. 
'I leave them in your care,' he said to him. 'Love them intensely and see that they 
practise spiritual disciplines even after my death, and that they do not return home.' He 
asked the young disciples to regard Naren as their leader. It was an easy task for them. 
Then, one day, Sri Ramakrishna initiated several of the young disciples into the 
monastic life, and thus himself laid the foundation of the future Ramakrishna Order of 
monks. 
Attendance on the Master during his sickness revealed to Narendra the true import of 
Sri Ramakrishna's spiritual experiences. He was amazed to find that the Master could 
dissociate himself from all consciousness of the body by a mere wish, at which time he 
was not aware of the least pain from his ailment. Constantly he enjoyed an inner bliss, 
in spite of the suffering of the body, and he could transmit that bliss to the disciples by 
a mere touch or look. To Narendra, Sri Ramakrishna was the vivid demonstration of 
the reality of the Spirit and the unsubstantiality of matter. 
One day the Master was told by a scholar that he could instantly cure himself of his 
illness by concentrating his mind on his throat. This Sri Ramakrishna refused to do 
since he could never withdraw his mind from God. But at Naren's repeated request, the 
Master agreed to speak to the Divine Mother about his illness. A little later he said to 
the disciple in a sad voice: 'Yes, I told Her that I could not swallow any food on 
account of the sore in my throat, and asked Her to do something about it. But the 
Mother said, pointing to you all, "Why, are you not eating enough through all these 
mouths?" I felt so humiliated that I could not utter another word.' Narendra realized 
how Sri Ramakrishna applied in life the Vedantic idea of the oneness of existence and 
also came to know that only through such realization could one rise above the pain and 
suffering of the individual life. 
To live with Sri Ramakrishna during his illness was in itself a spiritual experience. It 
was wonderful to witness how he bore with his pain. In one mood he would see that 
the Divine Mother alone was the dispenser of pleasure and pain and that his own will 
was one with the Mother's will, and in another mood he would clearly behold, the utter 
absence of diversity, God alone becoming men, animals, gardens, houses, roads, 'the 
executioner, the victim, and the slaughter-post,' to use the Master's own words. 
Narendra saw in the Master the living explanation of the scriptures regarding the divine 
nature of the soul and the illusoriness of the body. Further, he came to know that Sri 
Ramakrishna had attained to that state by the total renunciation of 'woman' and 'gold,' 
which, indeed, was the gist of his teaching. Another idea was creeping into Naren's 
mind. He began to see how the transcendental Reality, the Godhead, could embody 
Itself as the Personal God, and the Absolute become a Divine Incarnation. He was 
having a glimpse of the greatest of all divine mysteries: the incarnation of the Father as 
the Son for the redemption of the world. He began to believe that God becomes man so 
that man may become God. Sri Ramakrishna thus appeared to him in a new light. 
Under the intellectual leadership of Narendranath, the Cossipore garden house became 
a miniature university. During the few moments' leisure snatched from nursing and 
meditation, Narendra would discuss with his brother disciples religions and 
philosophies, both Eastern and Western. Along with the teachings of Sankara, Krishna, 
and Chaitanya, those of Buddha and Christ were searchingly examined. 
Narendra had a special affection for Buddha, and one day suddenly felt a strong desire 
to visit Bodh-Gaya, where the great Prophet had attained enlightenment. With Kali and 
Tarak, two of the brother disciples, he left, unknown to the others, for that sacred place 
and meditated for long hours under the sacred Bo-tree. Once while thus absorbed he 
was overwhelmed with emotion and, weeping profusely, embraced Tarak. Explaining 
the incident, he said afterwards that during the meditation he keenly felt the presence 
of Buddha and saw vividly how the history of India had been changed by his noble 
teachings; pondering all this he could not control his emotion. 
Back in Cossipore, Narendra described enthusiastically to the Master and the brother 
disciples of Buddha's life, experiences, and teachings. Sri Ramakrishna in turn related 
some of his own experiences. Narendra had to admit that the Master, after the 
attainment of the highest spiritual realization, had of his own will kept his mind on the 
phenomenal plane. 
He further understood that a coin, however valuable, which belonged to an older period 
of history, could not be used as currency at a later date. God assumes different forms in 
different ages to serve the special needs of the time. 
Narendra practised spiritual disciplines with unabating intensity. Sometimes he felt an 
awakening of a spiritual power that he could transmit to others. One night in March 
1886, he asked his brother disciple Kali to touch his right knee, and then entered into 
deep meditation. Kali's hand began to tremble; he felt a kind of electric shock. 
Afterwards Narendra was rebuked by the Master for frittering away spiritual powers 
before accumulating them in sufficient measure. He was further told that he had 
injured Kali's spiritual growth, which had been following the path of dualistic 
devotion, by forcing upon the latter some of his own non-dualistic ideas. The Master 
added, however, that the damage was not serious. 
Narendra had had enough of visions and manifestations of spiritual powers, and he 
now wearied of them. His mind longed for the highest experience of non-dualistic 
Vedanta, the nirvikalpa samadhi, in which the names and forms of the phenomenal 
world disappear and the aspirant realizes total non-difference between the individual 
soul, the universe, and Brahman, or the Absolute. He told Sri Ramakrishna about it, but 
the master remained silent. And yet one evening the experience came to him quite 
unexpectedly. 
He was absorbed in his usual meditation when he suddenly felt as if a lamp were 
burning at the back of his head. The light glowed more and more intensely and finally 
burst. Narendra was overwhelmed by that light and fell unconscious. After some time, 
as he began to regain his normal mood, he could feel only his head and not the rest of 
his body. 
In an agitated voice he said to Gopal, a brother disciple who was meditating in the 
same room, 'Where is my body?' 
Gopal answered: 'Why, Naren, it is there. Don't you feel it?' 
Gopal was afraid that Narendra was dying, and ran to Sri Ramakrishna's room. He 
found the Master in a calm but serious mood, evidently aware of what had happened in 
the room downstairs. After listening to Gopal the Master said, 'Let him stay in that 
state for a while; he has teased me long enough for it.' 
For some time Narendra remained unconscious. When he regained his normal state of 
mind he was bathed in an ineffable peace. As he entered Sri Ramakrishna's room the 
latter said: 'Now the Mother has shown you everything. But this realization, like the 
jewel locked in a box, will be hidden away from you and kept in my custody. I will 
keep the key with me. Only after you have fulfilled your mission on this earth will the 
box be unlocked, and you will know everything as you have known now'. 
The experience of this kind of samadhi usually has a most devastating effect upon the 
body; Incarnations and special messengers of God alone can survive its impact. By 
way of advice, Sri Ramakrishna asked Naren to use great discrimination about his food 
and companions, only accepting the purest. 
Later the master said to the other disciples: 'Narendra will give up his body of his own 
will. When he realizes his true nature, he will refuse to stay on this earth. Very soon he 
will shake the world by his intellectual and spiritual powers. I have prayed to the 
Divine Mother to keep away from him the Knowledge of the Absolute and cover his 
eyes with a veil of maya. There is much work to be done by him. But the veil, I see, is 
so thin that it may be rent at any time.' 
Sri Ramakrishna, the Avatar of the modern age, was too gentle and tender to labour 
himself, for humanity's welfare. He needed some sturdy souls to carry on his work. 
Narendra was foremost among those around him; therefore Sri Ramakrishna did not 
want him to remain immersed in nirvikalpa samadhi before his task in this world was 
finished. 
The disciples sadly watched the gradual wasting away of Sri Ramakrishna's physical 
frame. His body became a mere skeleton covered with skin; the suffering was intense. 
But he devoted his remaining energies to the training of the disciples, especially 
Narendra. He had been relieved of his worries about Narendra; for the disciple now 
admitted the divinity of Kali, whose will controls all things in the universe. Naren said 
later on: 'From the time he gave me over to the Divine Mother, he retained the vigour 
of his body only for six months. The rest of the time — and that was two long years — 
he suffered.' 
One day the Master, unable to speak even in a whisper, wrote on a piece of paper: 
'Narendra will teach others.' The disciple demurred. Sri Ramakrishna replied: 'But you 
must. Your very bones will do it.' He further said that all the supernatural powers he 
had acquired would work through his beloved disciple. 
A short while before the curtain finally fell on Sri Ramakrishna's earthly life, the 
Master one day called Naren to his bedside. Gazing intently upon him, he passed into 
deep meditation. Naren felt that a subtle force, resembling an electric current, was 
entering his body. He gradually lost outer consciousness. After some time he regained 
knowledge of the physical world and found the Master weeping. Sri Ramakrishna said 
to him: 'O Naren, today I have given you everything I possess — now I am no more 
than a fakir, a penniless beggar. By the powers I have transmitted to you, you will 
accomplish great things in the world, and not until then will you return to the source 
whence you have come.' 
Narendra from that day became the channel of Sri Ramakrishna's powers and the 
spokesman of his message. 
Two days before the dissolution of the Master's body, Narendra was standing by the 
latter's bedside when a strange thought flashed into his mind: Was the Master truly an 
Incarnation of God? He said to himself that he would accept Sri Ramakrishna's divinity 
if the Master, on the threshold of death, declared himself to be an Incarnation. But this 
was only a passing thought. He stood looking intently at the Master face. Slowly Sri 
Ramakrishna's lips parted and he said in a clear voice: 'O my Naren, are you still not 
convinced? He who in the past was born as Rama and Krishna is now living in this 
very body as Ramakrishna — but not from the standpoint of your Vedanta.' Thus Sri 
Ramakrishna, in answer to Narendra's mental query, put himself in the category of 
Rama and Krishna, who are recognized by orthodox Hindus as two of the Avatars, or 
Incarnations of God. 
A few words may be said here about the meaning of the Incarnation in the Hindu 
religious tradition. One of the main doctrines of Vedanta is the divinity of the soul: 
every soul, in reality, is Brahman. Thus it may be presumed that there is no difference 
between an Incarnation and an ordinary man. To be sure, from the standpoint of the 
Absolute, or Brahman, no such difference exists. But from the relative standpoint, 
where multiplicity is perceived, a difference must be admitted. Embodied human 
beings reflect godliness in varying measure. In an Incarnation this godliness is fully 
manifest. Therefore an Incarnation is unlike an ordinary mortal or even an illumined 
saint. To give an illustration: There is no difference between a clay lion and a clay 
mouse, from the standpoint of the clay. Both become the same substance when 
dissolved into clay. But the difference between the lion and the mouse, from the 
standpoint of form, is clearly seen. Likewise, as Brahman, an ordinary man is identical 
with an Incarnation. Both become the same Brahman when they attain final 
illumination. But in the relative state of name and form, which is admitted by Vedanta, 
the difference between them is accepted. According to the Bhagavad Gita (IV. 6-8), 
Brahman in times of spiritual crisis assumes a human body through Its own inscrutable 
power, called maya. Though birthless, immutable, and the Lord of all beings, yet in 
every age Brahman appears to be incarnated in a human body for the protection of the 
good and the destruction of the wicked. 
As noted above, the Incarnation is quite different from an ordinary man, even from a 
saint. Among the many vital differences may be mentioned the fact that the birth of an 
ordinary mortal is governed by the law of karma, whereas that of an Incarnation is a 
voluntary act undertaken for the spiritual redemption of the world. Further, though 
maya is the cause of the embodiment of both an ordinary mortal and an Incarnation, 
yet the former is fully under maya's control, whereas the latter always remains its 
master. A man, though potentially Brahman, is not conscious of his divinity; but an 
Incarnation is fully aware of the true nature of His birth and mission. The spiritual 
disciplines practised by an Incarnation are not for His own liberation, but for the 
welfare of humanity; as far as He is concerned, such terms as bondage and liberation 
have no meaning, He being ever free, ever pure, and ever illumined. Lastly, an 
Incarnation can bestow upon others the boon of liberation, whereas even an illumined 
saint is devoid of such power. 
Thus the Master, on his death-bed, proclaimed himself through his own words as the 
Incarnation or God-man of modern times. 
On August 15, 1886, the Master's suffering became almost unbearable. After midnight 
he felt better for a few minutes. He summoned Naren to his beside and gave him the 
last instructions, almost in a whisper. The disciples stood around him. At two minutes 
past one in the early morning of August 16, Sri Ramakrishna uttered three times in a 
ringing voice the name of his beloved Kali and entered into the final samadhi, from 
which his mind never again returned to the physical world. 
The body was given to the fire in the neighbouring cremation ground on the bank of 
the Ganga. But to the Holy Mother, as she was putting on the signs of a Hindu widow, 
there came these words of faith and reassurance: 'I am not dead. I have just gone from 
one room to another.' 
As the disciples returned from the cremation ground to the garden house, they felt great 
desolation. Sri Ramakrishna had been more than their earthly father. His teachings and 
companionship still inspired them. They felt his presence in his room. His words rang 
in their ears. But they could no longer see his physical body or enjoy his seraphic 
smile. They all yearned to commune with him. 
Within a week of the Master's passing away, Narendra one night was strolling in the 
garden with a brother disciple, when he saw in front of him a luminous figure. There 
was no mistaking: it was Sri Ramakrishna himself. Narendra remained silent, regarding 
the phenomenon as an illusion. But his brother disciple exclaimed in wonder, 'See, 
Naren! See!' There was no room for further doubt. Narendra was convinced that it was 
Sri Ramakrishna who had appeared in a luminous body. As he called to the other 
brother disciples to behold the Master, the figure disappeared.
AS A WANDERING MONK
Among the Master's disciples, Tarak, Latu, and the elder Gopal had already cut off 
their relationship with their families. The young disciples whom Sri Ramakrishna had 
destined for the monastic life were in need of a shelter. The Master had asked Naren to 
see to it that they should not become householders. Naren vividly remembered the 
Master's dying words: 'Naren, take care of the boys.' The householder devotees, 
moreover, wanted to meet, from time to time, at a place where they could talk about 
the Master. They longed for the company of the young disciples who had totally 
dedicated their lives to the realization of God. But who would bear the expenses of a 
house where the young disciples could live? How would they be provided with food 
and the basic necessaries of life? 
All these problems were solved by the generosity of Surendranath Mitra, the beloved 
householder disciple of Sri Ramakrishna. He came forward to pay the expenses of new 
quarters for the Master's homeless disciples. A house was rented at Baranagore, 
midway between Calcutta and Dakshineswar. Dreary and dilapidated, it was a building 
that had the reputation of being haunted by evil spirits. The young disciples were 
happy to take refuge in it from the turmoil of Calcutta. This Baranagore Math, as the 
new monastery was called, became the first headquarters of the monks of the 
Ramakrishna Order. Its centre was the shrine room, where the copper vessel containing 
the sacred ashes of the Master was daily worshipped as his visible presence.
Narendranath devoted himself heart and soul to the training of the young brother 
disciples. He spent the day-time at home, supervising a lawsuit that was pending in the 
court and looking after certain other family affairs; but during the evenings and nights 
he was always with his brothers at the monastery, exhorting them to practise spiritual 
disciplines. His presence was a source of unfailing delight and inspiration to all. 
The future career of the youths began to take shape during these early days at 
Baranagore. The following incident hastened the process. At the invitation of the 
mother of Baburam, one of the disciples, they all went to the village of Antpur to spend 
a few days away from the austerities of Baranagore. Here they realized, more intensely 
than ever before, a common goal of life, a sense of brotherhood and unity integrating 
their minds and hearts. Their consecrated souls were like pearls in a necklace held 
together by the thread of Ramakrishna's teachings. They saw in one another a reservoir 
of spiritual power, and the vision intensified their mutual love and respect. Narendra, 
describing to them the glories of the monastic life, asked them to give up the glamour 
of academic studies and the physical world, and all felt in their hearts the ground swell 
of the spirit of renunciation. This reached its height one night when they were sitting 
for meditation around a fire, in the fashion of Hindu monks. The stars sparkled 
overhead and the stillness was unbroken except for the crackling of the firewood. 
Suddenly Naren opened his eyes and began, with an apostolic fervour, to narrate to the 
brother disciples the life of Christ. He exhorted them to live like Christ, who had had 
no place 'to lay his head.' Inflamed by a new passion, the youths, making God and the 
sacred fire their witness, vowed to become monks. 
When they had returned to their rooms in a happy mood, someone found out that it was 
Christmas Eve, and all felt doubly blest. It is no wonder that the monks of the 
Ramakrishna Order have always cherished a high veneration for Jesus of Nazareth. 
The young disciples, after their return to Baranagore, finally renounced home and 
became permanent inmates of the monastery. And what a life of austerity they lived 
there! They forgot their food when absorbed in meditation, worship, study, or 
devotional music. At such times Sashi, who had constituted himself their caretaker, 
literally dragged them to the dining-room. The privations they suffered during this 
period form a wonderful saga of spiritual discipline. Often there would be no food at 
all, and on such occasions they spent day and night in prayer and meditation. 
Sometimes there would be only rice, with no salt for flavouring; but nobody cared. 
They lived for months on boiled rice, salt, and bitter herbs. Not even demons could 
have stood such hardship. Each had two pieces of loin-cloth, and there were some 
regular clothes that were worn, by turns, when anyone had to go out. They slept on 
straw mats spread on the hard floor. A few pictures of saints, gods, and goddesses hung 
on the walls, and some musical instruments lay here and there. The library contained 
about a hundred books. 
But Narendra did not want the brother disciples to be pain-hugging, cross-grained 
ascetics. They should broaden their outlook by assimilating the thought-currents of the 
world. He examined with them the histories of different countries and various 
philosophical systems. Aristotle and Plato, Kant and Hegel, together with 
Sankaracharya and Buddha, Ramanuja and Madhva, Chaitanya and Nimbarka, were 
thoroughly discussed. The Hindu philosophical systems of Jnana, Bhakti, Yoga, and 
Karma, each received a due share of attention, and their apparent contradictions were 
reconciled in the light of Sri Ramakrishna's teachings and experiences. The dryness of 
discussion was relieved by devotional music. There were many moments, too, when 
the inmates indulged in light-hearted and witty talk, and Narendra's bons mots on such 
occasions always convulsed them with laughter. But he would never let them forget the 
goal of the monastic life: the complete control of the lower nature, and the realization 
of God. 
'During those days,' one of the inmates of the monastery said, 'he worked like a 
madman. Early in the morning, while it was still dark, he would rise from bed and 
wake up the others, singing, "Awake, arise, all who would drink of the Divine Nectar!" 
And long after midnight he and his brother disciples would still be sitting on the roof 
of the monastery building, absorbed in religious songs. The neighbours protested, but 
to no avail. Pandits came and argued. He was never for one moment idle, never dull.' 
Yet the brother complained that they could not realize even a fraction of what 
Ramakrishna had taught. 
Some of the householder devotees of the Master, however, did not approve of the 
austerities of the young men, and one of them teasingly inquired if they had realized 
God by giving up the world. 'What do you mean?' Narendra said furiously. 'Suppose 
we have not realized God; must we then return to the life of the senses and deprave our 
higher nature?' 
Soon the youth of the Baranagore monastery became restless for the life of the 
wandering monk with no other possessions except staff and begging-bowl. Thus they 
would learn self-surrender to God, detachment, and inner serenity. They remembered 
the Hindu proverb that the monk who constantly moves on, remains pure, like water 
that flows. They wanted to visit the holy places and thus give an impetus to their 
spiritual life. 
Narendra, too, wished to enjoy the peace of solitude. He wanted to test his own inner 
strength as well as teach others not to depend upon him always. Some of the brother 
disciples had already gone away from the monastery when he began his wanderings. 
The first were in the nature of temporary excursions; he had to return to Baranagore in 
response to the appeal of the inmates of the monastery. But finally in 1890, when he 
struck out again — without a name and with only a staff and begging-bowl — he was 
swallowed in the immensity of India and the dust of the vast subcontinent completely 
engulfed him. When rediscovered by his brother monks he was no longer the unknown 
Naren, but the Swami Vivekananda who had made history in Chicago in 1893. 
In order to satisfy his wanderlust, Narendra went to Varanasi, considered the holiest 
place in India — a city sanctified from time out of mind by the association of monks 
and devotees. Here have come prophets like Buddha, Sankaracharya, and Chaitanya, to 
receive, as it were, the commandment of God to preach their messages. The Ganga 
charges the atmosphere with a rare holiness. Narendra felt uplifted by the spirit of 
renunciation and devotion that pervades this sacred place. He visited the temples and 
paid his respects to such holy men as Trailanga Swami, who lived on the bank of the 
Ganga constantly absorbed in meditation, and Swami Bhaskarananda, who annoyed 
Naren by expressing doubt as to the possibility of a man's total conquest of the 
temptation of 'woman' and 'gold.' With his own eyes Naren had seen the life of Sri 
Ramakrishna, who had completely subdued his lower nature. 
In Varanasi, one day, hotly pursued by a troop of monkeys, he was running away when 
a monk called to him: 'Face the brutes.' He stopped and looked defiantly at the ugly 
beasts. They quickly disappeared. Later, as a preacher, he sometimes used this 
experience to exhort people to face the dangers and vicissitudes of life and not run 
away from them. 
After a few days Naren returned to Baranagore and plunged into meditation, study, and 
religious discourses. From this time he began to feel a vague premonition of his future 
mission. He often asked himself if such truths of the Vedanta philosophy as the 
divinity of the soul and the unity of existence should remain imprisoned in the wormeaten pages of the scriptures to furnish a pastime for erudite scholars or to be enjoyed 
only by solitary monks in caves and the depths of the wilderness; did they not have any 
significance for the average man struggling with life's problems? Must the common 
man, because of his ignorance of the scriptures, be shut out from the light of Vedanta? 
Narendra spoke to his brother disciples about the necessity of preaching the strengthgiving message of the Vedanta philosophy to one and all, and especially to the 
downtrodden masses. But these monks were eager for their own salvation, and 
protested. Naren said to them angrily: 'All are preaching. What they do unconsciously, 
I will do consciously. Ay, even if you, my brother monks, stand in my way, I will go to 
the pariahs and preach in the lowest slums.' 
After remaining at Baranagore a short while, Naren set out again for Varanasi, where 
he met the Sanskrit scholar Pramadadas Mitra. These two felt for each other a mutual 
respect and affection, and they discussed, both orally and through letters, the social 
customs of the Hindus and abstruse passages of the scriptures. Next he visited 
Ayodhya, the ancient capital of Rama, the hero of the Ramayana. Lucknow, a city of 
gardens and palaces created by the Moslem Nawabs, filled his mind with the glorious 
memories of Islamic rule, and the sight of the Taj Mahal in Agra brought tears to his 
eyes. In Vrindavan he recalled the many incidents of Krishna's life and was deeply 
moved. 
While on his way to Vrindavan, trudging barefoot and penniless, Naren saw a man 
seated by the roadside enjoying a smoke. He asked the stranger to give him a puff from 
his tobacco bowl, but the man was an untouchable and shrank from such an act; for it 
was considered sacrilegious by Hindu society. Naren continued on his way, but said to 
himself suddenly: 'What a shame! The whole of my life I have contemplated the nonduality of the soul, and now I am thrown into the whirlpool of the caste-system. How 
difficult it is to get over innate tendencies!' He returned to the untouchable, begged him 
to lend him his smoking-pipe, and in spite of the remonstrances of the low-caste man, 
enjoyed a hearty smoke and went on to Vrindavan. 
Next we find Naren at the railroad station of Hathras, on his way to the sacred 
pilgrimage centre of Hardwar in the foothills of the Himalayas. The station-master, 
Sarat Chandra Gupta, was fascinated at the very first sight of him. 'I followed the two 
diabolical eyes,' he said later. Narendra accepted Sarat as a disciple and called him 'the 
child of my spirit', At Hathras he discussed with visitors the doctrines of Hinduism and 
entertained them with music, and then one day confided to Sarat that he must move on. 
'My son,' he said, 'I have a great mission to fulfil and I am in despair at the smallness of 
my power. My guru asked me to dedicate my life to the regeneration of my 
motherland. Spirituality has fallen to a low ebb and starvation stalks the land. India 
must become dynamic again and earn the respect of the world through her spiritual 
power.' 
Sarat immediately renounced the world and accompanied Narendra from Hathras to 
Hardwar. The two then went on to Hrishikesh, on the bank of the Ganga several miles 
north of Hardwar, where they found themselves among monks of various sects, who 
were practising meditation and austerities. Presently Sarat fell ill and his companion 
took him back to Hathras for treatment. But Naren, too, had been attacked with malaria 
fever at Hrishikesh. He now made his way to the Baranagore monastery. 
Naren had now seen northern India, the Aryavarta, the sacred land of the Aryans, 
where the spiritual culture of India had originated and developed. The main stream of 
this ancient Indian culture, issuing from the Vedas and the Upanishads and branching 
off into the Puranas and the Tantras, was subsequently enriched by contributions from 
such foreign peoples as the Saks, the Huns, the Greeks, the Pathans, and the Moguls. 
Thus India developed a unique civilization based upon the ideal of unity in diversity. 
Some of the foreign elements were entirely absorbed into the traditional Hindu 
consciousness; others, though flavoured by the ancient thought of the land, retained 
their individuality. Realizing the spiritual unity of India and Asia, Narendra discovered 
the distinctive characteristics of Oriental civilization: renunciation of the finite and 
communion with the Infinite. 
But the stagnant life of the Indian masses, for which he chiefly blamed the priests and 
the landlords, saddened his heart. Naren found that his country's downfall had not been 
caused by religion. On the contrary, as long as India had clung to her religious ideals, 
the country had over flowed with material prosperity. But the enjoyment of power for a 
long time had corrupted the priests. The people at large were debarred from true 
knowledge of religion, and the Vedas, the source of the Hindu culture, were 
completely forgotten, especially in Bengal. Moreover, the caste-system, which had 
originally been devised to emphasize the organic unity of Hindu society, was now 
petrified. Its real purpose had been to protect the weak from the ruthless competition of 
the strong and to vindicate the supremacy of spiritual knowledge over the power of 
military weapons, wealth, and organized labour; but now it was sapping the vitality of 
the masses. Narendra wanted to throw open the man-making wisdom of the Vedas to 
all, and thus bring about the regeneration of his motherland. He therefore encouraged 
his brothers at the Barangaore monastery to study the grammar of Panini, without 
which one could not acquire first-hand knowledge of the Vedas. 
The spirit of democracy and equality in Islam appealed to Naren's mind and he wanted 
to create a new India with Vedantic brain and Moslem body. Further, the idea began to 
dawn in his mind that the material conditions of the masses could not be improved 
without the knowledge of science and technology as developed in the West. He was 
already dreaming of building a bridge to join the East and the West. But the true 
leadership of India would have to spring from the soil of the country. Again and again 
he recalled that Sri Ramakrishna had been a genuine product of the Indian soil, and he 
realized that India would regain her unity and solidarity through the understanding of 
the Master's spiritual experiences. 
Naren again became restless to 'do something', but what, he did not know. He wanted 
to run away from his relatives since he could not bear the sight of their poverty. He 
was eager to forget the world through meditation. During the last part of December 
1889, therefore, he again struck out from the Baranagore monastery and turned his face 
towards Varanasi. 'My idea,' he wrote to a friend, 'is to live in Varanasi for some time 
and to watch how Viswanath and Annapurna deal out my lot. I have resolved either to 
realize my ideal or to lay down my life in the effort — so help me Lord of Varanasi!' 
On his way to Varanasi he heard that Swami Yogananda, one of his brother disciples, 
was lying ill in Allahabad and decided to proceed there immediately. In Allahabad he 
met a Moslem saint, 'every line and curve of whose face showed that he was a 
paramahamsa.' Next he went to Ghazipur and there he came to know the saint Pavhari 
Baba, the 'air-eating holy man.' 
Pavhari Baba was born near Varanasi of brahmin parents. In his youth he had mastered 
many branches of Hindu philosophy. Later he renounced the world, led an austere life, 
practised the disciplines of Yoga and Vedanta, and travelled over the whole of India. 
At last he settled in Ghazipur, where he built an underground hermitage on the bank of 
the Ganga and spent most of his time in meditation. He lived on practically nothing 
and so was given by the people the sobriquet of the 'air-eating holy man'; all were 
impressed by his humility and spirit of service. Once he was bitten by a cobra and said 
while suffering terrible pain, 'Oh, he was a messenger from my Beloved!' Another day, 
a dog ran off with his bread and he followed, praying humbly, 'Please wait, my Lord; 
let me butter the bread for you.' Often he would give away his meagre food to beggars 
or wandering monks, and starve. Pavhari Baba had heard of Sri Ramakrishna, held him 
in high respect as a Divine Incarnation, and kept in his room a photograph of the 
Master. People from far and near visited the Baba, and when not engaged in meditation 
he would talk to them from behind a wall. For several days before his death he 
remained indoors. Then, one day, people noticed smoke issuing from his underground 
cell with the smell of burning flesh. It was discovered that the saint, having come to 
realize the approaching end of his earthly life, had offered his body as the last oblation 
to the Lord, in an act of supreme sacrifice. 
Narendra, at the time of his meeting Pavhari Baba, was suffering from the sever pain of 
lumbago, and this had made it almost impossible for him either to move about or to sit 
in meditation. Further, he was mentally distressed, for he had heard of the illness of 
Abhedananda, another of his brother disciples, who was living at Hrishikesh. 'You 
know not, sir,' he wrote to a friend, 'that I am a very soft-natured man in spite of the 
stern Vedantic views I hold. And this proves to be my undoing. For however I may try 
to think only of my own good, I begin, in spite of myself, to think of other people's 
interests.' Narendra wished to forget the world and his own body through the practice 
of Yoga, and went for instruction to Pavhari Baba, intending to make the saint his 
guru. But the Baba, with characteristic humility, put him off from day to day. 
One night when Naren was lying in bed thinking of Pavhari Baba, Sri Ramakrishna 
appeared to him and stood silently near the door, looking intently into his eyes. The 
vision was repeated for twenty-one days. Narendra understood. He reproached himself 
bitterly for his lack of complete faith in Sri Ramakrishna. Now, at last, he was 
convinced, he wrote to a friend: 'Ramakrishna has no peer. Nowhere else in the world 
exists such unprecedented perfection, such wonderful kindness to all, such intense 
sympathy for men in bondage.' Tearfully he recalled how Sri Ramakrishna had never 
left unfulfilled a single prayer of his, how he had forgiven his offences by the million 
and removed his afflictions. 
But as long as Naren lived he cherished sincere affection and reverence for Pavhari 
Baba, and he remembered particularly two of his instructions. One of these was: 'Live 
in the house of your teacher like a cow,' which emphasizes the spirit of service and 
humility in the relationship between the teacher and the disciple. The second 
instruction of the Baba was: 'Regard spiritual discipline in the same way as you regard 
the goal,'which means that an aspirant should not differentiate between cause and 
effect. 
Narendranath again breathed peace and plunged into meditation. After a few days he 
went to Varanasi, where he learnt of the serious illness of Balaram Bose, one of the 
foremost lay disciples of Sri Ramakrishna. At Ghazipur he had heard that Surendranath 
Mitra, another lay disciple of the Master, was dying. He was overwhelmed with grief, 
and to Pramadadas, who expressed his surprise at the sight of a sannyasin indulging in 
a human emotion, he said: 'Please do not talk that way. We are not dry monks. Do you 
think that because a man has renounced the world he is devoid of all feeling?' 
He came to Calcutta to be at the bedside of Balaram, who passed away on May 13. 
Surendra Mitra died on May 25. But Naren steadied his nerves, and in addition to the 
practice of his own prayer and meditation, devoted himself again to the guidance of his 
brother disciples. Some time during this period he conceived the idea of building a 
permanent temple to preserve the relics of Sri Ramakrishna. 
From his letters and conversations one can gain some idea of the great storm that was 
raging in Naren's soul during this period. He clearly saw to what an extent the educated 
Hindus had come under the spell of the materialistic ideas of the West. He despised 
sterile imitation. But he was also aware of the great ideas that formed the basis of 
European civilization. He told his friends that in India the salvation of the individual 
was the accepted goal, whereas in the West it was the uplift of the people, without 
distinction of caste or creed. Whatever was achieved there was shared by the common 
man; freedom of spirit manifested itself in the common good and in the advancement 
of all men by the united efforts of all. He wanted to introduce this healthy factor into 
the Indian consciousness. 
Yet he was consumed by his own soul's hunger to remain absorbed in samadhi. He felt 
at this time a spiritual unrest like that which he had experienced at the Cossipore 
garden house during the last days of Sri Ramakrishna's earthly existence. The outside 
world had no attraction for him. But another factor, perhaps unknown to him, was 
working within him. Perfect from his birth, he did not need spiritual disciplines for his 
own liberation. Whatever disciplines he practised were for the purpose of removing the 
veil that concealed, for the time being, his true divine nature and mission in the world. 
Even before his birth, the Lord had chosen him as His instrument to help Him in the 
spiritual redemption of humanity. 
Now Naren began to be aware that his life was to be quite different from that of a 
religious recluse: he was to work for the good of the people. Every time he wanted to 
taste for himself the bliss of samadhi, he would hear the piteous moans of the teeming 
millions of India, victims of poverty and ignorance. Must they, Naren asked himself, 
for ever grovel in the dust and live like brutes? Who would be their saviour? 
He began, also, to feel the inner agony of the outwardly happy people of the West, 
whose spiritual vitality was being undermined by the mechanistic and materialistic 
conception of life encouraged by the sudden development of the physical sciences. 
Europe, he saw, was sitting on the crater of a smouldering volcano, and any moment 
Western culture might be shattered by its fiery eruption. The suffering of man, whether 
in the East or in the West, hurt his tender soul. The message of Vedanta, which 
proclaimed the divinity of the soul and the oneness of existence, he began to realize, 
could alone bind up and heal the wounds of India and the world. But what could he, a 
lad of twenty-five, do? The task was gigantic. He talked about it with his brother 
disciples, but received scant encouragement. He was determined to work alone if no 
other help was forthcoming. 
Narendra felt cramped in the monastery at Baranagore and lost interest in its petty 
responsibilities. The whole world now beckoned him to work. Hence, one day in 1890, 
he left the monastery again with the same old determination never to return. He would 
go to the Himalayas and bury himself in the depths of his own thought. To a brother 
disciple he declared, 'I shall not return until I gain such realization that my very touch 
will transform a man.' He prayed to the Holy Mother that he might not return before 
attaining the highest Knowledge, and she blessed him in the name of Sri Ramakrishna. 
Then she asked whether he would not like to take leave of his earthly mother. 'Mother,' 
Naren replied, 'you are my only mother.' 
Accompanied by Swami Akhandananda, Naren left Calcutta and set out for Northern 
India. The two followed the course of the Ganga, their first halting-place being 
Bhagalpur. To one of the people who came to visit him there Naren said that whatever 
of the ancient Aryan knowledge, intellect, and genius remained, could be found mostly 
in those parts of the country that lay near the banks of the Ganga. The farther one 
departed from the river, the less one saw of that culture. This fact, he believed, 
explained the greatness of the Ganga as sung in the Hindu scriptures. He further 
observed: 'The epithet "mild Hindu" instead of being a word of reproach, ought really 
to point to our glory, as expressing greatness of character. For see how much moral and 
spiritual advancement and how much development of the qualities of love and 
compassion have to be acquired before one can get rid of the brutish force of one's 
nature, which impels a man to slaughter his brother men for self-aggrandizement!' 
He spent a few days in Varanasi and left the city with the prophetic words: 'When I 
return here the next time, I shall burst upon society like a bomb-shell, and it will follow 
me like a dog.' 
After visiting one or two places, Naren and Akhandananda arrived at Nainital, their 
destination being the sacred Badrikashrama, in the heart of the Himalayas. They 
decided to travel the whole way on foot, and also not to touch money. Near Almora 
under an old peepul tree by the side of a stream, they spent many hours in meditation. 
Naren had a deep spiritual experience, which he thus jotted down in his note-book:
In the beginning was the Word, etc. 
The microcosm and the macrocosm are built on the same plan. Just as the individual 
soul is encased in a living body, so is the Universal Soul, in the living prakriti (nature), 
the objective universe. Kali is embracing Siva. This is not a fancy. This covering of the 
one (Soul) by the other (nature) is analogous to the relation between an idea and the 
word expressing it. They are one and the same, and it is only by a mental abstraction 
that one can distinguish them. Thought is impossible without words. Therefore in the 
beginning was the Word, etc. 
This dual aspect of the Universal Soul is eternal. So what we perceive or feel is the 
combination of the Eternally Formed and the Eternally Formless.
Thus Naren realized, in the depths of meditation, the oneness of the universe and man, 
who is a universe in miniature. He realized that, all that exists in the universe also 
exists in the body, and further, that the whole universe exists in the atom. 
Several other brother disciples joined Naren. But they could not go to Badrikashrama 
since the road was closed by Government order on account of famine. They visited 
different holy places, lived on alms, studied the scriptures, and meditated. At this time, 
the sad news arrived of the suicide of one of Naren's sisters under tragic conditions, 
and reflecting on the plight of Hindu women in the cruel present-day society, he 
thought that he would be a criminal if he remained an indifferent spectator of such 
social injustice. 
Naren proceeded to Hrishikesh, a beautiful valley at the foot of the Himalayas, which 
is surrounded by hills and almost encircled by the Ganga. From an immemorial past 
this sacred spot has been frequented by monks and ascetics. After a few days, however, 
Naren fell seriously ill and his friends despaired of his life. When he was convalescent 
he was removed to Meerut. There he met a number of his brother disciples and 
together they pursued the study of the scriptures, practised prayer and meditation, and 
sang devotional songs, creating in Meerut a miniature Baranagore monastery. 
After a stay of five months Naren became restless, hankering again for his wandering 
life; but he desired to be alone this time and break the chain of attachment to his 
brother disciples. He wanted to reflect deeply about his future course of action, of 
which now and then he was getting glimpses. From his wanderings in the Himalayas 
he had become convinced that the Divine Spirit would not allow him to seal himself 
within the four walls of a cave. Every time he had thought to do so, he had been 
thrown out, as it were, by a powerful force. The degradation of the Indian masses and 
the spiritual sickness of people everywhere were summoning him to a new line of 
action, whose outer shape was not yet quite clear to him. 
In the later part of January 1891, Naren bade farewell to his brother disciples and set 
out for Delhi, assuming the name of Swami Vividishananda. He wished to travel 
without being recognized. He wanted the dust of India to cover up his footprints. It was 
his desire to remain an unknown sannyasin, among the thousands of others seen in the 
country's thoroughfares, market-places, deserts, forests, and caves. But the fires of the 
Spirit that burnt in his eyes, and his aristocratic bearing, marked him as a prince among 
men despite all his disguises. 
In Delhi, Naren visited the palaces, mosques, and tombs. All around the modern city he 
saw a vast ruin of extinct empires dating from the prehistoric days of the Mahabharata, 
revealing the transitoriness of material achievements. But gay and lively Delhi also 
revealed to him the deathless nature of the Hindu spirit. 
Some of his brother disciples from Meerut came to the city and accidentally discovered 
their beloved leader. Naren was angry. He said to them: 'Brethren I told you that I 
desired to be left alone. I asked you not to follow me. This I repeat once more. I must 
not be followed. I shall presently leave Delhi. No one must try to know my 
whereabouts. I shall sever all old associations. Wherever the Spirit leads, there I shall 
wander. It matters not whether I wander about in a forest or in a desert, on a lonely 
mountain or in a populous city. I am off. Let everyone strive to realize his goal 
according to his lights.' 
Narendra proceeded towards historic Rajputana, repeating the words of the Suttanipata:
Go forward without a path, 
Fearing nothing, caring for nothing, 
Wandering alone, like the rhinoceros! 
Even as a lion, not trembling at noises, 
Even as the wind, not caught in a net, 
Even as the lotus leaf, untainted by water, 
Do thou wander alone, like the rhinoceros!
Several factors have been pointed out as influencing Naren's life and giving shape to 
his future message: the holy association of Sri Ramakrishna, his own knowledge of 
Eastern and Western cultures, and his spiritual experiences. To these another must be 
added: the understanding of India gained through his wanderings. This new 
understanding constituted a unique education for Naren. Here, the great book of life 
taught him more than the printed words of the libraries. 
He mixed with all — today sleeping with pariahs in their huts and tomorrow 
conversing on equal terms with Maharajas, Prime Ministers, orthodox pandits, and 
liberal college professors. Thus he was brought into contact with their joys and 
sorrows, hopes and frustrations. He witnessed the tragedy of present-day India and also 
reflected on its remedy. The cry of the people of India, the God struggling in humanity, 
and the anxiety of men everywhere to grasp a hand for aid, moved him deeply. In the 
course of his travels Naren came to know how he could make himself a channel of the 
Divine Spirit in the service of mankind. 
During these wandering days he both learnt and taught. The Hindus he asked to go 
back to the eternal truths of their religion, hearken to the message of the Upanishads, 
respect temples and religious symbols, and take pride in their birth in the holy land of 
India. He wanted them to avoid both the outmoded orthodoxy still advocated by 
fanatical leaders, and the misguided rationalism of the Westernized reformers. He was 
struck by the essential cultural unity of India in spite of the endless diversity of form. 
And the people who came to know him saw in him the conscience of India, her unity, 
and her destiny. 
As already noted, Narendranath while travelling in India often changed his name to 
avoid recognition. It will not be improper to call him, from this point of his life, by the 
monastic title of 'Swami,' or the more affectionate and respectful appellation of 
'Swamiji.' 
In Alwar, where Swamiji arrived one morning in the beginning of February 1891, he 
was cordially received by Hindus and Moslems alike. To a Moslem scholar he said: 
'There is one thing very remarkable about the Koran. Even to this day it exists as it was 
recorded eleven hundred years ago. The book has retained its original purity and is free 
from interpolation.' 
He had a sharp exchange of words with the Maharaja, who was Westernized in his 
outlook. To the latter's question as to why the Swami, an able-bodied young man and 
evidently a scholar, was leading a vagabond's life, the Swami retorted, 'Tell me why 
you constantly spend your time in the company of Westerners and go out on shooting 
excursions, neglecting your royal duties.' The Maharaja said, 'I cannot say why, but, no 
doubt, because I like to.' 'Well,' the Swami exclaimed, 'for that very reason I wander 
about as a monk.' 
Next, the Maharaja ridiculed the worship of images, which to him were nothing but 
figures of stone, clay, or metal. The Swami tried in vain to explain to him that Hindus 
worshipped God alone, using the images as symbols. The Prince was not convinced. 
Thereupon the Swami asked the Prime Minister to take down a picture of the 
Maharaja, hanging on the wall, and spit on it. Everyone present was horror-struck at 
this effrontery. The Swami turned to the Prince and said that though the picture was 
not the Maharaja himself, in flesh and blood, yet it reminded everyone of his person 
and thus was held in high respect; likewise the image brought to the devotee's mind the 
presence of the Deity and was therefore helpful for concentration, especially at the 
beginning of his spiritual life. The Maharaja apologized to Swamiji for his rudeness. 
The Swami exhorted the people of Alwar to study the eternal truths of Hinduism, 
especially to cultivate the knowledge of Sanskrit, side by side with Western science. 
He also encouraged them to read Indian history, which he remarked should be written 
by Indians following the scientific method of the West. European historians dwelt 
mainly on the decadent period of Indian culture. 
In Jaipur the Swami devoted himself to the study of Sanskrit grammar, and in Ajmer 
he recalled the magnificence of the Hindu and Moslem rules. At Mount Abu he gazed 
in wonder at the Jain temple of Dilwara, which it has been said, was begun by titans 
and finished by jewellers. There he accepted the hospitality of a Moslem official. To 
his scandalized Hindu friends the Swami said that he was, as a sannyasin belonging to 
the highest order of paramahamsas, above all rules of caste. His conduct in dining with 
Moslems, he further said, was not in conflict with the teachings of the scriptures, 
though it might be frowned upon by the narrow-minded leaders of Hindu society. 
At Mount Abu the Swami met the Maharaja of Khetri, who later became one of his 
devoted disciples. The latter asked the Swami for the boon of a male heir and obtained 
his blessing. 
Next we see the Swami travelling in Gujarat and Kathiawar in Western India. In 
Ahmedabad he refreshed his knowledge of Jainism. Kathiawar, containing a large 
number of places sacred both to the Hindus and the to Jains, was mostly ruled by 
Hindu Maharaja, who received the Swami with respect. To Babu Haridas Viharidas, 
the Prime Minister of the Moslem state of Junagad, he emphasized the need of 
preaching the message of Hinduism throughout the world. He spent eleven months in 
Porbandar and especially enjoyed the company of the Prime Minister, Pandit Sankar 
Pandurang, a great Sanskrit scholar who was engaged in the translation of the Vedas. 
Impressed by the Swami's intellectuality and originality, the pandit said: 'Swamiji, I am 
afraid you cannot do much in this country. Few will appreciate you here. You ought to 
go to the West, where people will understand you and your work. Surely you can give 
to the Western people your enlightening interpretation of Hinduism.' 
The Swami was pleased to hear these words, which coincided with something he had 
been feeling within. The Prime Minister encouraged the Swami to continue his study 
of the French language since it might be useful to him in his future work. 
During this period the Swami was extremely restless. He felt within him a boundless 
energy seeking channels for expression. The regeneration of India was uppermost in 
his mind. A reawakened India could, in her turn, help the world at large. The sight of 
the pettiness, jealousy, disunion, ignorance, and poverty among the Hindus filled his 
mind with great anguish. But he had no patience with the Westernized reformers, who 
had lost their contact with the soul of the country. He thoroughly disapproved of their 
method of social, religious, and political reform through imitation of the West. He 
wanted the Hindus to cultivate self-confidence. Appreciation of India's spiritual culture 
by the prosperous and powerful West, he thought, might give the Hindus confidence in 
their own heritage. He prayed to the Lord for guidance. He became friendly with the 
Hindu Maharajas who ruled over one-fifth of the country and whose influence was 
great over millions of people. Through them he wanted to introduce social reforms, 
improved methods of education, and other measures for the physical and cultural 
benefit of the people. The Swami felt that in this way his dream of India's regeneration 
would be realized with comparative ease. 
After spending a few days in Baroda, the Swami came to Khandwa in Central India. 
Here he dropped the first hint of his willingness to participate in the Parliament of 
Religions to be held shortly in Chicago. He had heard of this Parliament either in 
Junagad or Porbandar. 
After visiting Bombay, Poona, and Kolhapur, the Swami arrived at Belgaum. In 
Bombay he had accidentally met Swami Abhedananda and in the course of a talk had 
said to him, 'Brother, such a great power has grown within me that sometimes I feel 
that my whole body will burst.' 
All through this wandering life he exchanged ideas with people in all stations and 
stages of life and impressed everyone with his earnestness, eloquence, gentleness, and 
vast knowledge of India and Western culture. Many of the ideas he expressed at this 
time were later repeated in his public lectures in America and India. But the thought 
nearest to his heart concerned the poor and ignorant villagers, victims of social 
injustice: how to improve the sanitary condition of the villages, introduce scientific 
methods of agriculture, and procure pure water for daily drinking; how to free the 
peasants from their illiteracy and ignorance, how to give back to them their lost 
confidence. Problems like these tormented him day and night. He remembered vividly 
the words of Sri Ramakrishna that religion was not meant for 'empty stomachs.' 
To his hypochondriac disciple Haripada he gave the following sound advice: 'What is 
the use of thinking always of disease? Keep cheerful, lead a religious life, cherish 
elevating thoughts, be merry, but never indulge in pleasures which tax the body or for 
which you will feel remorse afterwards; then all will be well. And as regards death, 
what does it matter if people like you and me die? That will not make the earth deviate 
from its axis! We should not consider ourselves so important as to think that the world 
cannot move on without us.' 
When he mentioned to Haripada his desire to proceed to America, the disciple was 
delighted and wanted to raise money for the purpose, but the Swami said to him that he 
would not think about it until after making his pilgrimage to Rameswaram and 
worshipping the Deity there. 
From Belgaum the Swami went to Bangalore in the State of Mysore, which was ruled 
by a Hindu Maharaja. The Maharaja's Prime Minister described the young monk as 'a 
majestic personality and a divine force destined to leave his mark on the history of his 
country.' The Maharaja, too, was impressed by his 'brilliance of thought, charm of 
character, wide learning, and penetrating religious insight.' He kept the Swami as his 
guest in the palace. 
One day, in front of his high officials, the Maharaja asked the Swami, 'Swamiji, what 
do you think of my courtiers?' 
'Well,' came the bold reply, 'I think Your Highness has a very good heart, but you are 
unfortunately surrounded by courtiers who are generally flatterers. Courtiers are the 
same everywhere.' 
'But,' the Maharaja protested, 'my Prime Minster is not such. He is intelligent and 
trustworthy.' 
'But, Your Highness, Prime Minister is "one who robs the Maharaja and pays the 
Political Agent."' 
The Prince changed the subject and afterwards warned the Swami to be more discreet 
in expressing his opinion of the officials in a Native State; otherwise those 
unscrupulous people might even poison him. But the Swami burst out: 'What! Do you 
think an honest sannyasin is afraid of speaking the truth, even though it may cost him 
his very life? Suppose your own son asks me about my opinion of yourself; do you 
think I shall attribute to you all sorts of virtues which I am quite sure you do not 
possess? I can never tell a lie.' 
The Swami addressed a meeting of Sanskrit scholars and gained their applause for his 
knowledge of Vedanta. He surprised an Austrian musician at the Prince's court with his 
knowledge of Western music. He discussed with the Maharaja his plan of going to 
America, but when the latter came forward with an offer to pay his expenses for the 
trip, he declined to make a final decision before visiting Rameswaram. Perhaps he was 
not yet quite sure of God's will in the matter. When pressed by the Maharaja and the 
Prime Minister to accept some gifts, the costlier the better, the Swami took a tobacco 
pipe from the one and a cigar from the other. 
Now the Swami turned his steps towards picturesque Malabar. At Trivandrum, the 
capital of Travancore, he moved in the company of college professors, state officials, 
and in general among the educated people of the city. They found him equally at ease 
whether discussing Spencer or Sankaracharya, Shakespeare or Kalidasa, Darwin or 
Patanjali, Jewish history or Aryan civilization. He pointed out to them the limitations 
of the physical sciences and the failure of Western psychology to understand the 
superconscious aspect of human nature. 
Orthodox brahmins regarded with abhorrence the habit of eating animal food. The 
Swami courageously told them about the eating of beef by the brahmins in Vedic 
times. One day, asked about what he considered the most glorious period of Indian 
history, the Swami mentioned the Vedic period, when 'five brahmins used to polish off 
one cow.' He advocated animal food for the Hindus if they were to cope at all with the 
rest of the world in the present reign of power and find a place among the other great 
nations, whether within or outside the British Empire. 
An educated person of Travancore said about him: 'Sublimity and simplicity were 
written boldly on his features. A clean heart, a pure and austere life, an open mind, a 
liberal spirit, wide outlook, and broad sympathy were the outstanding characteristics of 
the Swami.' 
From Trivandrum the Swami went to Kanyakumari (Cape Comorin), which is the 
southernmost tip of India and from there he moved up to Rameswaram. At 
Rameswaram the Swami met Bhaskara Setupati, the Raja of Ramnad, who later 
became one of his ardent disciples. He discussed with the Prince many of his ideas 
regarding the education of the Indian masses and the improvement of their agricultural 
conditions. The Raja urged the Swami to represent India at the Parliament of Religions 
in Chicago and promised to help him in his venture.
"""

context = st.text_area("select a context",select_context)
options = st.multiselect(
     'Select question from Below List',
     ['Who are you?', 'Where am I going to?', "What's the purpose of Life ?", 'What do others mean to me?',"Where does Narendra's family lived?"])

st.markdown("OR")
question = st.text_input("Write a question of your choice here: Example: Who is Ramakrishna",options)

if context:
    # Execute question against paragraph
    if question:
        outputs = QAPipeline(question = question,context = context,topk = 3, max_seq_len = 512)
        answer = outputs[0]["answer"]
        output_answer = st.text_area("Answer",answer)
