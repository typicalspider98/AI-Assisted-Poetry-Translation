# Prompt 0

## 中文

```
你是一个专业的翻译辅助系统，专门负责给出prompt，用于指导大模型生成符合新西兰地区英语使用习惯的翻译。
用户需要翻译的内容是：
《静夜思》
床前明月光，疑是地上霜。举头望明月，低头思故乡。
请你直接给出一个中文prompt，用于指导大模型进行上述中文诗歌的英文翻译工作。
生成的中文prompt要求（生成结果不要带有“prompt”字样）：
1.prompt第一句话应提出将所给中文诗歌翻译成英文诗歌这个要求，并附上中文原诗。
2.prompt应参考用户给出的中文诗歌的具体内容，选择恰当的语言描述本首诗歌在翻译过程中应注意的事项。（例如：尽量传达原诗的核心思想、情感和主旨，避免曲解或过度延伸；保留诗歌中的文化意象，并选择合适的英文表达；语言要有韵味，保持诗性表达，如节奏、押韵（若能做到）、简练、修辞等；对一些特有的文化典故、历史人物或风俗需做注解、意译或文化转化，避免读者误解；根据具体诗句灵活取舍，重意境时意译，重结构或修辞时偏直译）
3.prompt应指出在翻译时要体现新西兰地区英语的语言风格和文化特点。（关于什么是新西兰英语，我指的是下面可能会提供给你的一部分关键词，这些关键词取自由一本新西兰英语词典生成的向量数据库。在翻译本诗的过程中，有些英文单词在新西兰英语中有特殊的表达，我们将这些特殊的使用情况从向量数据库中取出来，以关键词及例句的形式展示，你在生成prompt时可以参考这些词的使用场景，看是否可以将其融入你的诗歌翻译中。要尽可能保留诗歌的原有含义及意境，同时兼具新西兰英语的表达特色。）

```

## 英文

demo1

```
You are a professional translation support system, specifically designed to generate prompts that guide large language models in producing translations that align with the usage and stylistic features of New Zealand English.

The user needs to translate the following text:
《静夜思》
床前明月光，疑是地上霜。举头望明月，低头思故乡。


Please directly generate a prompt that instructs the model to translate the above Chinese poem into English poetry.

The requirements for the generated prompt are as follows (do not include the word **Prompt:**   in the output):
The prompt must be written in English. The first sentence should clearly instruct the model to translate the provided Chinese poem into an English poem, and it must include the original Chinese text.
The prompt should take into account the specific content of the poem and describe appropriate considerations for the translation process. For example:

Strive to convey the core ideas, emotions, and main message of the original poem, avoiding distortion or excessive elaboration.

Preserve the cultural imagery within the poem and choose suitable English expressions.

Maintain poetic qualities such as rhythm, rhyme (if feasible), conciseness, and literary devices.

When dealing with specific cultural references, historical figures, or customs, include annotation, cultural adaptation, or interpretive translation to prevent misunderstanding.

Adapt the translation approach flexibly according to each line—favoring interpretive translation for imagery-rich lines, and more direct translation for structurally or rhetorically important lines.

The prompt should also emphasize the need to reflect the linguistic style and cultural characteristics of New Zealand English in the translation.
(Regarding what New Zealand English entails, the user may provide a list of keywords derived from a vector database built from a New Zealand English dictionary. In some cases, certain English words may have unique usages in New Zealand English. We extract these usage cases from the database and present them as keywords with sample sentences. When generating the prompt, you may refer to these usage contexts to see whether they can be incorporated into the poem's translation. The goal is to retain the original poem's meaning and imagery as much as possible, while also capturing the distinctive qualities of New Zealand English.)
```

demo2

```
```

## keywords

> 你是一位精通中文古典诗歌与英语文化的翻译顾问。
> 请根据下列诗歌内容及长度提取适量的关键词，用于指导英文翻译。
> 要求:
> 1.不要逐句照搬原诗句，结合诗歌原意和意境，提取其中可用于翻译的核心主题词，包括动词、名词、形容词、副词等，以及意象和文化概念。
> 2.在不破坏诗歌本意和意境的前提下，提取的主题词是按中文诗歌韵律和语义停顿划分出的最小语素和词语，特殊情况下为不破坏诗歌整体含义也可以是中文短语。
> 3.主题词包括所有意象，并最终以准确简明的英文单词展示。
> 4.并返回 JSON 格式，关键词应具有翻译价值与文化象征性。
> JSON 示例格式如下：
> {
>   "keywords": [
>     "word1: English simple explanation",
>     "word2: English simple explanation",
>     "word3: English simple explanation"
>   ]
> }
>
> 诗歌原文：《静夜思》
> 床前明月光，疑是地上霜。举头望明月，低头思故乡。

```
你是一位精通中文古典诗歌与英语文化的翻译顾问。
请根据下列诗歌内容及长度提取适量的关键词，用于指导英文翻译。
要求:
1.不要逐句照搬原诗句，结合诗歌原意和意境，提取其中可用于翻译的核心主题词，包括动词、名词、形容词、副词等，以及意象和文化概念。
2.在不破坏诗歌本意和意境的前提下，提取的主题词是按中文诗歌韵律和语义停顿划分出的最小语素和词语，特殊情况下为不破坏诗歌整体含义也可以是中文短语。
3.主题词包括所有意象，并最终以准确简明的英文单词展示。
4.并返回 JSON 格式，关键词应具有翻译价值与文化象征性。
JSON 示例格式如下：
{
  "keywords": [
    "word1: English simple explanation",
    "word2: English simple explanation",
    "word3: English simple explanation"
  ]
}

诗歌原文：《静夜思》
床前明月光，疑是地上霜。举头望明月，低头思故乡。
```

```
You are a translation advisor well-versed in classical Chinese poetry and English cultural expression.
Based on the following poem and its length, extract a suitable set of thematic keywords to guide the English translation process.

Requirements:
Do not simply extract phrases directly from each line of the original poem. Instead, analyze the overall meaning and imagery to identify core thematic words that can assist in capturing the essence of the poem in translation. These may include verbs, nouns, adjectives, adverbs, as well as symbolic imagery and culturally relevant concepts.

Extract keywords according to natural pauses and semantic breaks in the poem’s structure, following the rhythm and meaning of the original Chinese. In special cases, short Chinese phrases may be accepted as a single unit if breaking them apart would compromise the poem’s overall meaning or imagery.

Include all key symbols and imagery from the poem, and present the extracted thematic keywords as accurate, concise English words.

Return the results in JSON format. The selected keywords should carry both translational value and cultural symbolism.

Expected JSON format:
{
  "keywords": [
    "word1: English simple explanation",
    "word2: English simple explanation",
    "word3: English simple explanation"
  ]
}
Original Poem:

```

---

# tmp=2

```
(deepseek-r1) root@VM-0-8-ubuntu:/workspace/AI-Assisted-Poetry-Translation/C2NZE/demo_v1.6_keywords-recommend# python test_poem_reranker_v3.py 
✅ Redis 数据库 0 已连接
✅ Redis 数据库 2 已连接

🔍 Keyword: 月
/root/miniforge3/envs/deepseek-r1/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
📘 TopK retrieved (by vector similarity):
1. marama-0                       | CosSim: 0.7553 | Content: marama: n.the moon
2. bum-cheek salute-0             | CosSim: 0.7146 | Content: bum-cheek salute: n.mooning
3. Paddy's lantern-0              | CosSim: 0.7090 | Content: Paddy's lantern: n.(humorous) the moon [ca1935 Havelock]
4. gipsy sun-0                    | CosSim: 0.6971 | Content: gipsy sun: n.The moon; cf Paddy's lantern [ca1914-1918 C.R. Carr: used on Gallipoli (? translation of a Polish term)]
5. come in-3                      | CosSim: 0.6793 | Content: come in: 3v.to have one's monthly period (female)
📊 Reranker similarity scores (standard semantic input):
You're using a XLMRobertaTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
1. [月] → marama: n.the moon                                 | RerankScore: 0.3922
2. [月] → bum-cheek salute: n.mooning                        | RerankScore: 0.0082
3. [月] → Paddy's lantern: n.(humorous) the moon [ca1935 Havelock] | RerankScore: 0.7144
4. [月] → gipsy sun: n.The moon; cf Paddy's lantern [ca1914-1918 C.R. Carr: used on Gallipoli (? translation of a Polish term)] | RerankScore: 0.0276
5. [月] → come in: 3v.to have one's monthly period (female)  | RerankScore: 0.0409

🔍 Keyword: moon
📘 TopK retrieved (by vector similarity):
1. marama-0                       | CosSim: 0.8361 | Content: marama: n.the moon
2. bum-cheek salute-0             | CosSim: 0.8113 | Content: bum-cheek salute: n.mooning
3. opah-0                         | CosSim: 0.7578 | Content: opah: n.moonfish
4. Paddy's lantern-0              | CosSim: 0.7421 | Content: Paddy's lantern: n.(humorous) the moon [ca1935 Havelock]
5. burrow-0                       | CosSim: 0.7157 | Content: burrow: n.of a muttonbird
📊 Reranker similarity scores (standard semantic input):
1. [moon] → marama: n.the moon                                 | RerankScore: 0.6289
2. [moon] → bum-cheek salute: n.mooning                        | RerankScore: 0.0034
3. [moon] → opah: n.moonfish                                   | RerankScore: 0.0104
4. [moon] → Paddy's lantern: n.(humorous) the moon [ca1935 Havelock] | RerankScore: 0.6615
5. [moon] → burrow: n.of a muttonbird                          | RerankScore: 0.0000

🔍 Keyword: light
📘 TopK retrieved (by vector similarity):
1. vampire light-0                | CosSim: 0.7289 | Content: vampire light: n.a stand-by light
2. lighthouse fish-0              | CosSim: 0.7269 | Content: lighthouse fish: n.(See?lightfish)
3. powdering-0                    | CosSim: 0.7081 | Content: powdering: n.a light snowfall
4. blinkie-0                      | CosSim: 0.6976 | Content: blinkie: n.a small star with flashing light
5. moit-0                         | CosSim: 0.6876 | Content: moit: n.light vegetable matter contamination
📊 Reranker similarity scores (standard semantic input):
1. [light] → vampire light: n.a stand-by light                  | RerankScore: 0.0002
2. [light] → lighthouse fish: n.(See?lightfish)                 | RerankScore: 0.0003
3. [light] → powdering: n.a light snowfall                      | RerankScore: 0.0000
4. [light] → blinkie: n.a small star with flashing light        | RerankScore: 0.0104
5. [light] → moit: n.light vegetable matter contamination       | RerankScore: 0.0000

🔍 Keyword: bed
📘 TopK retrieved (by vector similarity):
1. cart-1                         | CosSim: 0.8125 | Content: cart: n.a bed
2. power board-0                  | CosSim: 0.7751 | Content: power board: n.board at bottom of bed
3. pit-0                          | CosSim: 0.7568 | Content: pit: n.a sleeping bag or bed
4. flako-0                        | CosSim: 0.7543 | Content: flako: adj.asleep
5. bunky-0                        | CosSim: 0.7428 | Content: bunky: n.bunk, i.e. bed
📊 Reranker similarity scores (standard semantic input):
1. [bed] → cart: n.a bed                                      | RerankScore: 0.0088
2. [bed] → power board: n.board at bottom of bed              | RerankScore: 0.0005
3. [bed] → pit: n.a sleeping bag or bed                       | RerankScore: 0.0022
4. [bed] → flako: adj.asleep                                  | RerankScore: 0.0857
5. [bed] → bunky: n.bunk, i.e. bed                            | RerankScore: 0.3439

🔍 Keyword: hometown
📘 TopK retrieved (by vector similarity):
1. civilisation-1                 | CosSim: 0.7269 | Content: civilisation: 2n.the town when perceived from the country.
2. smarty-pie-0                   | CosSim: 0.7213 | Content: smarty-pie: 
3. scambuster-0                   | CosSim: 0.7137 | Content: scambuster: 
4. liquid heroin-0                | CosSim: 0.7109 | Content: liquid heroin: n.homebake
5. bake-1                         | CosSim: 0.7106 | Content: bake: v.homebake
📊 Reranker similarity scores (standard semantic input):
1. [hometown] → civilisation: 2n.the town when perceived from the country. | RerankScore: 0.0006
2. [hometown] → smarty-pie:                                        | RerankScore: 0.0000
3. [hometown] → scambuster:                                        | RerankScore: 0.0000
4. [hometown] → liquid heroin: n.homebake                          | RerankScore: 0.0000
5. [hometown] → bake: v.homebake                                   | RerankScore: 0.0000

🔍 Keyword: sorrow
📘 TopK retrieved (by vector similarity):
1. pouri-1                        | CosSim: 0.7253 | Content: pouri: n.sadness
2. pack a sad-0                   | CosSim: 0.6895 | Content: pack a sad: phr.to be in a bad or depressed mood
3. broken-down swell-1            | CosSim: 0.6837 | Content: broken-down swell: 2n.an emotionally distressed person
4. tangi-6                        | CosSim: 0.6822 | Content: tangi: n.a mourning ceremony; lament. [also tangie, taki]
5. happy as a sick eel on a sand spit-0 | CosSim: 0.6820 | Content: happy as a sick eel on a sand spit: phr.very unhappy
📊 Reranker similarity scores (standard semantic input):
1. [sorrow] → pouri: n.sadness                                   | RerankScore: 0.9842
2. [sorrow] → pack a sad: phr.to be in a bad or depressed mood   | RerankScore: 0.5321
3. [sorrow] → broken-down swell: 2n.an emotionally distressed person | RerankScore: 0.0404
4. [sorrow] → tangi: n.a mourning ceremony; lament. [also tangie, taki] | RerankScore: 0.0103
5. [sorrow] → happy as a sick eel on a sand spit: phr.very unhappy | RerankScore: 0.0000

🔍 Keyword: frost
📘 TopK retrieved (by vector similarity):
1. frostie-0                      | CosSim: 0.7801 | Content: frostie: n.a frostfish
2. hiku-0                         | CosSim: 0.7224 | Content: hiku: n.frostfish
3. taharangi-0                    | CosSim: 0.7224 | Content: taharangi: n.frostfish
4. thirty degrees below freezo-0  | CosSim: 0.7035 | Content: thirty degrees below freezo: phr.very cold
5. pull and tie weasand-0         | CosSim: 0.6947 | Content: pull and tie weasand: phr.freezing-worker
📊 Reranker similarity scores (standard semantic input):
1. [frost] → frostie: n.a frostfish                             | RerankScore: 0.0031
2. [frost] → hiku: n.frostfish                                  | RerankScore: 0.0035
3. [frost] → taharangi: n.frostfish                             | RerankScore: 0.0019
4. [frost] → thirty degrees below freezo: phr.very cold         | RerankScore: 0.5196
5. [frost] → pull and tie weasand: phr.freezing-worker          | RerankScore: 0.0001

🔍 Keyword: ground
📘 TopK retrieved (by vector similarity):
1. lea-ground-0                   | CosSim: 0.7448 | Content: lea-ground: n.sheltered ground
2. improved-0                     | CosSim: 0.7310 | Content: improved: adj.Of land
3. heavy-1                        | CosSim: 0.7109 | Content: heavy: adj.(of land)
4. property-0                     | CosSim: 0.7088 | Content: property: n.farm, land
5. cap-2                          | CosSim: 0.6967 | Content: cap: n.stratum of soil
📊 Reranker similarity scores (standard semantic input):
1. [ground] → lea-ground: n.sheltered ground                     | RerankScore: 0.0561
2. [ground] → improved: adj.Of land                              | RerankScore: 0.0031
3. [ground] → heavy: adj.(of land)                               | RerankScore: 0.0057
4. [ground] → property: n.farm, land                             | RerankScore: 0.2982
5. [ground] → cap: n.stratum of soil                             | RerankScore: 0.0172

🔍 Keyword: 望
📘 TopK retrieved (by vector similarity):
1. meanies-0                      | CosSim: 0.7185 | Content: meanies: n.a mean look
2. peg-3                          | CosSim: 0.7091 | Content: peg: v.notice, watch, survey
3. geezer-0                       | CosSim: 0.7085 | Content: geezer: n.a look
4. geek-0                         | CosSim: 0.7085 | Content: geek: n.a look
5. goop-0                         | CosSim: 0.7074 | Content: goop: v.to stare, gawp
📊 Reranker similarity scores (standard semantic input):
1. [望] → meanies: n.a mean look                             | RerankScore: 0.1088
2. [望] → peg: v.notice, watch, survey                       | RerankScore: 0.5061
3. [望] → geezer: n.a look                                   | RerankScore: 0.0008
4. [望] → geek: n.a look                                     | RerankScore: 0.0007
5. [望] → goop: v.to stare, gawp                             | RerankScore: 0.9169
```

