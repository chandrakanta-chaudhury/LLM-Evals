#Zero shot  -GLiNer -SPACY
import spacy
custom_spacy_config= {"gliner_model": "urchade/gliner_multi",
                      "chunk_size":250,
                      "labels":['people','company'],
                      "style": "ent" }

nlp=spacy.blank("en")
nlp.add_pipe("gliner_spacy",config=custom_spacy_config)

text="This is a testing text on Bill Gates and Microsoft"
doc=nlp(text)
for ent in doc.ents:
    print(ent.text,ent.label_,ent._.score)

