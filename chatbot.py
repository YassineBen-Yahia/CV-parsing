import openai
import spacy



openai.api_key = key
def advise_cv(entities):
    prompt = (
        "You are a career coach. Here is a CV summary:\n"
        f"{entities}\n"
        "Suggest 3 key skills to improve."
    )
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"Be concise and actionable."},
                  {"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content

sample_entities = [
    {"label": "SKILL", "text": "Python"},
    {"label": "EDUCATION", "text": "BSc Computer Science"},
    {"label": "EXPERIENCE", "text": "2 years data analysis"}
]

print(advise_cv(sample_entities))


"""# Load your spaCy model (update path as needed)
nlp = spacy.load(r"C:\ML\CV-Parsing\transformer\model-best")

def parse_text_with_spacy(text):
    doc = nlp(text)
    entities = [{"label": ent.label_, "text": ent.text} for ent in doc.ents]
    return entities

# Input text from user
txt = input("Paste your CV text here: ")
parsed_entities = parse_text_with_spacy(txt)

# Pass parsed entities to chatbot
print(advise_cv(parsed_entities))"""
