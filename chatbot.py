
import openai



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
