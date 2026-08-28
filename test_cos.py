from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

claim = "Marie Curie was a Polish-French physicist and chemist."
snippets = [
    "journalist and pianist. Ève Curie was the younger daughter of Marie Skłodowska-Curie and Pierre Curie. Her sister was Irène Joliot-Curie and her brother-in-law",
    "1970 [Ph.] Louis de Broglie (DSc) – 1929 [Ph.] [Ch.] Marie Curie (DSc) – 1903, 1911 [Ph.] Pierre Curie (DSc) – 1903 [Ph.] Pierre-Gilles de Gennes (DSc) –",
    "Master's degree in astrophysics and space techniques at the Pierre and Marie Curie University (UPMC), Paris 6, in 1986 and a master's degree in English"
]

claim_emb = model.encode(claim, convert_to_tensor=True)
snip_embs = model.encode(snippets, convert_to_tensor=True)

cos_scores = util.cos_sim(claim_emb, snip_embs)[0]
print("Cosine scores:", cos_scores)
