import json

# Input and output file paths
json_path = "scFMs/scGPT/.cache/helical/models/scgpt/scGPT_CP/vocab.json"
txt_path = "scgpt_vocab.txt"

# Load JSON
with open(json_path, "r") as f:
    vocab = json.load(f)

# Write gene symbols (keys) to text file
with open(txt_path, "w") as f_out:
    for gene in vocab:
        f_out.write(f"{gene}\n")

print("Done! Vocabulary written to scgpt_vocab.txt")
