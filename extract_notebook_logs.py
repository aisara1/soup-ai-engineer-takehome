import json, re, pathlib
nb_path = pathlib.Path("takehome_final.ipynb")
out = pathlib.Path("logs/notebook_extract")
out.mkdir(parents=True, exist_ok=True)
nb = json.loads(nb_path.read_text(encoding="utf-8"))
ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
index = ["Extracted verbatim from takehome_final.ipynb cell outputs.",
         "These are copies of preserved notebook outputs, not regenerated runs.", ""]
def text(v):
    return "".join(v) if isinstance(v, list) else ("" if v is None else str(v))
count = 0
for i, c in enumerate(nb.get("cells", [])):
    if c.get("cell_type") != "code" or not c.get("outputs"):
        continue
    src = text(c.get("source", ""))
    parts = []
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append(text(o.get("text", "")))
        elif o.get("output_type") == "error":
            parts.append("ERROR: " + str(o.get("ename", "")) + ": " + str(o.get("evalue", ""))[:500])
        elif "data" in o and "text/plain" in o["data"]:
            parts.append(text(o["data"]["text/plain"]))
    body = ansi.sub("", "".join(parts))
    name = f"cell_{i:03d}.txt"
    (out / name).write_text(f"# SOURCE\n{src}\n\n# OUTPUT\n{body}", encoding="utf-8")
    first = src.strip().splitlines()[0][:120] if src.strip() else ""
    index.append(f"{name}\t{first}")
    count += 1
(out / "INDEX.txt").write_text("\n".join(index), encoding="utf-8")
print(count, "cells extracted")
