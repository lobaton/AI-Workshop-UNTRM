"""One-time: run three sizes of the ESM-2 protein language model on ubiquitin and save, for each, its predicted
contact map and its most contact-like attention head, so the marimo notebook does not need the `transformers`
package or a model download.

Needs: pip install transformers (downloads the models from Hugging Face) and internet access to RCSB PDB.

Usage (from session2_inside_the_black_box/src):
    python tools/precompute_esm.py

Output: data/esm2_ubiquitin.npz (see data/README.md).
"""
import os
import sys
import urllib.request

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import attention_common as at  # noqa: E402

SEQUENCE = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"  # human ubiquitin
MODELS = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D", "facebook/esm2_t30_150M_UR50D"]
CONTACT_DISTANCE = 8.0  # Angstrom between C-beta atoms (C-alpha for glycine)
MIN_SEPARATION = 6  # ignore pairs closer than this along the chain when scoring


def true_contacts(pdb_id="1UBQ"):
    """Contact map of the crystal structure: 1 where two residues' C-beta atoms are within 8 Angstrom."""
    with urllib.request.urlopen(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60) as response:
        lines = response.read().decode().splitlines()
    coords = {}
    for line in lines:
        if line.startswith("ATOM"):
            atom, residue, number = line[12:16].strip(), line[17:20], int(line[22:26])
            if atom == "CB" or (atom == "CA" and residue == "GLY"):
                coords[number] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    xyz = np.array([coords[i] for i in range(1, len(SEQUENCE) + 1)])
    return (np.linalg.norm(xyz[:, None] - xyz[None], axis=-1) < CONTACT_DISTANCE).astype(np.uint8)


def top_precision(pred, true):
    """Share of the top-L predicted pairs (at least MIN_SEPARATION apart) that are true contacts."""
    iu = np.triu_indices(len(true), MIN_SEPARATION)
    top = np.argsort(-pred[iu])[:len(true)]
    return float(true[iu][top].mean())


def best_head(attention, true):
    """The head whose map (averaged in both directions) correlates best with the true contacts."""
    iu = np.triu_indices(len(true), MIN_SEPARATION)
    scores = np.array([[np.corrcoef((a + a.T)[iu], true[iu])[0, 1] for a in layer] for layer in attention])
    best = np.unravel_index(np.nanargmax(scores), scores.shape)
    return best, scores[best]


true = true_contacts()
contacts, params, precision, heads, head_maps = [], [], [], [], []
for name in MODELS:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = EsmForMaskedLM.from_pretrained(name, attn_implementation="eager").eval()
    inputs = tokenizer(SEQUENCE, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
        pred = model.predict_contacts(inputs["input_ids"], inputs["attention_mask"])[0].numpy()
    # (layers, heads, residues, residues), without the start/end tokens
    attention = torch.stack(out.attentions)[:, 0, :, 1:-1, 1:-1].numpy()
    best, corr = best_head(attention, true)
    heads.append(best)
    head_maps.append(attention[best])
    contacts.append(pred)
    params.append(sum(p.numel() for p in model.parameters()))
    precision.append(top_precision(pred, true))
    print(f"{name}: {params[-1]:,} parameters, top-L contact precision {precision[-1]:.0%}; most contact-like "
          f"head: layer {best[0] + 1}, head {best[1] + 1} (correlation {corr:.2f})")

out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), at.ESM_FILE)
np.savez_compressed(
    out_path, sequence=np.array(SEQUENCE), models=np.array(MODELS), params=np.array(params),
    contacts=np.stack(contacts).astype(np.float16),
    true_contacts=true, precision=np.array(precision), best_head=np.array(heads),
    best_head_attention=np.stack(head_maps).astype(np.float16),
)
print(f"Saved {out_path} ({os.path.getsize(out_path) / 1e3:.0f} kB)")
