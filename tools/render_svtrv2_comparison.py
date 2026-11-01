"""Create a local, licensed diagnostic viewer and a plate-free aggregate report."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path


def render(root, corpus, output, *, allow_private_evaluation=False):
    reports = [json.loads((root / filename).read_text(encoding="utf-8")) for filename in
               ("awiros-parent-test.json", "awiros-adapted-test.json", "svtrv2-test.json")]
    if len({row["manifest_sha256"] for row in reports}) != 1 or any(row["split"] != "test" for row in reports):
        raise ValueError("Comparison requires the same frozen test manifest")
    manifest = json.loads((corpus / "adaptation_manifest.json").read_text(encoding="utf-8"))
    actual_manifest_hash = hashlib.sha256((corpus / "adaptation_manifest.json").read_bytes()).hexdigest()
    if reports[0]["manifest_sha256"] != actual_manifest_hash:
        raise ValueError("Viewer manifest differs from the measured inputs")
    private = manifest.get("dataset", {}).get("purpose") == "private_noncommercial_evaluation_only"
    if private and not allow_private_evaluation:
        raise ValueError("Explicit local private-evaluation viewer flag required")
    if private and (not manifest.get("no_training") or not manifest["dataset"].get("no_redistribution_of_images_or_crops")):
        raise ValueError("Private evaluation permission receipt required")
    if private and not any(tuple(output.resolve().parts[i:i+3]) == ("data", "benchmarks", "reports") for i in range(len(output.resolve().parts)-2)):
        raise ValueError("Private images must remain in the ignored local benchmark viewer directory")
    source_samples = {sample["id"]: sample for sample in manifest["samples"]}
    if len({tuple(row["id"] for row in report["stills"]) for report in reports}) != 1:
        raise ValueError("Comparison sample ordering differs")
    names = [report["model"]["name"] for report in reports]
    esc = lambda value: html.escape(str(value), quote=True)
    parts = ["""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SVTRv2-B and Awiros comparison</title>
<style>body{font:15px/1.6 system-ui,sans-serif;background:#f6f5f1;color:#242925;max-width:1250px;margin:auto;padding:24px}h1,h2{line-height:1.3}section{background:#fffdf9;border:1px solid #dedfd8;border-radius:8px;padding:20px;margin:20px 0}img{max-width:100%;max-height:190px;object-fit:contain}.native{background:#edece6;padding:16px;text-align:center}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #dedfd8;vertical-align:top}.scroll{overflow:auto}code{font-size:16px;overflow-wrap:anywhere}.correct{color:#166534}.wrong{color:#b91c1c}.note{background:#fff6e6;padding:16px;border-left:4px solid #92400e}small{color:#626b65}a{color:#0f766e}</style>
<h1>SVTRv2-B vs Awiros</h1><p>Recorded benchmark diagnostics · 31 Indian still crops + one three-frame development passage · no production model switch</p>
<p class="note">These still-image reference labels include AI review and publisher agreement, not a large independently human-labelled CCTV test. Pretraining overlap is unknown. Stills are not counted as video observations. Scores are not accuracy probabilities.</p>
<h2>Measured comparison</h2><div class="scroll"><table><tr><th>Model</th><th>Exact stills</th><th>Character errors</th><th>Median OCR</th><th>P95 OCR</th></tr>"""]
    parts[0] = parts[0].replace("31 Indian still crops + one three-frame development passage", f'{len(reports[0]["stills"])} Indian still crops · {len(reports[0]["sequences"])} sequence passages')
    parts[0] = parts[0].replace("small{color:#626b65}", "small{color:#626b65;overflow-wrap:anywhere}")
    if private:
        parts[0] = parts[0].replace("Recorded benchmark diagnostics", "Separate-source benchmark diagnostics")
        parts[0] = parts[0].replace("These still-image reference labels include AI review and publisher agreement, not a large independently human-labelled CCTV test.", "Separate-source evaluation against publisher XML transcriptions, not new independent human verification.")
        parts[0] = parts[0].replace("<h2>Measured comparison</h2>", '<p class="note">Private noncommercial evaluation only. Do not redistribute this image-containing HTML or its crops. Dataset CC BY-NC-ND 4.0; no training performed.</p><h2>Measured comparison</h2>')
    for report in reports:
        s = report["summary"]
        parts.append(f'<tr><td>{esc(report["model"]["name"])}</td><td>{s["exact"]}/{s["samples"]}</td><td>{s["character_errors"]}/{s["characters"]}</td><td>{s["median_ocr_ms"]} ms</td><td>{s["p95_ocr_ms"]} ms</td></tr>')
    parts.append('</table></div><p><small>RTX 3050 6GB, FP32, batch one, one timed pass after warm-up. OCR timing includes input preprocessing/transfer/output materialisation, excludes detection, fusion and dashboard. Frameworks differ. These are diagnostic timings, not full-pipeline capacity.</small></p>')
    for index, sample in enumerate(reports[0]["stills"]):
        source = source_samples[sample["id"]]
        allowed_licences = {"CC-BY-NC-ND-4.0"} if private else {"CC-BY-4.0"}
        if any(item.get("license") not in allowed_licences for item in source["provenance"]):
            raise ValueError("Viewer needs reviewed crop provenance")
        filename = (corpus / sample["crop_path"]).resolve()
        if not filename.is_relative_to(corpus.resolve()):
            raise ValueError("Unsafe viewer image path")
        image = filename.read_bytes()
        if hashlib.sha256(image).hexdigest() != sample["crop_sha256"]:
            raise ValueError("Modified viewer image")
        data = base64.b64encode(image).decode("ascii")
        parts.append(f'<section><h2>Sample {index+1} · reference <code>{esc(sample["expected"])}</code></h2><p><small>{esc(sample["label_tier"])} · {sample["native_shape"][1]} × {sample["native_shape"][0]} native pixels</small></p><div class="native"><img alt="Licensed native plate diagnostic crop" src="data:image/png;base64,{data}"></div><div class="scroll"><table><tr><th>Recognizer</th><th>Reading</th><th>Against reference</th></tr>')
        for name, report in zip(names, reports):
            row = report["stills"][index]
            parts.append(f'<tr><td>{esc(name)}</td><td><code>{esc(row["reading"] or "No complete reading")}</code></td><td class="{"correct" if row["exact"] else "wrong"}">{"Exact" if row["exact"] else "Different"}</td></tr>')
        parts.append('</table></div><small>')
        for provenance in source["provenance"]:
            parts.append(f'{esc(provenance.get("attribution", "Publisher dataset"))} · {esc(provenance["license"])} · <a href="{esc(provenance["source_url"])}">Source</a><br>')
        parts.append('</small></section>')
    if all(report["sequences"] for report in reports):
        parts.append('<h2>Three-frame development passage</h2><p>Three distinct video-derived images, manually supplied plate boxes and assigned replay timestamps. This isolates recognition and does not test automatic detection or uninterrupted full-video operation.</p><div class="scroll"><table><tr><th>Model</th><th>Frame readings</th><th>Fused top candidate</th><th>Accepted by unchanged gate</th></tr>')
        for report in reports:
            row = report["sequences"][0]
            parts.append(f'<tr><td>{esc(report["model"]["name"])}</td><td>{"<br>".join(esc(f["reading"]) for f in row["frames"])}</td><td><code>{esc(row["top1"])}</code></td><td>{esc(row["accepted"] or "Abstained")}</td></tr>')
        parts.append('</table></div><p>Sequence source: <a href="https://universe.roboflow.com/nivu/indian-license-plate-knte7">NIVU Indian plate sequence pilot</a>, CC BY 4.0.</p>')
    parts.append('<p>Model sources: <a href="https://github.com/Topdu/OpenOCR">OpenOCR / SVTRv2</a> and <a href="https://huggingface.co/Awiros/anpr-ocr">Awiros ANPR OCR</a>. See the repository comparison report for hashes, reproduction commands and caveats.</p></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
    summary = {"evaluation_caveat": reports[0]["evaluation_caveat"], "production_promoted": False,
               "manifest_sha256": reports[0]["manifest_sha256"],
               "models": [{"model": report["model"], "summary": report["summary"],
                           "sequence": {k: report["sequences"][0][k] for k in ("top1_exact", "truth_rank", "correct_accepted", "wrong_accepted")} if report["sequences"] else None,
                           "by_source": {group: {"samples": sum(row.get("source_group") == group for row in report["stills"]),
                                                  "exact": sum(row["exact"] and row.get("source_group") == group for row in report["stills"])}
                                         for group in sorted({row.get("source_group") for row in report["stills"] if row.get("source_group")})}}
                          for report in reports]}
    (root / "aggregate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"viewer": str(output), "bytes": output.stat().st_size, "models": len(reports), "stills": len(reports[0]["stills"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-private-evaluation", action="store_true")
    args = parser.parse_args()
    render(args.reports, args.corpus, args.output, allow_private_evaluation=args.allow_private_evaluation)
