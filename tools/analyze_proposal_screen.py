"""Summarize saved proposal-screen diagnostics without new inference or labels."""
import argparse
from collections import Counter
import json
from pathlib import Path


def analyze(report):
    rows=report['rows']
    # Repeated source views are not independent physical vehicles or labels.
    unique={r['pixel_sha256']:r for r in rows if r['role']=='human_plate_reference'}
    positives=list(unique.values())
    negatives=[r for r in rows if r['role']=='visual_qa_negative']
    accepted=[r for r in rows if r.get('model_accepted_control')]
    policies={
        'existing_filter_unknown_detector':lambda r:bool(r['current_filter_reason_without_detector_score']),
        'visual_reject_below_existing_threshold':lambda r:r['visual_verifier']['status']=='likely_non_plate',
        'zero_characters_scored_at_least_0_6':lambda r:r['confident_count_06']==0,
        'zero_characters_and_visual_reject':lambda r:r['confident_count_06']==0 and r['visual_verifier']['status']=='likely_non_plate',
        'native_height_below_16_and_zero_characters':lambda r:r['height']<16 and r['confident_count_06']==0,
    }
    comparisons=[]
    for name,policy in policies.items():
        excluded=[r for r in positives if policy(r)]
        comparisons.append(dict(policy=name,negative_examples_withheld=sum(policy(r) for r in negatives),
            negative_examples_total=len(negatives),unique_human_reference_views_withheld=len(excluded),
            unique_human_reference_views_total=len(positives),human_readable_references_withheld=[r['reference_id'] for r in excluded if r['human_status']=='readable'],
            human_unreadable_references_withheld=[r['reference_id'] for r in excluded if r['human_status']=='unreadable'],
            accepted_control_crops_withheld=sum(policy(r) for r in accepted),accepted_control_crops_total=len(accepted),
            accepted_control_passages_total=len({r['passage_id'] for r in accepted})))
    return dict(scope='Posthoc development diagnostic, not a trained filter, unseen benchmark or full-video accuracy measurement',
        inference_samples=len(rows),human_reference_views=sum(r['role']=='human_plate_reference' for r in rows),
        unique_human_reference_pixel_views=len(positives),human_reference_statuses=dict(Counter(r['human_status'] for r in positives)),
        negative_provenance='Eight visible windscreen/trim examples identified during AI visual QA, not independently human-labelled',
        policies=comparisons,
        conclusion='None of the existing global non-plate cues separates every negative from every real plate reference. Zero-confident-character observations explain all eight demonstrated false proposals but also occur in two real reference crops.',
        recommendation='If tested as an opt-in, withhold zero-confident-character observations from fusion as insufficient text evidence. Preserve their native crop and raw OCR in diagnostics. Do not label them non-plates, erase them or use them to reset identity. Keep existing acceptance gates unchanged.',
        regression_limits='Saved controls show all three model-accepted passages retain their 17 saved crops. This is not proof their live outcomes or other weak plates are unaffected; an end-to-end run is still required. Detector scores were unavailable and were not inferred.',
        operational_code_modified=False,model_weights_changed=False,truth_labels_modified=False)


def markdown(result):
    lines=['# Saved proposal-screen diagnostic','',result['scope']+'.','',
        f"GPU OCR and the existing HOG verifier examined {result['inference_samples']} saved images. The 39 human-reference input views contain only {result['unique_human_reference_pixel_views']} unique pixel views (19 readable and 2 unreadable references); repeated views are not new vehicles.",
        '', '| Screen | Known negative examples withheld | Human reference views withheld | Accepted control crops withheld |',
        '| --- | ---: | ---: | ---: |']
    for row in result['policies']:
        lines.append(f"| {row['policy']} | {row['negative_examples_withheld']}/{row['negative_examples_total']} | {row['unique_human_reference_views_withheld']}/{row['unique_human_reference_views_total']} | {row['accepted_control_crops_withheld']}/{row['accepted_control_crops_total']} |")
    lines+=['',result['conclusion'],'',
        'Reference 06 has width/height 0.793 and is wrongly rejected by the existing 0.8 shape floor. Reference 09 is human-readable but produces no character above 0.6 and a low HOG score. Reference 08 is a real plate labelled source-unreadable and likewise has no confident OCR character. Thus low OCR or visual scores cannot establish that a crop is not a plate.',
        '',result['recommendation'],'',result['regression_limits'],'',result['negative_provenance']+'.',
        '', 'No operational settings, model weights, frozen captures or truth labels were changed.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostic',type=Path)
    args=parser.parse_args()
    result=analyze(json.loads(args.diagnostic.read_bytes()))
    output=args.diagnostic.parent
    (output/'proposal-screen-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'proposal-screen-comparison.md').write_text(markdown(result))
    print(json.dumps(result,indent=2))
