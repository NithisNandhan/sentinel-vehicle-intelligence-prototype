"""Optional learned plate-region verification before any OCR is invoked."""
from collections import Counter
import math

from .text_boundaries import warp_polygon


def verify_proposals(image, proposals, verifier, *, observe=None):
    """Return only likely plates; preserve every other decision for review.

    The callback receives native/rectified images, never OCR readings. No
    transformation edits the source. Scores describe the region classifier,
    not character confidence or confirmed vehicle identity.
    """
    accepted, audit = [], []
    counts = Counter()
    h,w=image.shape[:2]
    for index, proposal in enumerate(proposals):
        box=proposal.get('box',())
        if len(box)!=4 or any(not math.isfinite(float(v)) for v in box):
            raise ValueError('Finite native proposal coordinates required')
        a,b,c,d=map(int,box)
        if not 0<=a<c<=w or not 0<=b<d<=h:
            raise ValueError('Proposal outside native image')
        native=image[b:d,a:c]
        working=warp_polygon(native,proposal['quad']) if proposal.get('quad') else None
        if working is None:
            working=native.copy()
        verdict=verifier.verify(working,context={'search_box':proposal.get('search_box'),
            'proposal_box':list(box),'geometry':proposal.get('geometry')})
        if verdict.get('status') not in ('likely_plate','uncertain','likely_non_plate'):
            raise ValueError('Invalid verifier status')
        item={**proposal,'plate_verification':verdict,'ocr_eligible':verdict['status']=='likely_plate'}
        if observe is not None:
            observe(index,item,native,working)
        audit.append(item)
        counts[verdict['status']]+=1
        if item['ocr_eligible']:
            accepted.append(item)
    # With verification enabled, a complete oriented text enclosure gets a
    # chance before an overlapping legacy fragment. Only one view survives the
    # caller's same-frame spatial deduplication; this adds no temporal votes.
    accepted.sort(key=lambda p:(-int(bool(p.get('geometry'))),
                                -float(p['plate_verification'].get('classifier_score',0)),
                                -float(p.get('confidence',0))))
    return accepted,audit,dict(counts)
