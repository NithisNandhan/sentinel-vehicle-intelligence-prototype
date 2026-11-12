"""Engineering-only registration resemblance, never plate identity confirmation."""
import re

from .plate_format import is_valid_plate


POLICY = 'uncertain_registration_shape_v1'
_SHAPE = re.compile(r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}')


def unconstrained_top(rows,tokens):
    """Best bounded CTC prefix hypothesis BEFORE applying any plate grammar."""
    from .ctc_fusion import prefix_candidates
    ranked=prefix_candidates(rows,tokens,beam_width=100)
    return ranked[0][0] if ranked else None


def ocr_allowed(verdict, *, allow_uncertain=False):
    status=verdict.get('status')
    if status not in ('likely_plate','uncertain','likely_non_plate'):
        raise ValueError('Known visual-region decision required')
    return status=='likely_plate' or (allow_uncertain and status=='uncertain')


def check_reading(verdict, reading):
    """Check the selected OCR top only; never search/repair characters to fit.

    Formatting separators and case can be normalised. Unknown state codes may
    still resemble a registration, but remain explicitly unverified. Rare
    formats outside this project's templates keep their raw text for review.
    """
    ocr_allowed(verdict)
    raw=reading if isinstance(reading,str) and reading else None
    normalized=re.sub(r'[\s-]','',raw.upper()) if raw else None
    output=dict(state='review_needed',raw_reading=raw,normalized_reading=normalized,
        format_status='not_read' if not raw else 'not_registration_shaped',
        display_plate=None,operational_eligible=False,reasons=[])
    if verdict['status']=='likely_non_plate':
        output.update(state='rejected_region',reasons=['Visual region rejected; OCR is not authorised'])
        return output
    if not raw:
        output['reasons']=['No OCR hypothesis is available']
        return output
    recognized=is_valid_plate(normalized)
    shaped=(normalized.isascii() and 6<=len(normalized)<=12 and bool(_SHAPE.fullmatch(normalized))
            and any(c in '123456789' for c in normalized))
    if recognized or shaped:
        output.update(state='provisional_plate',display_plate=normalized,
            format_status='recognized_format' if recognized else 'registration_shape_only',
            reasons=['OCR resembles a registration; identity and character correctness are unconfirmed'])
        if not recognized:
            output['reasons'].append('Character layout only; state/district details are not validated')
    else:
        output['reasons']=['Raw OCR retained; the reading does not match the supported registration shapes']
    return output
