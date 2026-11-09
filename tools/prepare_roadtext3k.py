"""Decode a verified Indian archive once and retain bounded native plate crops."""
import argparse
from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile

from sentinel_edge.awiros_experiment import file_hash
from sentinel_edge.plate_quality import crop_quality
from sentinel_edge.roadtext_corpus import archive_member_kind,blocked_splits,check_splits,native_box,plate_text


def prepare(args):
    import av
    import cv2
    if any((args.output/name).exists() for name in ('manifest.json','crops.tar','all-crops.tar')):
        raise ValueError('Preserve the existing RoadText crop set; use a fresh output directory')
    receipt=json.loads(args.archive.with_suffix(args.archive.suffix+'.receipt.json').read_text())
    if file_hash(args.archive)!=receipt['sha256']:raise ValueError('Indian archive bytes changed')
    annotation_receipt=json.loads(args.annotations.with_suffix('.receipt.json').read_text())
    if file_hash(args.annotations)!=annotation_receipt['local_sha256']:
        raise ValueError('Publisher annotation subset changed after acquisition')
    annotations=json.loads(args.annotations.read_text())
    excluded=set()
    for path in args.exclude:
        document=json.loads(path.read_text())
        excluded.update(r['transcription'] for r in document.get('samples',[]))
        excluded.update(p['ground_truth'] for p in document.get('passages',[]))
    args.output.mkdir(parents=True,exist_ok=True)
    counts=Counter();rows=[];videos=[];global_pixels=set()
    archive_prefix=args.archive.name.removesuffix('.tar.gz')
    with tarfile.open(args.archive,mode='r|gz',bufsize=1024*1024) as archive, tarfile.open(args.output/'all-crops.tar',mode='w') as payload:
        for member in archive:
            if not member.isfile():continue
            if archive_member_kind(member.name,archive_prefix)=='annotations':
                counts['bundled_annotation_file_skipped']+=1;continue
            path=PurePosixPath(member.name)
            if member.size>100*1024*1024:raise ValueError('Individual source video exceeds bound')
            video_id=path.stem
            frames=annotations.get(video_id)
            if not frames:counts['videos_without_public_annotations']+=1;continue
            tracks=defaultdict(list)
            for frame_id,frame in frames.items():
                for item in frame.get('labels') or []:
                    text,reason=plate_text(item)
                    if reason:counts[reason]+=1;continue
                    if text in excluded:counts['legacy_registration_overlap']+=1;continue
                    if not isinstance(item.get('id'),(int,str)) or not str(frame_id).isdigit():
                        counts['invalid_identity']+=1;continue
                    track=str(item['id'])
                    tracks[track].append((int(frame_id),text,item.get('box2d')))
            admitted={}
            for track,observations in tracks.items():
                if len({o[1] for o in observations})!=1:counts['conflicting_track_labels']+=1;continue
                admitted[track]=observations
            if not admitted:counts['videos_without_admitted_full_plate_labels']+=1;continue
            by_frame=defaultdict(list)
            for track,observations in admitted.items():
                previous=-1000
                for frame_id,text,box in sorted(observations):
                    if frame_id-previous<9:continue
                    by_frame[frame_id].append((track,text,box));previous=frame_id
            media=archive.extractfile(member).read()
            source_hash=hashlib.sha256(media).hexdigest()
            retained=defaultdict(list);total_frames=0;first_pts=None;dimensions=None
            with av.open(io.BytesIO(media)) as container:
                stream=container.streams.video[0]
                stream.thread_type='AUTO'
                stream.codec_context.thread_count=4
                for index,frame in enumerate(container.decode(stream),1):
                    total_frames=index
                    if index not in by_frame:continue
                    image=frame.to_ndarray(format='bgr24')
                    dimensions=list(image.shape[:2])
                    pts_ms=round(float(frame.pts*frame.time_base)*1000) if frame.pts is not None else round((index-1)*1000/30)
                    if first_pts is None:first_pts=pts_ms
                    for track,text,raw_box in by_frame[index]:
                        box=native_box(raw_box,image.shape[1],image.shape[0])
                        if box is None:counts['invalid_or_small_native_box']+=1;continue
                        x1,y1,x2,y2=box;crop=image[y1:y2,x1:x2].copy()
                        quality=crop_quality(crop)
                        pixels=hashlib.sha256(crop.tobytes()).hexdigest()
                        if pixels in global_pixels:counts['duplicate_native_pixels']+=1;continue
                        global_pixels.add(pixels)
                        ok,encoded=cv2.imencode('.png',crop)
                        if not ok:raise ValueError('Native crop encoding failed')
                        uid=hashlib.sha256(f'{video_id}:{track}:{index}'.encode()).hexdigest()[:20]
                        record=dict(id=uid,video_id=video_id,track_id=track,frame_number=index,
                            pts_ms=pts_ms,box=list(box),width=crop.shape[1],height=crop.shape[0],
                            transcription=text,quality=quality,decoded_pixel_sha256=pixels,
                            source_video_sha256=source_hash,crop_path=f'crops/{uid}.png',
                            label_tier='publisher_human_transcription',label_grain='full_registration_shaped_plate_line')
                        retained[track].append((quality,record,encoded.tobytes()))
                        retained[track].sort(key=lambda value:(-value[0],value[1]['frame_number']))
                        del retained[track][args.frames_per_track:]
            # Some source JSONs contain terminal entries after the last frame.
            # They cannot produce pixels and are explicitly quarantined.
            counts['annotation_frame_keys_beyond_video']+=sum(int(i)>total_frames for i in frames)
            counts['requested_crop_observations_beyond_video']+=sum(
                len(items) for index,items in by_frame.items() if index>total_frames)
            for values in retained.values():
                for _,record,encoded in values:
                    member_info=tarfile.TarInfo(record['crop_path']);member_info.size=len(encoded)
                    payload.addfile(member_info,io.BytesIO(encoded))
                    record['crop_sha256']=hashlib.sha256(encoded).hexdigest();rows.append(record)
                if len(videos)<4 and values:
                    preview=args.output/'previews';preview.mkdir(exist_ok=True)
                    record,encoded=values[0][1:]
                    (preview/(record['id']+'.png')).write_bytes(encoded)
            videos.append(dict(video_id=video_id,frames=total_frames,source_sha256=source_hash,
                annotation_max_frame=max(map(int,frames),default=0),
                resolution=dimensions,retained_tracks=len(retained),retained_crops=sum(map(len,retained.values()))))
            print(json.dumps(dict(stage='extract_native_crops',video_id=video_id,tracks=len(retained),cumulative_crops=len(rows))),flush=True)
    # Preserve complete extraction metadata before any split admission gate.
    (args.output/'extraction.json').write_text(json.dumps(dict(samples=rows,videos=videos,exclusions=dict(counts)),indent=2))
    rows,quarantined=blocked_splits(rows)
    counts.update(quarantined)
    check_splits(rows)
    summary={split:dict(crops=sum(r['split']==split for r in rows),
        tracks=len({(r['video_id'],r['track_id']) for r in rows if r['split']==split}),
        registrations=len({r['transcription'] for r in rows if r['split']==split}),
        videos=len({r['video_id'] for r in rows if r['split']==split})) for split in ('train','validation','test')}
    if any(summary[s]['registrations']<5 for s in summary) or summary['train']['registrations']<20:
        raise ValueError('Insufficient disjoint registered vehicles for the bounded experiment')
    admitted_paths={row['crop_path'] for row in rows}
    with tarfile.open(args.output/'all-crops.tar',mode='r|') as source, tarfile.open(args.output/'crops.tar',mode='w') as selected:
        for member in source:
            if member.isfile() and member.name in admitted_paths:
                selected.addfile(member,source.extractfile(member))
    document=dict(schema_version=1,dataset='RoadText-3K Indian bounded archive',
        archive_sha256=receipt['sha256'],annotations_sha256=file_hash(args.annotations),
        source_url=receipt['url'],frame_indexing='1-based annotation frame matches decode index+1',
        split_policy='Custom ordered-video 50:20:30 blocks, all cross-block registrations quarantined; not official split IDs',
        selection='Publisher plate category, readable nonempty human text, unchanged canonical full-registration shape, native >=24x8',
        evaluation_caveat='Bounded vehicle-mounted Indian source subset; not fixed CCTV or an independent human audit of all source labels',
        source_terms='Published academic training benchmark; redistribution/commercial rights not established',
        samples=rows,summary=summary,exclusions=dict(counts),videos=videos,
        crop_archive='crops.tar',crop_archive_sha256=file_hash(args.output/'crops.tar'),
        training_source_sha256=file_hash(Path(__file__)),production_promoted=False)
    (args.output/'manifest.json').write_text(json.dumps(document,indent=2))
    print(json.dumps(dict(stage='roadtext_manifest_frozen',summary=summary,exclusions=dict(counts))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--exclude',type=Path,action='append',default=[])
    parser.add_argument('--frames-per-track',type=int,default=8,choices=range(3,13))
    prepare(parser.parse_args())
