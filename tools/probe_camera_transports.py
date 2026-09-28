"""Authorised, RAM-only one-frame transport probe; never prints URLs/secrets."""
import argparse
import json
from pathlib import Path
import sys

import av

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.auth import media_http_options
from sentinel_edge.cameras import (APPROVED_MANUAL_PAIRS, Camera, documented_stream_urls,
                                   hls_headers, rtsp_access_url, source_urls)
from sentinel_edge.config import Config
from sentinel_edge.diagnostic_gateway import discover


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_id',choices=('cam01','cam17'))
    args = parser.parse_args()
    config = Config.from_env(live=True)
    source = 'authorised_catalogue'
    try:
        camera = next((item for item in discover(config) if item.camera_id == args.camera_id),None)
    except Exception as exc:
        # The approved pair may still have working media when catalogue auth is
        # down. This diagnoses transport only; it does not validate catalogue access.
        if (len(config.camera_ids) != 2 or
                frozenset(config.camera_ids) not in APPROVED_MANUAL_PAIRS or
                args.camera_id not in config.camera_ids):
            print(json.dumps({'camera_id':args.camera_id,'status':'catalogue_unavailable',
                              'error_class':type(exc).__name__}),flush=True)
            raise SystemExit(1) from None
        source = 'approved_manual_pair_catalogue_unavailable'
        camera = Camera(args.camera_id,args.camera_id,*documented_stream_urls(config,args.camera_id))
    if camera is None:
        raise ValueError('Camera not in authorised catalogue')
    rtsp,hls = source_urls(camera)
    av.logging.set_level(av.logging.PANIC)
    results = []
    for protocol,url in (('rtsp',rtsp_access_url(config,rtsp)),('hls',hls)):
        if not url:
            results.append({'protocol':protocol,'status':'no_authorised_source'})
            continue
        try:
            options = {'rtsp_transport':'tcp','stimeout':'8000000'} if protocol=='rtsp' else {}
            if protocol=='hls':
                options.update(media_http_options(url,hls_headers(config,url)))
            with av.open(url,options=options,timeout=(8,8)) as container:
                video = next((stream for stream in container.streams if stream.type=='video'),None)
                if video is None:
                    results.append({'protocol':protocol,'status':'no_video_stream'})
                    continue
                frame = next(container.decode(video),None)
                results.append({'protocol':protocol,'status':'decoded' if frame is not None else 'no_frame',
                                'codec':video.codec_context.name if frame is not None else None,
                                'dimensions':[frame.width,frame.height] if frame is not None else None})
        except Exception as exc:
            # Error messages often include authenticated URLs. Class only.
            results.append({'protocol':protocol,'status':'unavailable','error_class':type(exc).__name__})
    print(json.dumps({'camera_id':camera.camera_id,'source':source,'results':results}),flush=True)


if __name__=='__main__':
    main()
