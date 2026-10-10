"""Generated test footage. Flat synthetic frames only; no genuine imagery."""
from fractions import Fraction


def make_video(path, *, frames=60, fps=10, size=(160, 120)):
    """Write a small MPEG-4 file whose frame i has PTS i/fps seconds."""
    import av
    import numpy as np
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = 'yuv420p'
        stream.codec_context.time_base = Fraction(1, fps)
        for index in range(frames):
            image = np.full((size[1], size[0], 3), (index * 4) % 250, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def register(path, **overrides):
    from sentinel_edge.recorded_source import write_receipt
    fields = dict(source='generated in the test suite', licence='not applicable: synthetic',
                  permission='created by this project for testing')
    fields.update(overrides)
    return write_receipt(path, **fields)
