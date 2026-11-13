"""Backward-compatible GPU-only entrypoint for the shared model preflight."""
import json
import os
from check_model_runtime import check_models


def main():
    result=check_models('gpu',os.environ.get('RECORDED_DETECTOR_PROFILE','general'))
    # Preserve the legacy success field consumed by existing launchers/logs.
    result['gpu_preflight']='passed'
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
