import os
from pathlib import Path
from lazy_dataset.database import JsonDatabase
import paderbox as pb

if 'LIBRISPEECH_JSON_PATH' in os.environ:
    LIBRISPEECH_JSON_PATH = Path(os.environ['LIBRISPEECH_JSON_PATH'])
else:
    LIBRISPEECH_JSON_PATH = None

if 'LIBRISPEECH_VAD_PATH' in os.environ:
    LIBRISPEECH_VAD_PATH = Path(os.environ['LIBRISPEECH_VAD_PATH'])
else:
    raise RuntimeError('LIBRISPEECH_VAD_PATH not set in environment variables.')

if 'LIBRISPEECH_VAD_JSON_PATH' in os.environ:
    LIBRISPEECH_VAD_JSON_PATH = Path(os.environ['LIBRISPEECH_VAD_JSON_PATH'])
else:
    raise RuntimeError('LIBRISPEECH_VAD_JSON_PATH not set in environment variables.')

def create():
    vad_json = pb.io.load_json(LIBRISPEECH_VAD_PATH)
    ds_json = pb.io.load_json(LIBRISPEECH_JSON_PATH)
    for ds_name in ds_json['datasets'].keys():
        missing = 0
        print(f"Processing VAD for dataset: {ds_name}")
        for key, ex in ds_json['datasets'][ds_name].items():
            if key in vad_json:
                ex['activity'] = vad_json[key]
            else:
                missing += 1
        if missing > 0:
            print(f"  Missing VAD for {missing} of {len(ds_json['datasets'][ds_name])} examples in dataset {ds_name}")
    pb.io.dump(ds_json, LIBRISPEECH_VAD_JSON_PATH)

if __name__ == "__main__":
    create()
