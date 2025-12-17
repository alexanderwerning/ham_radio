import os

import numpy as np
import paderbox as pb
from pathlib import Path
from lazy_dataset.database import JsonDatabase
import lazy_dataset
from paderbox.array.interval.core import ArrayInterval_from_str, zeros

import mms_msg
from mms_msg import keys
from mms_msg import sampling, simulation
from mms_msg.sampling.pattern.meeting.overlap_sampler import UniformOverlapSampler
from mms_msg.sampling.pattern.meeting.scenario_sequence_sampler import sample_random, sample_balanced_no_repeat
from mms_msg.simulation.noise import white_microphone_noise
from mms_msg.databases.reverberation.sms_wsj import SMSWSJRIRDatabase
from mms_msg.simulation.utils import load_audio
import hashlib

# from mms_msg.databases.reverberation.create_sms_wsj_rirs import get_rng
def get_rng(dataset, example_id):
    string = f"{dataset}_{example_id}"
    seed = (
            int(hashlib.sha256(string.encode("utf-8")).hexdigest(),
                16) % 2 ** 32
    )
    return np.random.default_rng(seed=seed)

if 'LIBRISPEECH_VAD_JSON_PATH' in os.environ:
    LIBRISPEECH_VAD_JSON_PATH = Path(os.environ['LIBRISPEECH_VAD_JSON_PATH'])
else:
    raise RuntimeError('LIBRISPEECH_VAD_PATH not set in environment variables.')

if 'RIR_SCENARIO_JSON_PATH' in os.environ:
    RIR_SCENARIO_JSON_PATH = Path(os.environ['RIR_SCENARIO_JSON_PATH'])
else:
    raise RuntimeError('RIR_SCENARIO_JSON_PATH not set in environment variables.')

if 'MUSAN_JSON_PATH' in os.environ:
    MUSAN_JSON_PATH = Path(os.environ['MUSAN_JSON_PATH'])
else:
    raise RuntimeError('MUSAN_JSON_PATH not set in environment variables.')

if 'MEETING_VAD_JSON_PATH' in os.environ:
    MEETING_VAD_JSON_PATH = Path(os.environ['MEETING_VAD_JSON_PATH'])
else:
    raise RuntimeError('MEETING_VAD_JSON_PATH not set in environment variables.')

def merge_activity(example):
    activity_list = example['activity']
    activity_arr_int = [ArrayInterval_from_str(*s) for s in activity_list]
    example_activity = zeros((int(example['num_samples']['observation']),))
    for activity, offset in zip(activity_arr_int, example['offset']['original_source']):
        example_activity[offset: len(activity) + offset] = activity
    example['activity'] = example_activity.to_serializable()
    return example

def add_noise_samples(example, noise_ds, amount, min_snr, max_snr):
    rng = get_rng(example['dataset'], example['example_id'])
    num_noise_samples = rng.integers(amount[0], amount[1], endpoint=True)
    audio_path = []
    noise_ds = noise_ds.shuffle(rng=rng)
    for noise_example, _ in zip(noise_ds, range(num_noise_samples)):
        audio_path.append(noise_example['audio_path']['observation'])
    # example['audio_path']['original_source'].extend(audio_path)
    num_samples = [pb.io.audioread.audio_length(path) for path in audio_path]
    log_weights = rng.uniform(min_snr, max_snr, size=num_noise_samples)
    # example['num_samples']['original_source'].extend(num_samples)
    # example['log_weights'].extend(rng.uniform(min_snr, max_snr, size=num_noise_samples).tolist())
    offsets = rng.uniform(0, example['num_samples']['observation']-1, size=num_noise_samples).astype(int)
    # example['offset']['original_source'].extend([rng.uniform(0, ns) for ns in num_samples])
    example['audio_path']['noise'] = audio_path
    example['num_samples']['noise'] = num_samples
    example['noise_log_weights'] = log_weights.tolist()
    example['offset']['noise'] = offsets.tolist()
    return example

def load_noise(example):
    for noise_path, offset, log_weight in zip(example['audio_path']['noise'], example['offset']['noise'], example['noise_log_weights']):
        noise_source = pb.io.audioread.load_audio(noise_path).astype(np.float32)
        end = min(offset + noise_source.shape[0], example['num_samples']['observation'])
        observation_power = np.mean( example['audio_data']['observation'][0] ** 2, keepdims=True)
        augmentation_power = np.mean(noise_source ** 2, keepdims=True)

        current_snr = 10 * np.log10(observation_power / augmentation_power)
        factor = 10 ** (-(log_weight - current_snr) / 20)
        example['audio_data']['observation'][0, offset:end] += noise_source[0:end-offset] * factor
    return example

def drop_audio(example):
    for key in list(example['audio_data'].keys()):
        if key != 'observation':
            del example['audio_data'][key]
        else:
            example['audio_data'][key] = example['audio_data'][key].astype(np.float32)
    return example


# create meeting dataset class
class LibrispeechMeeting():
    def __init__(
            self, json_path: [str, Path] = LIBRISPEECH_VAD_JSON_PATH,
            rir_scenario_json_path: [str, Path] = RIR_SCENARIO_JSON_PATH,
            musan_json_path: [str, Path] = MUSAN_JSON_PATH,
            num_speakers=8,
            max_weight=5,
            min_snr=20,
            max_snr=30,
            meeting_duration=240*16000,
            p_silence=1,
            maximum_silence=128000,
            maximum_overlap=0,
            max_concurrent_spk=1,
            soft_minimum_overlap=0,
            sampling_strategy='random',
            noise_number_of_samples=(5,20),
            noise_snr_range=(10,30),

            ):
        self.json_path = Path(json_path)
        self.rir_scenario_json_path = Path(rir_scenario_json_path)
        self.musan_json_path = Path(musan_json_path)
        self.num_speakers = num_speakers
        self.max_weight = max_weight
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.meeting_duration = meeting_duration
        self.p_silence = p_silence
        self.maximum_silence = maximum_silence
        self.maximum_overlap = maximum_overlap
        self.max_concurrent_spk = max_concurrent_spk
        self.soft_minimum_overlap = soft_minimum_overlap
        self.sampling_strategy = sampling_strategy
        self.noise_number_of_samples = noise_number_of_samples
        self.noise_snr_range = noise_snr_range
    
    def get_raw_dataset(self, dataset_name, seed=1):
        if dataset_name == 'train':
            input_ds = JsonDatabase(
                json_path=self.json_path
            ).get_dataset('train_clean_100')
            rir_ds = SMSWSJRIRDatabase(scenarios_json=self.rir_scenario_json_path).get_dataset('train_si284')
        elif dataset_name == 'dev':
            input_ds = JsonDatabase(
                json_path=self.json_path
            ).get_dataset('dev_clean')
            rir_ds = SMSWSJRIRDatabase(scenarios_json=self.rir_scenario_json_path).get_dataset('cv_dev93')
        elif dataset_name == 'test':
            input_ds = JsonDatabase(
                json_path=self.json_path
            ).get_dataset('test_clean')
            rir_ds = SMSWSJRIRDatabase(scenarios_json=self.rir_scenario_json_path).get_dataset('test_eval92')
        else:
            raise ValueError(dataset_name)
        noise_ds = JsonDatabase(
            json_path=self.musan_json_path
        ).get_dataset('noise')

        ds = sampling.source_composition.get_composition_dataset(input_dataset=input_ds, num_speakers=self.num_speakers, rng=seed)
        ds = ds.map(sampling.environment.scaling.UniformScalingSampler(max_weight=self.max_weight))
        noise_sampler = sampling.environment.noise.UniformSNRSampler(min_snr=self.min_snr, max_snr=self.max_snr)
        ds = ds.map(noise_sampler)
        ds = ds.map(sampling.environment.rir.RIRSampler(rir_ds))
        if self.sampling_strategy == 'balanced_no_repeat':
            scenario_sequence_sampler = sample_balanced_no_repeat
        elif self.sampling_strategy == 'random':
            scenario_sequence_sampler = sample_random
        else:
            raise ValueError(f'Unknown sampling strategy: {self.sampling_strategy}')
        ds = ds.map(sampling.pattern.meeting.MeetingSampler(duration=self.meeting_duration, overlap_sampler=UniformOverlapSampler(p_silence=self.p_silence,
                                                                                                                    maximum_silence=self.maximum_silence,
                                                                                                                    maximum_overlap=self.maximum_overlap,
                                                                                                                    max_concurrent_spk=self.max_concurrent_spk,
                                                                                                                    soft_minimum_overlap=self.soft_minimum_overlap
                                                                                                                    ),
                                                        scenario_sequence_sampler=scenario_sequence_sampler)(input_ds))
        ds = ds.map(lambda example: add_noise_samples(example, noise_ds, self.noise_number_of_samples, min_snr=self.noise_snr_range[0], max_snr=self.noise_snr_range[1]))
        return ds

    def get_dataset(self, dataset_name, seed=1):
        ds = self.get_raw_dataset(dataset_name, seed=seed)
        ds = ds\
            .map(lambda example: load_audio(example, keys.ORIGINAL_SOURCE, keys.RIR))\
            .map(lambda example: mms_msg.simulation.reverberant.reverberant_scenario_map_fn(example, channel_slice=1))\
            .map(white_microphone_noise)\
            .map(load_noise)

        ds = ds.map(merge_activity)
        if dataset_name == 'train':
            ds = ds.prefetch(num_workers=4, buffer_size=8)

        return ds
    
    def get_dataset_train(self):
        return self.get_dataset('train')
    
    def get_dataset_validation(self):
        return self.get_dataset('dev')
    
    def get_dataset_test(self):
        return self.get_dataset('test')

    def add_num_samples(self, example):
        if isinstance(example['num_samples'], dict):
            example['num_samples'] = example['num_samples']['observation']
        return example

class LibrispeechMeetingJson(JsonDatabase):
    def __init__(self, json_path: [str, Path] = MEETING_VAD_JSON_PATH):
        super().__init__(json_path)

    def get_dataset(self, name='dev'):
        ds = super().get_dataset(name)
        ds = ds\
            .map(lambda example: load_audio(example, keys.ORIGINAL_SOURCE, keys.RIR))\
            .map(lambda example: mms_msg.simulation.reverberant.reverberant_scenario_map_fn(example, channel_slice=1))\
            .map(white_microphone_noise)\
            .map(load_noise)\
            .map(merge_activity)\
            .map(drop_audio)\
            .map(self.add_num_samples)
        return ds
    
    def get_dataset_train(self):
        return self.get_dataset('train')
    
    def get_dataset_validation(self):
        return self.get_dataset('dev')
    
    def get_dataset_test(self):
        return self.get_dataset('test')
    
    def add_num_samples(self, example):
        if isinstance(example['num_samples'], dict):
            example['_num_samples'] = example['num_samples']
            example['num_samples'] = example['num_samples']['observation']
        return example

# Alias
LibrispeechJson = LibrispeechMeetingJson