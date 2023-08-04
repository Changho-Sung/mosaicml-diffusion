# Copyright 2022 MosaicML Diffusion authors
# SPDX-License-Identifier: Apache-2.0

"""Convert and upload LAION parquet shards."""

import multiprocessing as mp
import os
import time
import warnings
from argparse import ArgumentParser, Namespace
from io import BytesIO
from typing import List, Optional, Set, Union

import numpy as np
import wandb
import s3fs
import aiobotocore
import tarfile
import json
from PIL import Image
from pyarrow import parquet as pq
from streaming import MDSWriter
from loguru import logger

# Change PIL image size warnings to be errors
warnings.filterwarnings('error', module='PIL', message='Image size')

profile_session = aiobotocore.session.AioSession(profile='toonie')
s3 = s3fs.S3FileSystem(session=profile_session)

def parse_args() -> Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    args = ArgumentParser()
    args.add_argument('--path',
                      type=str,
                      required=True,
                      help='Local or Remote directory containing downloaded shards in parquet format.')
    args.add_argument('--local',
                      type=str,
                      default='./cache/sbu',
                      help='Local path to cache MDS-formatted shards to.')
    args.add_argument('--remote', type=str, default='', help='Remote path to upload MDS-formatted shards to.')
    args.add_argument('--keep_parquet',
                      action='store_true',
                      help='Whether to keep the parquet shards after conversion (about 10TB).')
    args.add_argument('--keep_cache',
                      action='store_true',
                      help='Whether to keep the local cache after conversion.')
    args.add_argument('--poll_interval',
                      type=float,
                      default=30,
                      help='Interval between polling for newly downloaded shards to process.')
    args.add_argument('--wait_before_start',
                      action='store_true',
                      help='Wait for poll_interval before starting. This is useful when download '
                      'download happens in parallel')
    args.add_argument('--bucketed',
                      action='store_true',
                      help='Create different buckets between [0, 64, 128, 256, 512, 768, 1024, inf] '
                      'resolutions in subfolders.')
    args.add_argument('--subfolder',
                      default=None,
                      help='Stores data inside each bucket subfolder. Useful for running multiple '
                      'parallel downloads.')
    # Add wandb arguments
    # args.add_argument('--wandb_disabled', action='store_true')
    # args.add_argument('--wandb_name', type=str, default='baseline')
    # args.add_argument('--wandb_project', type=str, default='laion-dataset')
    # args.add_argument('--wandb_entity', type=str, default='mosaic-ml')
    return args.parse_args()


def is_download_complete(path: str) -> bool:
    """Check if all parquet shards have been downloaded.

    Args:
        path (str): Local or Remote directory containing shards.

    Returns:
        bool: True if all shards have been downloaded.
    """
    if not s3.exists(path):
        logger.error(f'Path does not exist!!')
        return False
    return s3.exists(os.path.join(path, 'done'))

    # if not os.path.exists(path):
    #     logger.error(f'Path does not exist!!')
    #     return False
    # return os.path.exists(os.path.join(path, 'done'))


def filter_parquet_files(path: str, completed_parquets: Set, processing_parquets: Set) -> List:
    """List of parquet files to convert into MDS shards in sorted order.

    Args:
        path (str): Local or Remote directory containing shards.

    Returns:
        List[str]: Each parquet filename.
    """
    shards_to_process = []
    if not s3.exists(path):
        logger('Path does not exist!!')
        return shards_to_process
    # if not os.path.exists(path):
    #     logger.error(f'Path does not exist!!')
    #     return shards_to_process

    for filename in sorted(s3.ls(path)):
    # for filename in sorted(os.listdir(path)):
        # If _stats.json file is present, the parquet file has finished downloading
        if filename.endswith('_stats.json'):
            idx = os.path.basename(filename).split('_')[0]
            if idx not in completed_parquets and idx not in processing_parquets:
                shards_to_process.append(idx)

    return shards_to_process


def get_int(x: Union[float, int]) -> int:
    """Get an int field from pandas.

    Args:
        x (Union[float, int]): The pandas field.

    Returns:
        int: The normalized field.
    """
    if np.isnan(x):
        return 0
    else:
        return int(x)


def get_float(x: float) -> float:
    """Get a float field from pandas.

    Args:
        x (float): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x


def get_bytes(x: Optional[bytes]) -> bytes:
    """Get a bytes field from pandas.

    Args:
        x (bytes, optional): The pandas field.

    Returns:
        float: The normalized field.
    """
    return x or b''


def get_str(x: Optional[str]) -> str:
    """Get a str field from pandas.

    Args:
        x (str, optional): The pandas field.

    Returns:
        str: The normalized field.
    """
    return x or ''


def process_parquet(args, queue, writer, shard, completed_parquets, lower_res, upper_res, bucket_id):
    """Process a parquet file and upload to MDS."""
    # Open parquet file
    parquet_filename = os.path.join(args.path, f'{shard}.parquet')
    table = pq.read_table(parquet_filename, filesystem=s3)
    n_rows = table.num_rows
    table = table.to_pandas()

    # # Download .tar file and extract all files into local cache dicrectory
    # if not os.path.exists(os.path.join(args.local, shard)):
    #     os.makedirs(os.path.join(args.local, shard))
    # tar_filename = os.path.join(args.local, f'{shard}.tar')
    # if not os.path.exists(tar_filename):
    #     s3.get(os.path.join(args.path, f'{shard}.tar'), tar_filename)
    #     with tarfile.open(tar_filename, 'r') as tar:
    #         tar.extractall(os.path.join(args.local, shard))

    # Iterate through rows of parquet file
    for i in range(n_rows):
        x = table.iloc[i]

        # Only write samples that were successfully downloaded
        success = x['status'] == 'success'
        width, height = get_int(x['width']), get_int(x['height'])
        if success:
            # update x from json file
            json_filename = os.path.join(args.local, shard, f'{x["key"]}.json')
            try:
                with open(json_filename, 'r') as f:
                    x.update(json.load(f))
            except Exception as e:
                pass

            try:
                img_filename = os.path.join(args.local, shard, f'{x["key"]}.jpg')
                img = Image.open(img_filename)
                width, height = img.size
                # convert .jpg to binary
                img_binary = img.tobytes()
                x['jpg'] = img_binary
            except Exception as e:
                # print(e)
                logger.error(e)
                # if unable to decode image, set success to false
                success = False
        success &= lower_res <= min(width, height) < upper_res
        if success:
            sample = {
                'caption': get_str(x['caption']),
                'url': get_str(x['url']),
                'key': get_str(x['key']),
                'status': get_str(x['status']),
                'error_message': get_str(x['error_message']),
                'width': get_int(x['width']),
                'height': get_int(x['height']),
                'original_width': get_int(x['original_width']),
                'original_height': get_int(x['original_height']),
                'exif': get_str(x['exif']),
                'jpg': get_bytes(x['jpg']),
                'sha256': get_str(x['sha256']),
            }
            writer.write(sample)

    # Mark shard as done for this bucket
    queue.put((shard, bucket_id))
    # print(f'Process {process_id} for bucket {bucket_id} completed shard {shard}...')

    # Add shard to completed set
    completed_parquets.add(shard)


def convert_and_upload_shards(args: Namespace, queue, lock, lower_res: int, upper_res: int, bucket_id: int):
    """Process any newly downloaded shards."""
    columns = {
        'caption': 'str',
        'url': 'str',
        'key': 'str',
        'status': 'str',
        'error_message': 'str',
        'width': 'int32',
        'height': 'int32',
        'original_width': 'int32',
        'original_height': 'int32',
        'exif': 'str',
        'jpg': 'bytes',
        'sha256': 'str',
    }

    logger.info(f'Starting uploader processs for bucket {bucket_id}...')
    remote_path = os.path.join(args.remote, f'{lower_res}-{upper_res}')
    if args.subfolder is not None:
        remote_path = os.path.join(remote_path, args.subfolder)
    writer = MDSWriter(out=remote_path,
                       columns=columns,
                       compression=None,
                       hash=[],
                       size_limit=256 * (2**20),
                       max_workers=64)
    completed_parquets = set()
    processing_parquets = set()
    shards_to_process = filter_parquet_files(path=args.path, 
                                             completed_parquets=completed_parquets, processing_parquets=processing_parquets)
    while not is_download_complete(args.path) or len(shards_to_process) > 0:
        start_time = time.time()

        for shard in shards_to_process:
            processing_parquets.add(shard)
            # check tar file download is complete
            # Download .tar file and extract all files into local cache dicrectory
            if not os.path.exists(os.path.join(args.local, shard)):
                os.makedirs(os.path.join(args.local, shard))
            tar_filename = os.path.join(args.local, f'{shard}.tar')
            with lock:
                if not os.path.exists(tar_filename):
                    logger.info(f"Starting download of {shard}.tar...")
                    s3.get(os.path.join(args.path, f'{shard}.tar'), tar_filename)
                    logger.info(f"Done downloading {shard}.tar!")
                    logger.info(f"Extracting {shard}.tar...")
                    with tarfile.open(tar_filename, 'r') as tar:
                        tar.extractall(os.path.join(args.local, shard))
                    logger.info(f"Done extracting {shard}.tar!")

            process_parquet(args, queue, writer, shard, completed_parquets, lower_res, upper_res, bucket_id)

        elapsed_time = time.time() - start_time
        if elapsed_time < args.poll_interval:
            time.sleep(args.poll_interval - elapsed_time)

        shards_to_process = filter_parquet_files(path=args.path, completed_parquets=completed_parquets)

    writer.finish()
    logger.info(f'Finished uploader process for bucket {bucket_id}...')


def remove_shards(args: Namespace, queue, signal_queue, num_buckets) -> None:
    """Remove shards from local or remote directory as they are completed."""
    logger.info(f'Starting remover process...')

    # if not args.wandb_disabled:
    #     wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name)
    #     wandb.log({'cloudwriter/remove_path': args.remote})
    logger.info(f'cloudwriter/remove_path: {args.local}')

    start_time = time.time()
    completed_map = {}
    completed_count = 0
    while True:
        if not queue.empty():
            shard, bucket_id = queue.get()
            if shard not in completed_map:
                completed_map[shard] = set()
            completed_map[shard].add(bucket_id)
            if len(completed_map[shard]) == num_buckets:
                completed_count += 1
                # if not args.wandb_disabled:
                #     wandb.log({'cloudwriter/count': completed_count})
                logger.info(f'cloudwriter/count: {completed_count}')
                logger.info(
                    f'Shard {shard} finished. Completed {completed_count} shards in {time.time() - start_time} seconds')
                if not args.keep_parquet:
                    os.remove(os.path.join(args.path, f'{shard}.parquet'))
                    os.remove(os.path.join(args.path, f'{shard}_stats.json'))
                if not args.keep_cache:
                    os.remove(os.path.join(args.local, f'{shard}.tar'))
                    os.rmdir(os.path.join(args.local, shard))
                    logger.info(
                        f'Removed {shard}.tar and {shard} from local cache')
        else:
            time.sleep(1)

        # if not args.wandb_disabled:
        #     wandb.log({'finished': True})
        logger.info(f'finished: True')
        if not signal_queue.empty():
            break
    logger.info(f'Finished remover process...')


def main(args: Namespace) -> None:
    """Convert and upload shards as they are created.

    Args:
        args (Namespace): Command-line arguments.
    """
    queue = mp.Queue()
    signal_queue = mp.Queue()
    lock = mp.Lock()

    # Append 0 and "infinity" to bin resolutions
    if args.bucketed:
        bin_resolutions = [0, 64, 128, 256, 512, 768, 1024, 2**20]
    else:
        bin_resolutions = [0, 2**20]
    uploaders = []
    for bucket_id in range(len(bin_resolutions) - 1):
        uploader = mp.Process(target=convert_and_upload_shards,
                              args=(args, queue, lock, bin_resolutions[bucket_id], bin_resolutions[bucket_id + 1], bucket_id))
        uploader.start()
        uploaders.append(uploader)

    for uploader in uploaders:
        uploader.join()
    signal_queue.put(1)

    remove = mp.Process(target=remove_shards, args=(args, queue, signal_queue, len(bin_resolutions) - 1))
    remove.start()
    remove.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main(parse_args())
