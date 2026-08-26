"""Export selected input/target image pairs from BasicSR-style LMDBs."""

import argparse
import os

import lmdb
from PIL import Image


def read_image_bytes(lmdb_path, key):
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False,
                    meminit=False)
    try:
        with env.begin(write=False) as transaction:
            data = transaction.get(key.encode('utf-8'))
    finally:
        env.close()
    if data is None:
        raise KeyError(f'{key!r} not found in {lmdb_path}')
    return data


def write_verified_png(data, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + '.tmp'
    with open(temporary, 'wb') as output:
        output.write(data)
    try:
        with Image.open(temporary) as image:
            image.verify()
        os.replace(temporary, destination)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-lmdb', required=True)
    parser.add_argument('--target-lmdb', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('keys', nargs='+')
    args = parser.parse_args()

    for key in args.keys:
        for label, lmdb_path in (
                ('input', args.input_lmdb), ('target', args.target_lmdb)):
            destination = os.path.join(args.output, label, key + '.png')
            write_verified_png(read_image_bytes(lmdb_path, key), destination)
            with Image.open(destination) as image:
                print(f'{destination}: {image.width}x{image.height} {image.mode}')


if __name__ == '__main__':
    main()
