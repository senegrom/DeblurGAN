"""Find full-resolution source images matching lower-resolution demo frames.

The comparison is deliberately coarse and compression-tolerant: every image is
converted to RGB and reduced to the same small thumbnail before mean-squared
pixel and edge errors are measured. Exact source frames remain strong matches
after GIF quantization or JPEG compression.
"""

import argparse
import heapq
import io
import os

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.ppm', '.tif', '.tiff', '.webp'}


def load_thumbnail(path, size):
    with Image.open(path) as image:
        image = image.convert('RGB').resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def thumbnail_from_bytes(data, size):
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert('RGB').resize(size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def edge_map(image):
    dx = image[:, 1:] - image[:, :-1]
    dy = image[1:] - image[:-1]
    return dx, dy


def score(candidate, query, query_edges):
    pixel_mse = float(np.mean((candidate - query) ** 2))
    candidate_edges = edge_map(candidate)
    edge_mse = 0.5 * (
        float(np.mean((candidate_edges[0] - query_edges[0]) ** 2))
        + float(np.mean((candidate_edges[1] - query_edges[1]) ** 2))
    )
    return pixel_mse + edge_mse, pixel_mse, edge_mse


def image_paths(root):
    for current_root, _, names in os.walk(root):
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                yield os.path.join(current_root, name)


def candidates(source, size):
    if source.lower().endswith('.lmdb'):
        try:
            import lmdb
        except ImportError as exc:
            raise SystemExit('LMDB input needs `python -m pip install lmdb`') from exc
        env = lmdb.open(source, readonly=True, lock=False, readahead=False,
                        meminit=False)
        try:
            with env.begin(write=False) as transaction:
                for key, data in transaction.cursor():
                    try:
                        thumbnail = thumbnail_from_bytes(data, size)
                    except Exception as exc:
                        print(f'[warn] skipping LMDB key {key!r}: {exc}')
                        continue
                    yield key.decode('utf-8', errors='replace'), thumbnail
        finally:
            env.close()
        return

    for path in image_paths(source):
        try:
            yield path, load_thumbnail(path, size)
        except Exception as exc:
            print(f'[warn] skipping {path}: {exc}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='directory of candidate images')
    parser.add_argument('--query', required=True, nargs='+', help='demo frames to match')
    parser.add_argument('--width', type=int, default=80, help='comparison thumbnail width')
    parser.add_argument('--height', type=int, default=45, help='comparison thumbnail height')
    parser.add_argument('--top', type=int, default=10, help='matches to print per query')
    args = parser.parse_args()

    size = (args.width, args.height)
    queries = {
        path: load_thumbnail(path, size)
        for path in args.query
    }
    query_edges = {path: edge_map(image) for path, image in queries.items()}
    best = {path: [] for path in queries}

    count = 0
    for path, candidate in candidates(args.source, size):
        count += 1
        for query_path, query in queries.items():
            values = score(candidate, query, query_edges[query_path])
            item = (-values[0], path, values)
            heap = best[query_path]
            if len(heap) < args.top:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    print(f'compared {count} candidate images')
    for query_path, heap in best.items():
        print(f'\n{query_path}')
        for _, path, values in sorted(heap, reverse=True):
            total, pixel_mse, edge_mse = values
            print(f'  {total:.8f}  pixel={pixel_mse:.8f}  edge={edge_mse:.8f}  {path}')


if __name__ == '__main__':
    main()
