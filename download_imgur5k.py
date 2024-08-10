'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
IMGUR5K is shared as a set of image urls with annotations. This code downloads
th images and verifies the hash to the image to avoid data contamination.

Usage:
      python downloaad_imgur5k.py --dataset_info_dir <dir_with_annotaion_and_hashes> --output_dir <path_to_store_images>

Output:
     Images dowloaded to output_dir
     data_annotations.json : json file with image annotation mappings -> dowloaded to dataset_info_dir
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import argparse
import hashlib
import json
import numpy as np
import os
import requests

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Processing imgur5K dataset download...")
    parser.add_argument(
        "--dataset_info_dir",
        type=str,
        default="dataset_info",
        required=False,
        help="Directory with dataset information",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images",
        required=False,
        help="Directory path to download the image",
    )
    parser.add_argument(
        "--imgur_client_id",
        type=str,
        default="",
        required=True,
        help="Create one here: https://api.imgur.com/oauth2/addclient",
    )
    args = parser.parse_args()
    return args


# Image hash computed for image using md5..
def compute_image_hash(img_path):
    return hashlib.md5(open(img_path, 'rb').read()).hexdigest()


# Create a sub json based on split idx
def _create_split_json(anno_json, _split_idx):

    split_json = {}

    split_json['index_id'] = {}
    split_json['index_to_ann_map'] = {}
    split_json['ann_id'] = {}

    for _idx in _split_idx:
        # Check if the idx is not bad
        if _idx not in anno_json['index_id']:
            continue

        split_json['index_id'][_idx] = anno_json['index_id'][_idx]
        split_json['index_to_ann_map'][_idx] = anno_json['index_to_ann_map'][_idx]

        for ann_id in split_json['index_to_ann_map'][_idx]:
            split_json['ann_id'][ann_id] = anno_json['ann_id'][ann_id]

    return split_json


# Returns direct URL to image with given hash or empty string if unsuccessful
def fetch_image_url(image_hash, imgur_client_id):
    '''
    Returns direct URL to image with given hash or empty string if unsuccessful
    '''

    imgur_api_url = f'https://api.imgur.com/3/image/{image_hash}'

    api_response = requests.get(imgur_api_url, headers={'Authorization': f'Client-ID {imgur_client_id}'})
    if not api_response.ok:
        if api_response.status_code == 404:
            print(f"Image doesn't exist anymore (404 not found)")
        else:
            print(f"Unexpected status code: {api_response.status_code}\n")
        return ''
    
    response_data = api_response.json()['data']

    # We have to fix direct link to i.imgur.com because the link from API doesn't work smh:

    # e.g. If the image file ends with ,jpeg, the API will give .jpg instead and it won't work.
    image_type = response_data['type'].split('/')[1] # e.g. 'image/jpeg' -> 'jpeg'

    # e.g. 'https://i.imgur.com/ABCDEF.jpg', 'image/jpeg' -> 'https://i.imgur.com/ABCDEF.jpeg'
    image_url = '.'.join(response_data['link'].split('.')[:-1]) + '.' + image_type

    return image_url


def invalidate_url(image_hash, invalid_urls):
    '''
    I don't really want to mess with the URLs in the .lst file, so I'm just going
    to reuse the faulty `i.imgur.com/<hash>.jpg` links
    '''
    
    url = f'https://i.imgur.com/{image_hash}.jpg'
    invalid_urls.append(url)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a hash dictionary with image index and its correspond gt hash
    with open(f"{args.dataset_info_dir}/imgur5k_hashes.lst", "r", encoding="utf-8") as _H:
        hashes = _H.readlines()
        hash_dict = {}

        for hash in hashes:
            hash_dict[f"{hash.split()[0]}"] = f"{hash.split()[1]}"

    tot_evals = 0
    num_match = 0
    invalid_urls = []
    # Download the urls and save only the ones with valid hash o ensure underlying image has not changed
    for index in list(hash_dict.keys()):
        image_url = fetch_image_url(index, args.imgur_client_id)
        if image_url == '':
            continue

        # e.g. 'https://i.imgur.com/ABCDEF.jpeg' -> 'jpeg'
        image_type = image_url.split('.')[-1]

        print(image_url, image_type, invalid_urls)

        # User-Agent required otherwise 429
        img_data = requests.get(image_url, headers={'User-Agent': f'my bot 1.0'}).content
        
        if len(img_data) < 100:
            print(f"URL retrieval for {index} failed!!\n")
            invalidate_url(index, invalid_urls)
            continue
        with open(f'{args.output_dir}/{index}.{image_type}', 'wb') as handler:
            handler.write(img_data)

        compute_image_hash(f'{args.output_dir}/{index}.{image_type}')
        tot_evals += 1
        if hash_dict[index] != compute_image_hash(f'{args.output_dir}/{index}.{image_type}'):
            print(f"For IMG: {index}, ref hash: {hash_dict[index]} != cur hash: {compute_image_hash(f'{args.output_dir}/{index}.{image_type}')}")
            os.remove(f'{args.output_dir}/{index}.{image_type}')
            invalidate_url(index, invalid_urls)
            continue
        else:
            num_match += 1

    # Generate the final annotations file
    # Format: { "index_id" : {indexes}, "index_to_annotation_map" : { annotations ids for an index}, "annotation_id": { each annotation's info } }
    # Bounding boxes with '.' mean the annotations were not done for various reasons

    _F = np.loadtxt(f'{args.dataset_info_dir}/imgur5k_data.lst', delimiter="\t", dtype=np.str, encoding="utf-8")
    anno_json = {}

    anno_json['index_id'] = {}
    anno_json['index_to_ann_map'] = {}
    anno_json['ann_id'] = {}

    cur_index = ''
    for cnt, image_url in enumerate(_F[:,0]):
        if image_url in invalid_urls:
            continue

        index = image_url.split('/')[-1][:-4]
        if index != cur_index:
            anno_json['index_id'][index] = {'image_url': image_url, 'image_path': f'{args.output_dir}/{index}.jpg', 'image_hash': hash_dict[index]}
            anno_json['index_to_ann_map'][index] = []

        ann_id = f"{index}_{len(anno_json['index_to_ann_map'][index])}"
        anno_json['index_to_ann_map'][index].append(ann_id)
        anno_json['ann_id'][ann_id] = {'word': _F[cnt,2], 'bounding_box': _F[cnt,1]}

        cur_index = index

    json.dump(anno_json, open(f'{args.dataset_info_dir}/imgur5k_annotations.json', 'w'), indent=4)

    # Now split the annotations json in train, validation and test jsons
    splits = ['train', 'val', 'test']
    for split in splits:
        _split_idx = np.loadtxt(f'{args.dataset_info_dir}/{split}_index_ids.lst', delimiter="\n", dtype=np.str)
        split_json = _create_split_json(anno_json, _split_idx)
        json.dump(split_json, open(f'{args.dataset_info_dir}/imgur5k_annotations_{split}.json', 'w'), indent=4)

    print(f"MATCHES: {num_match}/{tot_evals}\n")


if __name__ == '__main__':
    main()
