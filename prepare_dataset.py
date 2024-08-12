import cv2
import numpy as np
import json
import os


# Crops the image given details of bounding box
def crop_image(img, xc, yc, w, h, a):
    box = cv2.boxPoints(((xc, yc), (w, h), -a))  # Creates box contour with center coords, width height, and angle
    w, h = int(w), int(h)
    box = np.intp(box)  # Get points of 4 corners of box
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, h - 1],
                        [0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1]], dtype="float32")  # Straighten the image/box
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h))

    return warped


# annotations_path is the directory to the annotations json
# save_path is the desired directory to save the cropped images
def main(annotations_path: str, save_path: str):
    annotations: list[str]
    annotations = [
        annotations_path + '/imgur5k_annotations_train.json',
        annotations_path + '/imgur5k_annotations_val.json',
        annotations_path + '/imgur5k_annotations_test.json',
    ]

    outputs: list[str]
    outputs = [
        save_path + '/train',
        save_path + '/val',
        save_path + '/test'
    ]
    print('Begin cropping images...')
    for annotation_path, output_path in zip(annotations, outputs):
        words = {}
        annotation = json.load(open(annotation_path, 'r'))
        annotations2 = list(annotation['index_to_ann_map'].items())

        # Each index is mapped to an array of annot_ids, which are the individual words/parts to crop in an image
        for index_ids, annot_ids in annotations2:
            img_info = annotation['index_id'][index_ids]
            img = cv2.imread(img_info['image_path'])
            if img is None:
                print("Image not found")
                continue
            # Loop through all the words in that image
            for ann_id in annot_ids:
                info = annotation['ann_id'][ann_id]
                info['word'] = str(info['word'])
                if len(info['word']) == 0:
                    print("Word not found")
                    continue
                words[ann_id] = info['word']
                if os.path.exists(output_path + '/' + ann_id + '.jpg'):
                    print("Image already exists")
                    continue
                img_cropped = crop_image(img, *eval(info['bounding_box']))
                cv2.imwrite(output_path + '/' + ann_id + '.jpg', img_cropped)
        with open(output_path + '/words.json', 'w') as f:
            json.dump(words, f)  # Save a json of words as labels for each image
    print('Dataset preparation complete!')


if __name__ == '__main__':
    main()
