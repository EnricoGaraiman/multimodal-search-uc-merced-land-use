import os

import torch
from PIL import Image
from matplotlib import gridspec
from torchvision import transforms
import sqlite3
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def extract_features(PARAMETERS, image_path, model):
    transform = transforms.Compose(
        [transforms.Resize((PARAMETERS['size'], PARAMETERS['size'])), transforms.ToTensor(), ])

    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        model.eval()
        features = model(img)

    return features.squeeze().numpy()


def create_database(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, filename TEXT, features BLOB, class VARCHAR(20))')
    conn.commit()
    conn.close()


def insert_image(PARAMETERS, database_path, image_path, model):
    features = extract_features(PARAMETERS, image_path, model)

    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO images (filename, features, class) VALUES (?, ?, ?)',
                       (str(image_path), features.tobytes(), image_path.split('/')[-2]))
        conn.commit()


def search_similar_images(PARAMETERS, database_path, query_image_path, model):
    query_features = extract_features(PARAMETERS, query_image_path, model)

    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, features, class FROM images')
        results = cursor.fetchall()

    similar_images = []

    for result in tqdm(results, desc='Searching similar images'):
        image_id, filename, stored_features, class_name = result
        stored_features = np.frombuffer(stored_features, dtype=np.float32)

        distance = np.linalg.norm(query_features - stored_features)
        similar_images.append({'id': image_id, 'filename': filename, 'distance': distance, 'class': class_name})

    similar_images.sort(key=lambda x: x['distance'])

    return similar_images


def search(PARAMETERS, model, image):
    img = Image.open(image)

    similar_images = search_similar_images(PARAMETERS, PARAMETERS['database'], image, model)

    sorted_images = sorted(similar_images, key=lambda x: x['distance'])
    top_3_images = sorted_images[:3]

    fig = plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(2, 3, height_ratios=[1, 1.5])

    axarr1 = plt.subplot(spec[0, 1])
    axarr1.imshow(img)
    axarr1.set_title('Searched image')

    for i, top_image in enumerate(top_3_images):
        img_sim = Image.open(top_image['filename'])
        axarr2 = plt.subplot(spec[1, i])
        axarr2.imshow(img_sim)
        axarr2.set_title(f"Most similar image\n"
                         f"Image ID: {top_image['id']},\n"
                         f"Class: {top_image['class']},\n"
                         f"Distance: {round(top_image['distance'], 2)}")

    plt.subplots_adjust(hspace=0.5)

    plt.savefig('results/search/search_engine_' + image.split('/')[-1].split('.png')[0] + '.png')


def insert_all_images_from_database(PARAMETERS, model):
    create_database(PARAMETERS['database'])

    for root, dirs, files in tqdm(os.walk(PARAMETERS['dataset_path'])):
        for filename in files:
            if filename.lower().endswith('.tif'):
                image_path = os.path.join(root, filename)
                insert_image(PARAMETERS, PARAMETERS['database'], image_path, model)
