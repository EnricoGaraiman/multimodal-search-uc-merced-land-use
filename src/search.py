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
    """
    Extract features

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @param image_path: path for image
    @type image_path: str
    @param model: model
    @type model: object
    @return: features
    @rtype: list
    """
    transform = transforms.Compose(
        [transforms.Resize((PARAMETERS['size'], PARAMETERS['size'])), transforms.ToTensor(), ])

    img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        model.eval()
        features = model(img)

    return features.squeeze().numpy()


def create_database(database_path):
    """
    Create database if not exist

    @param database_path: path for database
    @type database_path: str
    """
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute(
        'CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, filename TEXT, features BLOB, class VARCHAR(20))')
    conn.commit()
    conn.close()


def insert_image(PARAMETERS, database_path, image_path, model):
    """
    Insert image in database with relevant features

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @param database_path: path for database
    @type database_path: str
    @param image_path: path for image
    @type image_path: str
    @param model: model
    @type model: object
    """
    features = extract_features(PARAMETERS, image_path, model)

    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO images (filename, features, class) VALUES (?, ?, ?)',
                       (str(image_path), features.tobytes(), os.path.basename(os.path.dirname(image_path))))
        conn.commit()


def search_similar_images(PARAMETERS, database_path, query_image_path, model):
    """
    Search for similar images based on input image

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @param database_path: path for database
    @type database_path: str
    @param query_image_path: path for searched image
    @type query_image_path: str
    @param model: model
    @type model: object
    @return: similar images
    @rtype: list
    """
    query_features = extract_features(PARAMETERS, query_image_path, model)

    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, features, class FROM images')
        results = cursor.fetchall()

    similar_images = []

    for result in tqdm(results, desc='Searching similar images'):
        image_id, filename, stored_features, class_name = result
        stored_features = np.frombuffer(stored_features, dtype=np.float32)

        distance = round(np.linalg.norm(query_features - stored_features), 4)
        similar_images.append({'id': image_id, 'filename': filename, 'distance': distance, 'class': class_name})

    similar_images.sort(key=lambda x: x['distance'])

    return similar_images


def search(PARAMETERS, model, image):
    """
    Search for similar image

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @param model: model
    @type model: object
    @param image: searched image path
    @type image: str
    """
    img = Image.open(image)

    similar_images = search_similar_images(PARAMETERS, PARAMETERS['database'], image, model)

    top_3_images = similar_images[:3]

    plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(2, 3, height_ratios=[1, 1.5])

    ax_arr1 = plt.subplot(spec[0, 1])
    ax_arr1.imshow(img)
    ax_arr1.set_title('Searched image')

    for i, top_image in enumerate(top_3_images):
        img_sim = Image.open(top_image['filename'])
        ax_arr2 = plt.subplot(spec[1, i])
        ax_arr2.imshow(img_sim)
        ax_arr2.set_title(f"Most similar image\n"
                          f"Image ID: {top_image['id']},\n"
                          f"Class: {top_image['class']},\n"
                          f"Distance: {top_image['distance']}")

    plt.subplots_adjust(hspace=0.5)

    plt.savefig('results/search/search_engine_' + image.split('/')[-1].split('.')[0] + '.png')


def insert_all_images_from_database(PARAMETERS, model):
    """
    Insert all images in database

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @param model: model
    @type model: object
    """
    create_database(PARAMETERS['database'])

    for root, dirs, files in tqdm(os.walk(PARAMETERS['dataset_path'])):
        for filename in files:
            if filename.lower().endswith('.tif'):
                image_path = os.path.join(root, filename)
                insert_image(PARAMETERS, PARAMETERS['database'], image_path, model)


def search_similar_images_by_text(database_path, text_to_search):
    """
    Search similar images by text

    @param database_path: path for database
    @type database_path: str
    @param text_to_search: text to search
    @type text_to_search: str
    @return: similar images
    @rtype: list
    """
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, features, class FROM images')
        results = cursor.fetchall()

    similar_images = []

    for result in tqdm(results, desc='Searching similar images based on text'):
        image_id, filename, stored_features, class_name = result

        if class_name.lower() in text_to_search.lower():
            similar_images.append({'id': image_id, 'filename': filename, 'distance': None, 'class': class_name})

    return similar_images


def search_by_text(PARAMETERS, text_to_search):
    """
    Search by text

    @param PARAMETERS: global parameters
    @type PARAMETERS: dict
    @param text_to_search: text to search
    @type text_to_search: str
    """
    similar_images = search_similar_images_by_text(PARAMETERS['database'], text_to_search)

    top_3_images = similar_images[:3]

    plt.figure(figsize=(8, 6))
    spec = gridspec.GridSpec(1, 3)

    for i, top_image in enumerate(top_3_images):
        img_sim = Image.open(top_image['filename'])
        ax_arr2 = plt.subplot(spec[0, i])
        ax_arr2.imshow(img_sim)
        ax_arr2.set_title(f"Most similar image\n"
                          f"Image ID: {top_image['id']},\n"
                          f"Class: {top_image['class']}")

    plt.suptitle('Searched text: \n' + text_to_search)

    plt.savefig('results/search/search_engine_by_text_' + text_to_search + '.png')
