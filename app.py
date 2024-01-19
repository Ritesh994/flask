from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from roboflow import Roboflow
from matplotlib.path import Path

app = Flask(__name__)

# Initialize Roboflow
rf = Roboflow(api_key="zpylwepWJURMjyaus8J7")
project = rf.workspace().project("room-lyc7i")
model = project.version(1).model

def create_mask_from_points(image_size, points):
    y, x = np.mgrid[:image_size[1], :image_size[0]]
    points = np.array(points)
    path = Path(points)
    mask = path.contains_points(np.vstack((x.ravel(), y.ravel())).T)
    mask = mask.reshape((image_size[1], image_size[0]))
    return mask

def process_image(room_image_path, texture_image_path):
    room_image = Image.open(room_image_path)
    texture_image = Image.open(texture_image_path)

    # JSON data
    json_data = model.predict(room_image_path).json()

    # Extract floor coordinates
    floor_data = json_data['predictions'][0]
    floor_points = floor_data['points']

    # Create a list of tuples for the points
    floor_coordinates = [(point['x'], point['y']) for point in floor_points]

    # Create mask for the floor
    mask = create_mask_from_points(room_image.size, floor_coordinates)

    # Get the dimensions of the room image
    room_width, room_height = room_image.size

    # Repeat the texture to cover the entire room
    repeated_texture = np.tile(np.array(texture_image), (room_height // texture_image.size[1] + 1, room_width // texture_image.size[0] + 1, 1))
    repeated_texture = repeated_texture[:room_height, :room_width, :]

    # Use the mask to combine the images
    room_with_texture = np.array(room_image)
    room_with_texture[mask] = repeated_texture[mask]

    # Convert back to an image
    room_with_texture_image = Image.fromarray(room_with_texture)

    # Save the result
    output_image_path = 'static/newfinal.jpg'  # Save in the static folder to serve via Flask
    room_with_texture_image.save(output_image_path)

    return output_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        room_image = request.files['room_image']
        texture_image = request.files['texture_image']

        room_image_path = 'uploads/' + room_image.filename
        texture_image_path = 'uploads/' + texture_image.filename

        room_image.save(room_image_path)
        texture_image.save(texture_image_path)

        output_image_path = process_image(room_image_path, texture_image_path)

        return render_template('index.html', result_image=output_image_path)

    return render_template('index.html')

