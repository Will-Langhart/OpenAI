import openai
import requests
import os
import sys
import argparse
import logging
from PIL import Image
from io import BytesIO
import json

# Setting up logging
logging.basicConfig(level=logging.INFO)

def load_config(config_path='config.json'):
    """
    Loads configuration from a JSON file.
    """
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f'Configuration file not found at {config_path}')
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f'Error parsing config file: {e}')
        sys.exit(1)

def generate_image(prompt, size="1024x1024", n=2, referenced_image_ids=None, format='png'):
    """
    Generates images using OpenAI's DALL·E based on the given prompt.
    """
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size,
            referenced_image_ids=referenced_image_ids
        )

        image_urls = []
        for data in response['data']:
            image_urls.append(data['url'])

        return image_urls

    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return []

def download_images(image_urls, folder='downloaded_images', format='png'):
    """
    Downloads images from the provided URLs.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    for idx, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            response.raise_for_status()

            image_path = os.path.join(folder, f'image_{idx}.{format}')
            with open(image_path, 'wb') as f:
                f.write(response.content)

            logging.info(f"Image {idx} downloaded successfully to {image_path}")

        except Exception as e:
            logging.error(f"Error downloading image {idx}: {e}")

def display_images(image_urls):
    """
    Displays the downloaded images.
    """
    for url in image_urls:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image.show()

def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate and download images using DALL·E.")
    parser.add_argument('prompt', type=str, help='Text prompt for image generation')
    parser.add_argument('--size', type=str, default='1024x1024', help='Image size (e.g., 1024x1024)')
    parser.add_argument('--number', type=int, default=2, help='Number of images to generate')
    parser.add_argument('--format', type=str, default='png', help='Image format (e.g., png, jpg)')
    parser.add_argument('--display', action='store_true', help='Display images after download')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)

    # Set API Key
    openai.api_key = config.get('api_key')

    images = generate_image(args.prompt, size=args.size, n=args.number, format=args.format)
    if images:
        download_images(images, format=args.format)
        if args.display:
            display_images(images)

if __name__ == "__main__":
    main()
