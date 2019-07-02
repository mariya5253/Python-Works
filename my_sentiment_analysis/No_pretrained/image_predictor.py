# if you haven't already, install the SDK with "pip install sightengine"
from sightengine.client import SightengineClient
client = SightengineClient('487625360', '6wHNhLnF6jCaoZcQAiix')
# Detect nudity, weapons, alcohol, drugs and faces in an image, along with image properties and type
output = client.check('wad').set_file('image1.jpg')
print(output)
    