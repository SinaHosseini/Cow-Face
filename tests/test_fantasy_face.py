from fantasy_face import transparent_sticker
import cv2
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'fantasy_face')))


class TestFantasyFace(unittest.TestCase):
    def setUp(self):
        self.cow_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'input', 'cow.jpg'))
        self.image_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'input', 'test_image.jpg'))

        self.test_image = 255 * np.ones((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(self.image_path, self.test_image)

    def test_transparent_sticker(self):
        image_cow = cv2.imread(self.cow_path)
        my_image = cv2.imread(self.image_path)

        self.assertIsNotNone(image_cow, "Cow image not found!")
        self.assertIsNotNone(my_image, "Test image not found!")

        image_cow = cv2.resize(image_cow, (640, 640))
        my_image = cv2.resize(my_image, (640, 640))

        image_cow_ghost, image_cow_transparent = transparent_sticker(image_cow)
        convert_image = my_image * image_cow_ghost + image_cow_transparent

        output_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'output', 'animal_face.jpg'))
        cv2.imwrite(output_path, convert_image)

        self.assertTrue(os.path.exists(output_path),
                        "Output image not created!")

    def tearDown(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        output_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'output', 'animal_face.jpg'))
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == '__main__':
    unittest.main()
