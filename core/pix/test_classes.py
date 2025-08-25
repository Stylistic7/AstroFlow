from pix.classes import Image, ImageList


def test_image_list_contains(test_context):
  images = ImageList(
      [Image(filter='Red'),
       Image(filter='green'),
       Image(filter='b')])

  assert 'RGB' in images

  red_image = images['R']
  green_image = images['G']
  blue_image = images['B']

  assert red_image
  assert green_image
  assert blue_image
