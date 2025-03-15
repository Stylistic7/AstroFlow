from pix.classes import Image, ImageList
from pix.context import add_images_to_save


def SaveAs(images: ImageList, output_filenames: list[str]) -> ImageList:
  print(f'### Entering SaveAs, images {images} '
        f'output_filename {output_filenames}')

  images = [
      Image(filename=output_filename,
            view_id=image.view_id,
            filter=image.filter,
            dirty=True)
      for image, output_filename in zip(images, output_filenames)
  ]

  add_images_to_save(images)

  print(f'### SaveAs returning {images}')
  return images
