import os

from context import cache

if __name__ == '__main__':
  if 'measurements' in cache:
    for k in cache['measurements']:
      print(f'measurements["{k}"]')

  # for k in cache.iterkeys():
  #   print(k)

  video_index = 0
  for k in cache.iterkeys():
    if k[0] == 'image_handling.thumbnail_video':
      video_filename = f'/tmp/video_{video_index}.mp4'
      open(video_filename, 'wb').write(cache[k])
      print(f'wrote video filename {video_filename}')
      video_index += 1
