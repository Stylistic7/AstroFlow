from previews.thumbnails import make_thumbnail_decorator
from context import context
from decorators.content_hasher import ContentHasher

thumbnail = make_thumbnail_decorator(cache=context,
                                     content_hasher=ContentHasher())
