from cache.cached_decorator import make_cached_decorator
from previews.create_thumbnail import make_create_thumbnail_decorator
from previews.create_video import make_create_video_decorator
from profiling.measure_performance import make_measure_performance_decorator
from hashing import hash_call
from context import context, get_cache
from decorators.content_hasher import ContentHasher

cached = make_cached_decorator(cache_provider=get_cache,
                               content_hasher=ContentHasher(),
                               call_hasher=hash_call)

create_thumbnail = make_create_thumbnail_decorator(context)
create_video = make_create_video_decorator(context)

measure_performance = make_measure_performance_decorator(context)
