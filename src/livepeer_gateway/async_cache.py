from __future__ import annotations

from collections import OrderedDict
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

_T = TypeVar("_T")


def async_lru_cache(
    maxsize: int,
) -> Callable[[Callable[..., Awaitable[_T]]], Callable[..., Awaitable[_T]]]:
    def decorator(func: Callable[..., Awaitable[_T]]) -> Callable[..., Awaitable[_T]]:
        cache: OrderedDict[tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]], _T] = OrderedDict()

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> _T:
            key = (args, tuple(sorted(kwargs.items())))
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached

            value = await func(*args, **kwargs)
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > maxsize:
                cache.popitem(last=False)
            return value

        wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
        return wrapper

    return decorator
