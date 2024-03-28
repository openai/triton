import importlib
import json
import os
import pickle
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


def default_cache_dir():
    return os.path.join(Path.home(), ".triton", "cache")


def default_override_dir():
    return os.path.join(Path.home(), ".triton", "override")


def default_dump_dir():
    return os.path.join(Path.home(), ".triton", "dump")


class CacheManager(ABC):

    def __init__(self, key):
        pass

    @abstractmethod
    def get_file(self, key: str) -> Optional[str]:
        pass

    @abstractmethod
    def put(self, key: str, data: bytes) -> str:
        pass

    @abstractmethod
    def put_group(self, key: str, files: Dict[str, bytes]) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_group(self, key: str) -> Optional[Dict[str, str]]:
        pass


class FileCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        self.key = key
        self.lock_path = None
        if dump:
            self.cache_dir = default_dump_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
            self.lock_path = os.path.join(self.cache_dir, "lock")
            os.makedirs(self.cache_dir, exist_ok=True)
        elif override:
            self.cache_dir = default_override_dir()
            self.cache_dir = os.path.join(self.cache_dir, self.key)
        else:
            # create cache directory if it doesn't exist
            self.cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()
            if self.cache_dir:
                self.cache_dir = os.path.join(self.cache_dir, self.key)
                self.lock_path = os.path.join(self.cache_dir, "lock")
                os.makedirs(self.cache_dir, exist_ok=True)
            else:
                raise RuntimeError("Could not create or locate cache dir")

    def _make_path(self, filename) -> str:
        return os.path.join(self.cache_dir, filename)

    def _has_file(self, filename) -> bool:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        return os.path.exists(self._make_path(filename))

    def get_file(self, filename) -> Optional[str]:
        if self._has_file(filename):
            return self._make_path(filename)
        else:
            return None

    def get_group(self, key: str) -> Optional[Dict[str, str]]:
        grp_filename = f"__grp__{key}.json"
        if not self._has_file(grp_filename):
            return None
        grp_filepath = self._make_path(grp_filename)
        with open(grp_filepath) as f:
            grp_data = json.load(f)
        child_paths = grp_data.get("child_paths", None)
        # Invalid group data.
        if child_paths is None:
            return None
        result = {}
        for c, p in child_paths.items():
            if os.path.exists(p):
                result[c] = p
        return result

    # Note a group of pushed files as being part of a group
    def _put_group_metadata(self, key: str, group: Dict[str, str]) -> str:
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        grp_contents = json.dumps({"child_paths": group}).encode("utf-8")
        grp_filename = f"__grp__{key}.json"
        return self.put(grp_filename, grp_contents)

    def put(self, filename: str, data: bytes) -> str:
        assert isinstance(data, bytes), f"{filename} data is not bytes: {type(data)}"
        if not self.cache_dir:
            raise RuntimeError("Could not create or locate cache dir")
        assert self.lock_path is not None
        filepath = self._make_path(filename)
        # Random ID to avoid any collisions
        rnd_id = random.randint(0, 1000000)
        # we use the PID in case a bunch of these around so we can see what PID made it
        pid = os.getpid()
        # use tempfile to be robust against program interruptions
        temp_path = f"{filepath}.tmp.pid_{pid}_{rnd_id}"
        with open(temp_path, mode="wb") as f:
            f.write(data)
        # Replace is guaranteed to be atomic on POSIX systems if it succeeds
        # so filepath cannot see a partial write
        os.replace(temp_path, filepath)
        return filepath

    def put_group(self, key, group: Dict[str, bytes]) -> Dict[str, str]:
        result = {}
        for name, data in group.items():
            result[name] = self.put(name, data)
        self._put_group_metadata(key, result)
        return result


class RemoteCacheBackend:
    """
    A backend implementation for accessing a remote/distributed cache.
    """

    def __init__(self, key: str):
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        pass

    @abstractmethod
    def put(self, key: str, data: bytes):
        pass


class RedisRemoteCacheBackend(RemoteCacheBackend):

    def __init__(self, key):
        import redis
        self._key = key
        self._key_fmt = os.environ.get("TRITON_REDIS_KEY_FORMAT", "triton:{key}:{filename}")
        self._redis = redis.Redis(
            host=os.environ.get("TRITON_REDIS_HOST", "localhost"),
            port=int(os.environ.get("TRITON_REDIS_PORT", 6379)),
        )

    def _get_key(self, filename: str) -> str:
        return self._key_fmt.format(key=self._key, filename=filename)

    def get(self, filename: str) -> Optional[bytes]:
        return self._redis.get(self._get_key(filename))

    def put(self, filename: str, data: bytes):
        self._redis.set(self._get_key(filename), data)


class RemoteCacheManager(CacheManager):

    def __init__(self, key, override=False, dump=False):
        # Setup backend pointed too by `TRITON_REMOTE_CACHE_BACKEND`.
        remote_cache_manager = os.environ["TRITON_REMOTE_CACHE_BACKEND"]
        module_path, clz_nme = remote_cache_manager.split(":")
        module = importlib.import_module(module_path)
        remote_cache_cls = getattr(module, clz_nme)
        self._backend = remote_cache_cls(key)

        self._override = override
        self._dump = dump

        # Use a `FileCacheManager` to materialize remote cache paths locally.
        self._file_cache_manager = FileCacheManager(key, override=override, dump=dump)

    def get_file(self, key: str) -> Optional[str]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_file(key)

        data = self._backend.get(key)
        if data is None:
            return None

        return self._file_cache_manager.put(key, data)

    def put(self, key: str, data: bytes) -> str:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put(key, data)

        self._backend.put(key, data)

        return self._file_cache_manager.put(key, data)

    def get_group(self, key: str) -> Optional[Dict[str, str]]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.get_group(filename)

        data = self._backend.get(key)
        if data is None:
            return None

        return self._file_cache_manager.put_group(key, pickle.loads(data))

    def put_group(self, key: str, group: Dict[str, bytes]) -> Dict[str, str]:
        # We don't handle the dump/override cases.
        if self._dump or self._override:
            return self._file_cache_manager.put_group(key, group)

        self._backend.put(key, pickle.dumps(group))

        return self._file_cache_manager.put_group(key, group)


__cache_cls = FileCacheManager
__cache_cls_nme = "DEFAULT"


def get_cache_manager(key) -> CacheManager:
    import os

    user_cache_manager = os.environ.get("TRITON_CACHE_MANAGER", None)
    global __cache_cls
    global __cache_cls_nme

    if user_cache_manager is not None and user_cache_manager != __cache_cls_nme:
        module_path, clz_nme = user_cache_manager.split(":")
        module = importlib.import_module(module_path)
        __cache_cls = getattr(module, clz_nme)
        __cache_cls_nme = user_cache_manager

    return __cache_cls(key)


def get_override_manager(key) -> CacheManager:
    return __cache_cls(key, override=True)


def get_dump_manager(key) -> CacheManager:
    return __cache_cls(key, dump=True)


def make_so_cache_key(version_hash, signature, constants, ids, **kwargs):
    # Get unique key for the compiled code
    signature = {k: 'ptr' if v[0] == '*' else v for k, v in signature.items()}
    key = f"{version_hash}-{''.join(signature.values())}-{constants}-{ids}"
    for kw in kwargs:
        key = f"{key}-{kwargs.get(kw)}"
    key = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return key
