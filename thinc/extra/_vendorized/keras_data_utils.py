# https://raw.githubusercontent.com/fchollet/keras/master/keras/utils/data_utils.py
# Copyright Francois Chollet, Google, others (2015)
# Under MIT license
import tarfile
import zipfile
import os
import shutil
import hashlib
from urllib.error import URLError, HTTPError
from urllib.request import urlretrieve

from .keras_generic_utils import Progbar


def get_file(
    fname, origin, untar=False, unzip=False, md5_hash=None, cache_subdir="datasets"
):
    """Downloads a file from a URL if it not already in the cache.

    Passing the MD5 hash will verify the file after download as well as if it is already present in the cache.

    # Arguments
        fname: name of the file
        origin: original URL of the file
        untar: boolean, whether the file should be decompressed
        md5_hash: MD5 hash of the file for verification
        cache_subdir: directory being used as the cache

    # Returns
        Path to the downloaded file
    """
    datadir_base = os.path.expanduser(os.path.join("~", ".keras"))
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar or unzip:
        untar_fpath = os.path.join(datadir, fname)
        if unzip:
            fpath = untar_fpath + ".zip"
        else:
            fpath = untar_fpath + ".tar.gz"
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # file found; verify integrity if a hash was provided
        if md5_hash is not None:
            if not validate_file(fpath, md5_hash):
                print(
                    "A local file was found, but it seems to be "
                    "incomplete or outdated."
                )
                download = True
    else:
        download = True

    if download:
        print("Downloading data from", origin)
        global progbar
        progbar = None

        def dl_progress(count, block_size, total_size):
            global progbar
            if progbar is None:
                progbar = Progbar(total_size)
            else:
                progbar.update(count * block_size)

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            print("Untaring file...")
            tfile = tarfile.open(fpath, "r:gz")
            try:
                tfile.extractall(path=datadir)
            except (Exception, KeyboardInterrupt) as e:
                if os.path.exists(untar_fpath):
                    if os.path.isfile(untar_fpath):
                        os.remove(untar_fpath)
                    else:
                        shutil.rmtree(untar_fpath)
                raise
            tfile.close()
        return untar_fpath
    elif unzip:
        if not os.path.exists(untar_fpath):
            print("Unzipping file...")
            with zipfile.ZipFile(fpath) as file_:
                try:
                    file_.extractall(path=datadir)
                except (Exception, KeyboardInterrupt) as e:
                    if os.path.exists(untar_fpath):
                        if os.path.isfile(untar_fpath):
                            os.remove(untar_fpath)
                        else:
                            shutil.rmtree(untar_fpath)
                    raise
        return untar_fpath

    return fpath


def validate_file(fpath, md5_hash):
    """Validates a file against a MD5 hash

    # Arguments
        fpath: path to the file being validated
        md5_hash: the MD5 hash being validated against

    # Returns
        Whether the file is valid
    """
    hasher = hashlib.md5()
    with open(fpath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    if str(hasher.hexdigest()) == str(md5_hash):
        return True
    else:
        return False
