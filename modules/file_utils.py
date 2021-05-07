import json
import os
import time
import platform

from modules.dict_to_obj import DictToObj

PLATFORM_WINDOWS = 'Windows'

if platform.system() == PLATFORM_WINDOWS:
    # conda install -c anaconda pywin32
    import win32file, win32con, pywintypes
else:
    import fcntl


class FileUtils(object):

    @staticmethod
    def write_text_file(filepath, text, encoding='utf-8'):
        try:
            with open(filepath, 'w', encoding=encoding, errors="ignore") as fp:
                fp.write(text)
        except Exception as e:
            print(e)

    @staticmethod
    def lock_file(f):
        while True:
            try:
                if platform.system() == PLATFORM_WINDOWS:
                    break
                    hfile = win32file._get_osfhandle(f.fileno())
                    win32file.LockFileEx(hfile, win32con.LOCKFILE_FAIL_IMMEDIATELY | win32con.LOCKFILE_EXCLUSIVE_LOCK,
                                         0, 0xffff0000, pywintypes.OVERLAPPED())
                else:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except:
                time.sleep(0.1)

    @staticmethod
    def unlock_file(f):
        while True:
            try:
                if platform.system() == PLATFORM_WINDOWS:
                    break
                    hfile = win32file._get_osfhandle(f.fileno())
                    win32file.UnlockFileEx(hfile, 0, 0, 0xffff0000, pywintypes.OVERLAPPED())
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)
                break
            except:
                time.sleep(0.1)

    @staticmethod
    def deleteDir(dirPath, is_delete_dir_path = False):
        if os.path.exists(dirPath):
            try:
                deleteFiles = FileUtils.listSubFiles(dirPath)
                deleteDirs = FileUtils.listSubDirs(dirPath)

                for f in deleteFiles:
                    os.remove(f)
                for d in deleteDirs:
                    FileUtils.deleteDir(d, True)

                if len(deleteDirs) == 0:
                    if is_delete_dir_path:
                        os.rmdir(dirPath)
            except Exception as e:
                print(e)

    @staticmethod
    def createDir(dirPath):
        if not os.path.exists(dirPath):
            try:
                os.makedirs(dirPath)
            except Exception as e:
                print(e)

    @staticmethod
    def listSubDirs(dirPath):
        dirs = []
        if os.path.exists(dirPath):
            paths = os.listdir(dirPath)
            for each in paths:
                each_path = f'{dirPath}/{each}'
                if os.path.isdir(each_path):
                    dirs.append(each_path)
        return dirs

    @staticmethod
    def listSubFiles(dirPath):
        files = []
        if os.path.exists(dirPath):
            paths = os.listdir(dirPath)
            for each in paths:
                each_path = f'{dirPath}/{each}'
                if not os.path.isdir(each_path):
                    files.append(each_path)
        return files

    @staticmethod
    def readJSON(path):
        return FileUtils.loadJSON(path)

    @staticmethod
    def loadJSON(path):
        result = None
        if os.path.exists(path):
            with open(path, 'r') as fp:
                result = json.load(fp)
        return result

    @staticmethod
    def load_json_as_object(path):
        dict_ = FileUtils.loadJSON(path)
        return DictToObj(**dict_)

    @staticmethod
    def writeJSON(path, obj):
        with open(path, 'w') as fp:
            FileUtils.lock_file(fp)
            json.dump(obj, fp, indent=4)
            FileUtils.lock_file(fp)

    @staticmethod
    def saveJSON(path, obj):
        return FileUtils.writeJSON(path, obj)

    @staticmethod
    def readAllAsString(path, encoding=None):
        result = None
        if os.path.exists(path):
            with open(path, 'r', encoding=encoding) as fp:
                FileUtils.lock_file(fp)
                result = '\n'.join(fp.readlines())
                FileUtils.lock_file(fp)
        return result


    @staticmethod
    def readAllAsInt(path):
        result = FileUtils.readAllAsString(path)
        if result is not None:
            result = int(result)
        return result

    @staticmethod
    def readAllAsFloat(path):
        result = FileUtils.readAllAsString(path)
        if result is not None:
            result = float(result)
        return result