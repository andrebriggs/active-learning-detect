import os

from fabric import *

os.environ["DSVM_SSH_KEY_PATH"] = "/Users/andrebriggs/.ssh/act-learn-key"
os.environ["DSVM_HOST"] = 'abrig@40.117.208.179'

def test_fabric():
    is_done = True
    try:      
        with get_connection() as c:
            result = c.run('ls -l', hide=True)
            msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            print(msg.format(result))
            c.local('git status')

            # result = c.run('cd repos && ls -l', hide=True)
            # msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            # print(msg.format(result))
    except Exception as e:
            print(str(e))
            is_done = False
    finally:
        return is_done

def make_directory():
    result = False
    try:      
        with get_connection() as c:
            result = c.run('mkdir repos', hide=False)
            msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            print(msg.format(result))
            result = True
    except Exception as e:
            print(str(e))
    finally:
        return result

def clone_tensorflow_models_repo():
    result = False
    try:      
        with get_connection() as c:
            result = c.run('git clone https://github.com/tensorflow/models.git repos/models', hide=True)
            msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            print(msg.format(result))
            c.run('cd repos/models/ && git status')
            result = True
    except Exception as e:
            print(str(e))
    finally:
        return result

def pip_install_tensorflow():
    result = False
    try:      
        with get_connection() as c:
            result = c.run('cd repos/models/ && pip install tensorflow', hide=True)
            msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            print(msg.format(result))
            result = True
    except Exception as e:
            print(str(e))
    finally:
        return result

def install_coco_api():
    result = False
    try:      
        with get_connection() as c:
            result = c.run('git clone https://github.com/cocodataset/cocoapi.git repos/cocoapi', hide=True)
            msg = "Ran {0.command!r} on {0.connection.host}, got stdout:\n{0.stdout}"
            print(msg.format(result))
            c.run('cd repos/cocoapi/PythonAPI/ && make')
            c.run('cd repos/cocoapi/PythonAPI/ && cp -r cp -r pycocotools  ~/repos/models/research/')
            result = True
    except Exception as e:
            print(str(e))
    finally:
        return result


def get_connection():
    #http://docs.fabfile.org/en/2.4/api/connection.html
    #with Connection(host='bob@104.42.172.28',connect_kwargs={"password": "guW7hADoyFYUGWZR86ZL"}) as c:
    return Connection(host=os.environ["DSVM_HOST"],connect_kwargs={"key_filename": os.environ['DSVM_SSH_KEY_PATH']})

#clone_tensorflow_models_repo()
#pip_install_tensorflow()
install_coco_api()


'''
def local_git():
    is_done = True

    try:
        with lcd('/Users/andrebriggs/Code/pyfunction/'):
            r = local('git status', capture=True)
            # Make the branch is not Upto date before doing further operations

            if 'Your branch is up-to-date with' not in r.stdout:
                local('git add .', capture=True)
                local('git commit -m Initial Commit')
                local('git push origin master')

    except Exception as e:
        print(str(e))
        is_done = False
    finally:
        return is_done
'''

'''
Pull Tensorflow Repo
Pull Active Learning Repo

Instructions from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#testing-the-installation
pip install tensorflow or pip install tensorflow
COCO API installation https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#coco-api-installation
Protobuf install https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#manual-protobuf-compiler-installation-and-usage
Set Python PATH https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath
Test install https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#testing-the-installation
'''