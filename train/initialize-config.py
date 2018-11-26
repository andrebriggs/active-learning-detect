import os
import sys
import argparse
from configparser import ConfigParser, ExtendedInterpolation

def main(config_path):
    config_file = ConfigParser(interpolation=ExtendedInterpolation())
    #config_file.read('/Users/andrebriggs/Desktop/MyConfig.ini')
    config_file.read(config_path)

    for x in config_file.sections():
        #print(x)
        for k,v in config_file.items(x):
            os.environ[k] = v
            sys.stdout.write("{0}={1}\n".format(k,v))
    

# for section in config_file.sections():
#     print(section)
#     #for key,val in config_file[section]:
#         #print("{0}:{1}".format(key,val))

if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the config.ini file')
    args = parser.parse_args()
    main(args.config_path)