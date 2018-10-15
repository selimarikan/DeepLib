import os

# Returns the absolute path of the files in a given directory
# Filters by file extension 
def GetFilesInFolder(directory, extension):
    files = os.listdir(directory)
    filteredFiles = [os.path.join(directory, file) 
        for file in files if file.endswith(extension)]
    return filteredFiles


if __name__ == '__main__':
    directory = '/home/selim/Dev/DeepLib/DeepLib'
    files = GetFilesInFolder(directory, '.png')