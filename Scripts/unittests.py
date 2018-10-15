import unittest
import auxiliary

class AuxiliaryTestMethods(unittest.TestCase):
    def test_GetFilesInFolder(self):
        directory = '/home/selim/Dev/DeepLib/DeepLib'
        files = auxiliary.GetFilesInFolder(directory, '.png')
        self.assertEqual(len(files), 1)

if __name__ == '__main__':
    unittest.main()